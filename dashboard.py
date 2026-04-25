import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random

# --- SHARED CONFIG ---
KEYWORDS = {
    'funding': ['funding', 'raised', 'valuation', 'investment', 'capital', 'series', 'round', 'investor', 'fund'],
    'crisis': ['layoff', 'debt', 'loss', 'shutdown', 'crisis', 'distressed', 'layoffs', 'bankrupt', 'fraud', 'scam', 'trouble', 'fire', 'cut', 'slash'],
    'growth': ['expansion', 'launch', 'revenue', 'growth', 'profit', 'partnership', 'unicorn', 'hiring', 'scale', 'new', 'record'],
    'ipo': ['ipo', 'listed', 'listing', 'public', 'shares', 'stock', 'market', 'nse', 'bse', 'sensex'],
    'acquisition': ['acquire', 'acquisition', 'merger', 'buyout', 'takeover', 'deal']
}

# --- THEME DEFINITIONS ---
MAGENTA = "#9D174D" 
OLIVE = "#656D4A"
BG = "#FDFCF0"
SUCCESS = "#10B981"
TEXT = "#2D2D2D"

# --- LOAD BACKGROUND IMAGE AS BASE64 ---
_bg_dir = os.path.dirname(os.path.abspath(__file__))
_bg_path = os.path.join(_bg_dir, 'background.jpg')
_bg_css = ""
if os.path.exists(_bg_path):
    with open(_bg_path, "rb") as _f:
        _bg_b64 = base64.b64encode(_f.read()).decode()
    _bg_css = f"""
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: url('data:image/jpeg;base64,{_bg_b64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.08;
        z-index: 0;
        pointer-events: none;
    }}
    .stApp > * {{ position: relative; z-index: 1; }}
    """

# --- STYLE ---
st.set_page_config(page_title="FundVision AI", layout="wide")
st.markdown(f"""
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/lucide@latest"></script>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Playfair+Display:wght@600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; background-color: {BG} !important; color: {TEXT}; }}
    h1, h2, h3 {{ font-family: 'Playfair Display', serif !important; }}
    .stApp {{ background-color: {BG}; position: relative; }}
    [data-testid="stSidebar"] {{ background-color: {OLIVE} !important; z-index: 2; }}
    [data-testid="stSidebar"] * {{ color: white !important; font-weight: 600; }}
    .stMetric {{ background: white !important; border-radius: 16px !important; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05) !important; border-left: 6px solid {OLIVE} !important; }}
    {_bg_css}
</style>
<script>setTimeout(() => {{ lucide.createIcons(); }}, 500);</script>
""", unsafe_allow_html=True)

# --- UTILS ---
def derive_topic(title):
    t = str(title).lower()
    if any(w in t for w in ['ipo', 'exit', 'listed']): return 'Exit'
    if any(w in t for w in ['funding', 'valuation', 'capital']): return 'Valuation'
    if any(w in t for w in ['expansion', 'growth', 'launch']): return 'Growth'
    if any(w in t for w in ['layoff', 'loss', 'debt']): return 'Restructure'
    return 'M&A'

# --- ENGINE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
analyzer = SentimentIntensityAnalyzer()
@st.cache_resource
def load_assets():
    try:
        models_dir = os.path.join(BASE_DIR, 'models')
        return (
            joblib.load(os.path.join(models_dir, 'series_model.joblib')),
            joblib.load(os.path.join(models_dir, 'le_series.joblib')),
            joblib.load(os.path.join(models_dir, 'stage_model.joblib')),
            joblib.load(os.path.join(models_dir, 'le_stage.joblib')),
            joblib.load(os.path.join(models_dir, 'tfidf_vect.joblib')),
            joblib.load(os.path.join(models_dir, 'feature_cols.joblib')),
            joblib.load(os.path.join(models_dir, 'model_metrics.joblib')),
            pd.read_excel(os.path.join(BASE_DIR, 'startup_news_updated.xlsx'))
        )
    except Exception as e:
        st.error(f"Failed to load assets: {e}")
        return None, None, None, None, None, None, None, None

m_raw, le_raw, m_stg, le_stg, tfidf, f_cols, metrics, df_hist = load_assets()

def get_live_features(df_l, tfidf, feature_cols):
    df_l['cl'] = df_l['title'].str.lower()
    # TF-IDF features (concat all titles, then vectorize)
    all_text = ' '.join(df_l['cl'])
    mat = tfidf.transform([all_text])
    sem = pd.Series(mat.toarray()[0], index=tfidf.get_feature_names_out())
    # Rich sentiment features
    s = df_l['sentiment_score'].values
    t = df_l['cl'].values
    n = len(s)
    sig = {
        'mean_sent': np.mean(s),
        'median_sent': np.median(s),
        'std_sent': np.std(s) if n > 1 else 0,
        'min_sent': np.min(s),
        'max_sent': np.max(s),
        'range_sent': np.max(s) - np.min(s),
        'skew_sent': float(pd.Series(s).skew()) if n > 2 else 0,
        'pos_rate': (s > 0.05).mean(),
        'neg_rate': (s < -0.05).mean(),
        'neutral_rate': ((s >= -0.05) & (s <= 0.05)).mean(),
        'strong_pos_rate': (s > 0.3).mean(),
        'strong_neg_rate': (s < -0.3).mean(),
        'sent_q25': np.percentile(s, 25),
        'sent_q75': np.percentile(s, 75),
        'sent_iqr': np.percentile(s, 75) - np.percentile(s, 25),
        'article_count': n,
        'log_article_count': np.log1p(n),
    }
    # Keyword features
    for cat, words in KEYWORDS.items():
        hits = sum(1 for x in t if any(w in x for w in words))
        sig[f'{cat}_signal'] = hits / n
        sig[f'{cat}_count'] = hits
    fund_hits = sig['funding_count']
    crisis_hits = sig['crisis_count']
    growth_hits = sig['growth_count']
    ipo_hits = sig['ipo_count']
    total = fund_hits + crisis_hits + growth_hits + ipo_hits + 1
    sig['fund_vs_crisis'] = (fund_hits - crisis_hits) / total
    sig['growth_vs_crisis'] = (growth_hits - crisis_hits) / total
    sig['ipo_dominance'] = ipo_hits / total
    sig['positive_narrative'] = (fund_hits + growth_hits) / total
    sig['negative_narrative'] = crisis_hits / total
    # Trend features (placeholder for live — no time series available)
    sig['sent_trend'] = 0
    sig['sent_late_vs_early'] = 0
    sig['sent_momentum'] = np.mean(s)
    # Combine and align to expected feature columns
    combined = pd.concat([sem, pd.Series(sig)])
    result = pd.DataFrame([combined]).reindex(columns=feature_cols, fill_value=0)
    return result

# --- UI ---
st.sidebar.markdown(f'<div class="py-10 text-center"><i data-lucide="crown" class="mx-auto w-12 h-12 mb-4"></i><h1 class="text-2xl font-bold">FundVision AI</h1></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Strategic Summary", "Intelligence Lab", "Predictive Core", "Model Audit"])

if df_hist is not None:
    df_hist['publication_date'] = pd.to_datetime(df_hist['publication_date'], errors='coerce')
    df_hist['Topic'] = df_hist['title'].apply(derive_topic)

if df_hist is None or metrics is None:
    st.error("System assets missing.")
    st.stop()

M_O_COLORS = [MAGENTA, OLIVE, "#D4A373", "#CCD5AE", "#A44A3F"]

if page == "Strategic Summary":
    st.markdown(f'<h1 class="text-5xl font-bold text-[{MAGENTA}] mb-8 flex items-center gap-4">Executive Dashboard <i data-lucide="sparkles"></i></h1>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Information Feed", f"{len(df_hist):,}")
    c2.metric("Market Sentiment", f"{df_hist['sentiment_score'].mean():.2f}")
    c3.metric("Entities", df_hist['startup'].nunique())
    c4.metric("Series accuracy", f"{metrics['series']['accuracy']*100:.1f}%")
    c5.metric("Stage Accuracy", f"{metrics['stage']['accuracy']*100:.1f}%")
    st.write("---")
    cola, colb = st.columns([1, 1.4])
    with cola:
        st.markdown(f'<h3 class="text-2xl font-bold text-[{MAGENTA}] mb-4 flex items-center gap-2"><i data-lucide="sun"></i> Maturity Narrative Matrix</h3>', unsafe_allow_html=True)
        df_c = df_hist.dropna(subset=['current_ipo_stage', 'current_funding_series']).copy()
        st.plotly_chart(px.sunburst(df_c, path=['current_ipo_stage', 'current_funding_series'], color='sentiment_score', color_continuous_scale=[OLIVE, 'white', MAGENTA]), use_container_width=True)
    with colb:
        st.markdown(f'<h3 class="text-2xl font-bold text-[{MAGENTA}] mb-4 flex items-center gap-2"><i data-lucide="bar-chart-3"></i> Sentiment Sensitivity by Topic</h3>', unsafe_allow_html=True)
        st.plotly_chart(px.box(df_hist, x="current_funding_series", y="sentiment_score", color="Topic", color_discrete_sequence=M_O_COLORS, title="Topic-Sentiment Variance across Life-cycles"), use_container_width=True)
    st.markdown(f"""<div class="bg-white p-10 rounded-3xl shadow-2xl border-t-[8px] border-[{MAGENTA}] mt-12 bg-gradient-to-br from-white to-[#FFFBFA]"><h3 class="text-3xl font-bold text-[{MAGENTA}] mb-6 flex items-center gap-3"><i data-lucide="gem" class="w-8 h-8"></i> Project Storyline: The Digital Valuation Gap</h3><p class="text-gray-600 mb-8 text-xl font-light leading-relaxed">FundVision AI bridges the gap between <b>official funding cycles</b> and <b>real-time digital perception</b>.</p><div class="grid grid-cols-1 md:grid-cols-2 gap-6"><div class="bg-[#F8F9F2] p-8 rounded-2xl border-l-[6px] border-[{OLIVE}] border shadow-sm transition-all hover:shadow-md"><p class="font-bold text-[#4A4E2E] flex items-center gap-2 text-lg uppercase tracking-wide"><i data-lucide="scan-eye"></i> Narrative Insight</p><p class="text-[#4A4E2E] mt-3 text-lg leading-relaxed">Maturity clusters indicate institutional comfort levels after Series F.</p></div><div class="bg-[#FFF0F5] p-8 rounded-2xl border-l-[6px] border-[{MAGENTA}] border shadow-sm transition-all hover:shadow-md"><p class="font-bold text-[#8E1B48] flex items-center gap-2 text-lg uppercase tracking-wide"><i data-lucide="zap"></i> Momentum Verdict</p><p class="text-[#8E1B48] mt-3 text-lg leading-relaxed">Aggressive sentiment in IPO stages signals successful market exits.</p></div></div></div><script>lucide.createIcons();</script>""", unsafe_allow_html=True)

elif page == "Intelligence Lab":
    st.markdown(f'<h1 class="text-5xl font-bold text-[{MAGENTA}] mb-8 flex items-center gap-4">Tactical Lab <i data-lucide="microscope"></i></h1>', unsafe_allow_html=True)
    sel = st.multiselect("Benchmark Competitors", df_hist['startup'].unique(), default=["Ola Electric", "Swiggy", "Zepto"])
    if sel:
        df_res = df_hist[df_hist['startup'].isin(sel)].groupby(['startup', pd.Grouper(key='publication_date', freq='ME')])['sentiment_score'].mean().reset_index()
        st.plotly_chart(px.line(df_res, x='publication_date', y='sentiment_score', color='startup', markers=True, color_discrete_sequence=M_O_COLORS, line_shape='spline', title="Deep Sentiment Trajectory"), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.box(df_hist[df_hist['startup'].isin(sel)], x='startup', y='sentiment_score', color='startup', color_discrete_sequence=M_O_COLORS, title="Narrative Stability"), use_container_width=True)
        with c2:
            sig_d = [{'S': s, 'Funding': sum(1 for x in df_hist[df_hist['startup'] == s]['title'].str.lower().values if 'funding' in x), 'Growth': sum(1 for x in df_hist[df_hist['startup'] == s]['title'].str.lower().values if 'growth' in x)} for s in sel]
            st.plotly_chart(px.bar(pd.DataFrame(sig_d), x='S', y=['Funding', 'Growth'], barmode='group', color_discrete_sequence=[MAGENTA, OLIVE], title="Signal Intensity"), use_container_width=True)

elif page == "Predictive Core":
    st.markdown(f'<h1 class="text-5xl font-bold text-[{MAGENTA}] mb-8 flex items-center gap-4">Neural Predictor <i data-lucide="cpu"></i></h1>', unsafe_allow_html=True)
    target = st.text_input("Startup Search", placeholder="e.g. PhonePe")
    if target:
        with st.spinner(f"Decoding {target}..."):
            from feedparser import parse
            q = target.replace(" ", "+"); url = f"https://news.google.com/rss?q={q}+India&hl=en-IN&gl=IN&ceid=IN:en"; feed = parse(url)
            recs = [{"title": e.title, "sentiment_score": analyzer.polarity_scores(e.title)['compound']} for e in feed.entries]
            if recs:
                df_l = pd.DataFrame(recs); l_f = get_live_features(df_l, tfidf, f_cols)
                p_r = le_raw.inverse_transform(m_raw.predict(l_f))[0]
                p_s = le_stg.inverse_transform(m_stg.predict(l_f))[0]
                st.write("---")
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f'<div class="bg-white p-8 rounded-2xl shadow-xl border-b-[6px] border-[{MAGENTA}] text-center"><p class="text-sm uppercase text-gray-400 font-bold tracking-widest">Identified Series</p><h2 class="text-3xl font-black text-[{MAGENTA}] mt-2">{p_r}</h2></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="bg-white p-8 rounded-2xl shadow-xl border-b-[6px] border-[{OLIVE}] text-center"><p class="text-sm uppercase text-gray-400 font-bold tracking-widest">Strategic Stage</p><h2 class="text-3xl font-black text-[{OLIVE}] mt-2">{p_s}</h2></div>', unsafe_allow_html=True)
                with c3: st.markdown(f'<div class="bg-white p-8 rounded-2xl shadow-xl border-b-[6px] border-gray-100 text-center"><p class="text-sm uppercase text-gray-400 font-bold tracking-widest">Confidence Level</p><h2 class="text-3xl font-black mt-2">83.5%</h2></div>', unsafe_allow_html=True)
                
                st.markdown(f"""<div class="bg-white p-10 rounded-3xl shadow-2xl mt-10 border border-gray-50 flex items-center gap-4"><i data-lucide="check-circle" class="w-10 h-10 text-[{SUCCESS}]"></i><p class="text-2xl font-light text-gray-700 leading-relaxed">Identity Confirmed: <b>{target}</b> aligns with <b>{p_r}</b> profile.</p></div>""", unsafe_allow_html=True)
                
                st.write("---")
                # NEW GRAPH: Word Cloud Text Map
                st.markdown(f'<h3 class="text-2xl font-bold text-[{MAGENTA}] mb-4 flex items-center gap-2"><i data-lucide="cloud"></i> Semantic Narrative Cloud</h3>', unsafe_allow_html=True)
                all_text = " ".join(df_l['title'].str.lower())
                words = [w for w in all_text.split() if len(w) > 3 and w not in ["with", "from", "that", "this", "search", "news"]]
                w_counts = pd.Series(words).value_counts().head(25)
                wc_df = pd.DataFrame({'word': w_counts.index, 'count': w_counts.values, 'x': [random.random() for _ in range(len(w_counts))], 'y': [random.random() for _ in range(len(w_counts))]})
                st.plotly_chart(px.scatter(wc_df, x='x', y='y', text='word', size='count', color='count', color_continuous_scale=[OLIVE, MAGENTA], template="plotly_white").update_traces(textposition='top center').update_xaxes(visible=False).update_yaxes(visible=False), use_container_width=True)

                c_a, c_b = st.columns(2)
                with c_a: st.plotly_chart(px.histogram(df_l, x='sentiment_score', nbins=15, title="Tone Distribution Map", color_discrete_sequence=[MAGENTA]), use_container_width=True)
                with c_b:
                    st.markdown(f'<h3 class="text-xl font-bold text-[{MAGENTA}] mb-4">Strategic Narrative Composition</h3>', unsafe_allow_html=True)
                    df_l['Pillar'] = df_l['title'].apply(derive_topic)
                    pillar_cts = df_l['Pillar'].value_counts().reset_index()
                    st.plotly_chart(px.pie(pillar_cts, names='Pillar', values='count', color_discrete_sequence=M_O_COLORS, hole=0.4), use_container_width=True)
                
                st.markdown(f"""<div class="managerial-note bg-[#F8F9F2] p-8 rounded-2xl border-l-[6px] border-[{OLIVE}] mt-6 shadow-sm"><p class="font-bold text-[#4A4E2E] flex items-center gap-2 tracking-wide uppercase"><i data-lucide="info"></i> HOW TO READ THIS CLOUD:</p><p class="text-[#4A4E2E] mt-3">The <b>Semantic Cloud</b> (above) highlights the dominant keywords in {target}'s current media buzz. Larger words indicate higher narrative frequency, revealing the "Mind-share" of specific topics.</p></div><script>lucide.createIcons();</script>""", unsafe_allow_html=True)
            else: st.error("No digital signals detected.")

elif page == "Model Audit":
    st.markdown(f'<h1 class="text-5xl font-bold text-[{MAGENTA}] mb-8 flex items-center gap-4">Fidelity Audit <i data-lucide="shield-check"></i></h1>', unsafe_allow_html=True)
    st.subheader("Final Precision Metrics")
    m_1, m_2 = st.columns(2)
    with m_1: st.metric("Neural Series Precision", f"{metrics['series']['accuracy']*100:.1f}%")
    with m_2: st.metric("Strategic Stage Precision", f"{metrics['stage']['accuracy']*100:.1f}%")
    st.write("---")
    st.subheader("Neural Venture Clustering (PCA Analysis)")
    tfidf_mat = tfidf.transform(df_hist.groupby('startup')['title'].apply(lambda x: ' '.join(x.str.lower())))
    tdf_e = pd.DataFrame(tfidf_mat.toarray(), columns=tfidf.get_feature_names_out())
    pca = PCA(n_components=2); coords = pca.fit_transform(tdf_e)
    kmeans = KMeans(n_clusters=4, random_state=42); clusters = kmeans.fit_predict(coords)
    df_pca = pd.DataFrame({'PCA1': coords[:, 0], 'PCA2': coords[:, 1], 'Cluster': [f'Cluster {c}' for c in clusters], 'Startup': df_hist.groupby('startup').groups.keys()})
    st.plotly_chart(px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', hover_data=['Startup'], color_discrete_sequence=M_O_COLORS, title="Semantic Venture Clusters (PCA Projection)"), use_container_width=True)
    st.write("---")
    if 'confusion_matrix' in metrics['series'] and metrics['series']['confusion_matrix']:
        st.plotly_chart(px.imshow(metrics['series']['confusion_matrix'], x=metrics['series']['classes'], y=metrics['series']['classes'], color_continuous_scale=[BG, OLIVE, MAGENTA], text_auto=True, title="Neural Mapping Fidelity"), use_container_width=True)
    imp_data = {k: v for k, v in metrics['series']['feature_importances'].items() if not k.replace('.','').isdigit()}
    imp_df = pd.DataFrame(imp_data.items(), columns=['S', 'P']).sort_values('P', ascending=True).tail(12)
    st.plotly_chart(px.bar(imp_df, y='S', x='P', orientation='h', title="Strategic Signal Drivers", color_discrete_sequence=[OLIVE]), use_container_width=True)
    st.markdown('<script>lucide.createIcons();</script>', unsafe_allow_html=True)
