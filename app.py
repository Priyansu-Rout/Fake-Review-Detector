import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import datetime

st.set_page_config(
    page_title="FakeReview AI Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%); }
.hero-title { font-size:2.6rem; font-weight:800; background:linear-gradient(135deg,#667eea,#764ba2,#f093fb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub { color:#8892a4; font-size:1rem; margin-bottom:1.5rem; }
.metric-box { background:#1e2132; border:1px solid #2d3148; border-radius:12px; padding:1.2rem; text-align:center; }
.metric-value { font-size:2rem; font-weight:800; margin-bottom:0.2rem; }
.metric-label { color:#8892a4; font-size:0.78rem; font-weight:500; letter-spacing:0.04em; text-transform:uppercase; }
.badge-fake { background:#ff475722; color:#ff6b7a; border:1px solid #ff475744; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:700; }
.badge-genuine { background:#2ed57322; color:#2ed573; border:1px solid #2ed57344; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:700; }
.badge-suspicious { background:#ffa50222; color:#ffb730; border:1px solid #ffa50244; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:700; }
.flag-item { background:#2d1f0e; border-left:3px solid #ffa502; border-radius:0 8px 8px 0; padding:0.4rem 0.85rem; margin-bottom:0.3rem; color:#f5c842; font-size:0.82rem; }
section[data-testid="stSidebar"] { background:#151827; border-right:1px solid #2d3148; }
.stButton>button { background:linear-gradient(135deg,#667eea,#764ba2); color:white; border:none; border-radius:10px; font-weight:600; padding:0.6rem 2rem; width:100%; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True, max_length=512
    )
    zero_shot = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        truncation=True,
    )
    return sentiment, zero_shot

# ── LINGUISTIC ANALYSIS ───────────────────────────────────────────────────────
def linguistic_features(text):
    text_lower = text.lower()
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

    superlatives = ['best','greatest','amazing','perfect','excellent','outstanding',
                    'incredible','unbelievable','fantastic','wonderful','superb',
                    'exceptional','awesome','brilliant']
    generic_phrases = ['highly recommend','five stars','5 stars','must buy',
                       'do not hesitate','will not be disappointed','changed my life',
                       'best purchase','waste of money','exactly as described',
                       'fast shipping','great quality','very happy','love this product']

    word_freq = {}
    for w in words:
        w_clean = re.sub(r'[^a-zA-Z]', '', w.lower())
        if len(w_clean) > 4:
            word_freq[w_clean] = word_freq.get(w_clean, 0) + 1

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "exclamation_ratio": text.count('!') / max(len(words), 1),
        "caps_ratio": sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1),
        "superlative_ratio": sum(text_lower.count(s) for s in superlatives) / max(len(words), 1),
        "generic_count": sum(1 for p in generic_phrases if p in text_lower),
        "first_person_count": sum(text_lower.count(p) for p in [' i ',' my '," i've "," i'm ",' me ']),
        "repeat_ratio": max(word_freq.values(), default=0) / max(len(words), 1),
        "specificity_score": min(len(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', text)) / max(len(words), 1) * 10, 1.0),
    }

def compute_fake_score(features, sentiment_score, zs_fake, sensitivity_offset=0):
    flags = []
    score = 0.0

    if features["exclamation_ratio"] > 0.05:
        score += 15; flags.append("⚠️ Excessive exclamation marks — astroturfing pattern")
    if features["superlative_ratio"] > 0.08:
        score += 20; flags.append("⚠️ Heavy superlative use without specific details")
    if features["generic_count"] >= 2:
        score += 15; flags.append(f"⚠️ {features['generic_count']} generic review phrases found")
    elif features["generic_count"] == 1:
        score += 8
    if features["word_count"] < 12:
        score += 20; flags.append("⚠️ Very short — lacks detail and specificity")
    elif features["word_count"] < 20:
        score += 8
    if features["first_person_count"] == 0 and features["word_count"] > 15:
        score += 12; flags.append("⚠️ No personal experience — reads like a template")
    if features["repeat_ratio"] > 0.12:
        score += 10; flags.append("⚠️ Repetitive word usage — low linguistic diversity")
    if features["caps_ratio"] > 0.05:
        score += 10; flags.append("⚠️ ALL CAPS usage — emotional manipulation tactic")
    if zs_fake > 0.6:
        score += 20; flags.append("⚠️ AI zero-shot classifier flagged as fake")
    elif zs_fake > 0.45:
        score += 10
    if sentiment_score > 0.97 and features["specificity_score"] < 0.05:
        score += 12; flags.append("⚠️ Extreme positivity with zero specific product details")

    return min(max(score + sensitivity_offset, 0), 100), flags

def classify(fake_score):
    if fake_score >= 65:
        return "FAKE", '<span class="badge-fake">🚨 LIKELY FAKE</span>'
    elif fake_score >= 40:
        return "SUSPICIOUS", '<span class="badge-suspicious">⚠️ SUSPICIOUS</span>'
    return "GENUINE", '<span class="badge-genuine">✅ LIKELY GENUINE</span>'

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FakeReview AI")
    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")
    sensitivity = st.slider("Sensitivity", 1, 10, 5, help="Higher = stricter fake detection")
    sensitivity_offset = (sensitivity - 5) * 3

    st.markdown("---")
    st.markdown("### 📊 What We Detect")
    for item in ["Generic template phrases","Superlative overuse","Lack of specific details",
                 "No personal experience","Emotional manipulation","Repetitive language"]:
        st.markdown(f"🔴 {item}")
    for item in ["Specific product details","Personal context","Balanced feedback"]:
        st.markdown(f"🟢 {item}")

    st.markdown("---")
    st.markdown("### 🤖 Models Used")
    st.markdown("**DistilBERT SST-2** — Sentiment\n\n**DistilBERT MNLI** — Zero-shot\n\n**Custom NLP** — Linguistic rules")
    st.caption("✅ No API key · Runs fully offline · HuggingFace Transformers")

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔍 Fake Review Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Paste product reviews. AI analyzes authenticity using transformer models + linguistic patterns — no API key, fully offline.</div>', unsafe_allow_html=True)

with st.spinner("🧠 Loading AI models (downloads ~250MB on first run)..."):
    try:
        sentiment_pipe, zero_shot_pipe = load_models()
        st.success("✅ AI Models loaded — running fully offline!", icon="🤖")
        models_loaded = True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        models_loaded = False

st.markdown("---")

# Sample reviews
SAMPLES = {
    "🚨 Obvious Fake": "AMAZING PRODUCT!!! Best purchase I ever made!!! Highly recommend to everyone!!! Five stars!!! Do not hesitate to buy!!! Changed my life!!!",
    "⚠️ Suspicious": "Great product. Fast shipping. Exactly as described. Very happy with purchase. Will buy again. Highly recommend.",
    "✅ Genuine": "I've been using this blender for 3 months. The 1200W motor handles frozen fruits well but struggles with hard ice. The lid seal is tight — no leaks. Louder than my old Vitamix but solid value for the price.",
    "✅ Genuine Critical": "Bought for my daughter's birthday. Build quality is decent but the instruction manual is confusing — took 45 minutes to assemble. Customer service responded in 2 days when I had questions. 3/5 overall.",
    "🚨 Paid Promo": "OUTSTANDING EXCEPTIONAL INCREDIBLE product!!! BEST BEST BEST!!! Amazing quality perfect in every way MUST BUY NOW love this product five stars!!!",
}

tab1, tab2 = st.tabs(["📝 Paste Reviews", "📋 Sample Reviews"])

with tab2:
    selected = st.selectbox("Choose a sample:", list(SAMPLES.keys()))
    if st.button("📋 Load Sample"):
        st.session_state["input"] = SAMPLES[selected]

with tab1:
    reviews_input = st.text_area(
        "Paste one or more reviews (separate with a blank line):",
        value=st.session_state.get("input", ""),
        height=200,
        placeholder="Paste reviews here...\n\nSeparate multiple reviews with a blank line."
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_btn = st.button("🔍 Analyze Reviews", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["input"] = ""
            st.rerun()

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
if analyze_btn and reviews_input.strip() and models_loaded:
    reviews = [r.strip() for r in re.split(r'\n\s*\n', reviews_input.strip()) if r.strip()]

    if not reviews:
        st.warning("Enter at least one review.")
        st.stop()

    st.markdown("---")
    st.markdown(f"### 📊 Results — Analyzing {len(reviews)} Review{'s' if len(reviews)>1 else ''}")

    progress = st.progress(0)
    status = st.empty()
    results = []

    for i, review in enumerate(reviews):
        status.text(f"Analyzing review {i+1}/{len(reviews)}...")
        progress.progress((i+1)/len(reviews))

        # Sentiment
        try:
            sent = sentiment_pipe(review[:512])[0]
            s_label, s_conf = sent["label"], sent["score"]
            s_score = s_conf if s_label == "POSITIVE" else 1 - s_conf
        except:
            s_label, s_conf, s_score = "UNKNOWN", 0.5, 0.5

        # Zero-shot
        try:
            zs = zero_shot_pipe(review[:512], candidate_labels=["genuine customer review","fake or paid review","spam"])
            zs_scores = dict(zip(zs["labels"], zs["scores"]))
            zs_fake = zs_scores.get("fake or paid review", 0) + zs_scores.get("spam", 0) * 0.5
        except:
            zs_fake = 0.3

        features = linguistic_features(review)
        fake_score, flags = compute_fake_score(features, s_score, zs_fake, sensitivity_offset)
        label, badge = classify(fake_score)

        results.append({
            "review": review, "fake_score": fake_score, "trust_score": 100 - fake_score,
            "label": label, "badge": badge, "flags": flags, "features": features,
            "sentiment": s_label, "sentiment_conf": s_conf, "zs_fake": zs_fake,
        })

    progress.empty()
    status.empty()

    # ── SUMMARY METRICS ───────────────────────────────────────────────────────
    total = len(results)
    fake_n = sum(1 for r in results if r["label"] == "FAKE")
    susp_n = sum(1 for r in results if r["label"] == "SUSPICIOUS")
    genu_n = sum(1 for r in results if r["label"] == "GENUINE")
    avg_trust = sum(r["trust_score"] for r in results) / total
    trust_color = "#2ed573" if avg_trust >= 70 else "#ffa502" if avg_trust >= 45 else "#ff4757"

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, fake_n, "Fake Reviews", "#ff4757"),
        (c2, susp_n, "Suspicious", "#ffa502"),
        (c3, genu_n, "Genuine", "#2ed573"),
        (c4, f"{avg_trust:.0f}%", "Overall Trust", trust_color),
    ]:
        col.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS ───────────────────────────────────────────────────────────────
    if total > 1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(go.Pie(
                labels=["Genuine","Suspicious","Fake"], values=[genu_n, susp_n, fake_n],
                hole=0.6, marker=dict(colors=["#2ed573","#ffa502","#ff4757"], line=dict(color="#1e2132",width=2)),
                textfont=dict(color="white"),
            ))
            fig.update_layout(title=dict(text="Review Distribution", font=dict(color="white",size=14)),
                paper_bgcolor="#1e2132", plot_bgcolor="#1e2132", font=dict(color="white"),
                legend=dict(font=dict(color="white")), height=280, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            bar_colors = ["#ff4757" if r["label"]=="FAKE" else "#ffa502" if r["label"]=="SUSPICIOUS" else "#2ed573" for r in results]
            fig2 = go.Figure(go.Bar(
                x=[f"R{i+1}" for i in range(total)],
                y=[r["fake_score"] for r in results],
                marker=dict(color=bar_colors),
                text=[f"{r['fake_score']:.0f}%" for r in results],
                textposition="auto", textfont=dict(color="white"),
            ))
            fig2.update_layout(title=dict(text="Fake Score per Review", font=dict(color="white",size=14)),
                paper_bgcolor="#1e2132", plot_bgcolor="#1e2132", font=dict(color="white"),
                xaxis=dict(gridcolor="#2d3148",color="white"), yaxis=dict(gridcolor="#2d3148",color="white",range=[0,110]),
                height=280, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig2, use_container_width=True)

    # ── PER-REVIEW BREAKDOWN ──────────────────────────────────────────────────
    st.markdown("### 🧾 Review-by-Review Breakdown")

    for i, r in enumerate(results):
        score_color = "#ff4757" if r["fake_score"]>=65 else "#ffa502" if r["fake_score"]>=40 else "#2ed573"
        with st.expander(f"Review {i+1}  |  {r['label']}  |  Trust: {r['trust_score']:.0f}%", expanded=(total<=3)):
            col_text, col_gauge = st.columns([3, 1])
            with col_text:
                st.markdown("**Review Text:**")
                st.info(r["review"][:500] + ("..." if len(r["review"]) > 500 else ""))
            with col_gauge:
                st.markdown(f"""<div style="text-align:center;padding:1rem;background:#0f1117;border-radius:12px;margin-top:0.5rem">
                    <div style="font-size:2.5rem;font-weight:800;color:{score_color}">{r['fake_score']:.0f}</div>
                    <div style="color:#8892a4;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em">Fake Score</div>
                    <div style="margin-top:0.6rem">{r['badge']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**📊 Linguistic Stats**")
                st.markdown(f"- Words: `{r['features']['word_count']}`")
                st.markdown(f"- Sentences: `{r['features']['sentence_count']}`")
                st.markdown(f"- Generic phrases: `{r['features']['generic_count']}`")
                st.markdown(f"- Superlatives: `{r['features']['superlative_ratio']:.1%}`")
                st.markdown(f"- Specificity: `{r['features']['specificity_score']:.1%}`")
            with cb:
                st.markdown("**🤖 AI Model Scores**")
                st.markdown(f"- Sentiment: `{r['sentiment']}` ({r['sentiment_conf']:.1%})")
                st.markdown(f"- Zero-shot fake: `{r['zs_fake']:.1%}`")
                st.markdown(f"- Trust score: `{r['trust_score']:.0f}%`")
                st.markdown(f"- Fake score: `{r['fake_score']:.0f}/100`")
            with cc:
                st.markdown("**🚩 Red Flags**")
                if r["flags"]:
                    for flag in r["flags"]:
                        st.markdown(f'<div class="flag-item">{flag}</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ No significant red flags detected")

    # ── EXPORT ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    df = pd.DataFrame([{
        "Review #": i+1, "Text (preview)": r["review"][:150],
        "Verdict": r["label"], "Fake Score": f"{r['fake_score']:.0f}",
        "Trust Score": f"{r['trust_score']:.0f}%", "Red Flags": len(r["flags"]),
        "Sentiment": r["sentiment"], "Words": r["features"]["word_count"],
        "Generic Phrases": r["features"]["generic_count"],
    } for i, r in enumerate(results)])

    st.download_button(
        "⬇️ Download CSV Report", df.to_csv(index=False),
        f"fake_review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv"
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

elif analyze_btn and not reviews_input.strip():
    st.warning("⚠️ Please paste at least one review.")
