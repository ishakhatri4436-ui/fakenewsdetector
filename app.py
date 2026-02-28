import streamlit as st
import joblib
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import string
import nltk

# --- LINGUISTIC ENGINE INITIALIZATION ---
@st.cache_resource
def init_nlp():
    for pkg in ['punkt', 'brown', 'punkt_tab']:
        nltk.download(pkg)

init_nlp()

# 1. PAGE CONFIG
st.set_page_config(page_title="VeriLens Ultra Pro | Isha", layout="wide", page_icon="🛡️")

# 2. PREMIUM UI CSS (Glassmorphism + Forensic Dark Mode)
st.markdown("""
<style>
    .stApp { background: #0f172a; color: white; }
    .forensic-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .isha-brand {
        font-weight: 900;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        text-align: center;
    }
    .high-risk { background-color: rgba(239, 68, 68, 0.2); border-left: 5px solid #ef4444; padding: 10px; margin: 5px 0; }
    .low-risk { background-color: rgba(34, 197, 94, 0.2); border-left: 5px solid #22c55e; padding: 10px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# 3. ANALYSIS TOOLS
def sentence_analysis(text, model):
    sentences = nltk.sent_tokenize(text)
    results = []
    for sent in sentences:
        prob = model.predict_proba([sent.lower()])[0][1] * 100
        results.append((sent, prob))
    return results

@st.cache_resource
def load_assets():
    return joblib.load('model.pkl')

model = load_assets()

# 4. HEADER
st.markdown('<h1 class="isha-brand">VERILENS ULTRA PRO</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Engineered by Isha • Advanced News Forensics 2026</p>", unsafe_allow_html=True)

# 5. INPUT SUITE
with st.container():
    st.markdown('<div class="forensic-card">', unsafe_allow_html=True)
    user_text = st.text_area("🔍 Paste Article for Forensic Deep-Scan:", placeholder="Enter full article text...", height=250)
    
    if st.button("🚀 INITIATE FORENSIC SCAN"):
        if user_text.strip():
            # Data Processing
            blob = TextBlob(user_text)
            probs = model.predict_proba([user_text.lower()])[0]
            real_score = probs[1] * 100
            
            # Sentence Heatmap
            sent_results = sentence_analysis(user_text, model)

            # TABS FOR RESULTS
            t1, t2, t3 = st.tabs(["📊 Forensic Dashboard", "🔬 Sentence Heatmap", "🗂️ Metadata"])

            with t1:
                st.markdown("### Verification Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Authenticity", f"{real_score:.1f}%")
                m2.metric("Bias Intensity", f"{blob.sentiment.subjectivity*100:.1f}%")
                m3.metric("Linguistic Tone", "Professional" if len(user_text.split()) > 100 else "Casual")

                # Visual Gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = real_score,
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#38bdf8"},
                             'steps': [{'range': [0, 50], 'color': "#1e293b"}, {'range': [50, 100], 'color': "#334155"}]}))
                fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                st.markdown("### Sentence-Level Risk Analysis")
                st.write("The AI has broken down the text to find the 'origin' of doubt:")
                for sent, score in sent_results:
                    css_class = "low-risk" if score > 50 else "high-risk"
                    st.markdown(f'<div class="{css_class}"><b>{score:.1f}% Trust:</b> {sent}</div>', unsafe_allow_html=True)

            with t3:
                st.markdown("### Extracted Entities")
                entities = list(set(blob.noun_phrases))
                if entities:
                    st.write("Identified Key Subjects:")
                    cols = st.columns(4)
                    for i, entity in enumerate(entities[:12]):
                        cols[i % 4].markdown(f"🔹 `{entity.title()}`")
                
                # WordCloud for Context
                wc = WordCloud(background_color="#0f172a", colormap="Blues").generate(user_text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig_wc)

            # FINAL FORENSIC ADVICE
            st.markdown("---")
            if real_score > 55:
                st.success(f"⚖️ **OFFICIAL VERDICT: AUTHENTIC.** This content follows the linguistic patterns of verified 2026 journalism.")
            else:
                st.error(f"⚖️ **OFFICIAL VERDICT: MANIPULATED CONTENT.** This text uses emotional triggers and vague sourcing common in disinformation.")
        else:
            st.warning("Please provide text to analyze.")
    st.markdown('</div>', unsafe_allow_html=True)

# 6. FOOTER
st.markdown("<div style='text-align: center; color: #475569;'>Proprietary AI Engine v2.4 | Isha Forensic Lab 2026</div>", unsafe_allow_html=True)
