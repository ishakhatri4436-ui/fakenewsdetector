import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import string
import nltk

# --- NLTK DATA DOWNLOAD (Fixes MissingCorpusError) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('brown')
        nltk.download('punkt_tab')
    except Exception as e:
        st.error(f"Error downloading linguistic data: {e}")

download_nltk_data()

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens Forensic | Isha", layout="wide", page_icon="🕵️")

# 2. THE CLEANER (Essential for Accuracy)
def clean_news(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W"," ",text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

# 3. COLOURFUL FRONTEND CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); }
    .main-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #eef2f7;
    }
    .isha-header {
        font-weight: 900;
        background: linear-gradient(90deg, #FF512F 0%, #DD2476 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 4. LOAD MODEL
@st.cache_resource
def load_assets():
    # Ensure your model.pkl is in the same folder on GitHub
    return joblib.load('model.pkl')

model = load_assets()

# 5. HEADER
st.markdown('<h1 class="isha-header">VERILENS FORENSIC</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-weight: bold;'>Advanced AI Verification Suite by Isha</p>", unsafe_allow_html=True)

# 6. MAIN INTERFACE
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    input_text = st.text_area("📄 Analyze News Content", placeholder="Paste article text here for a deep-dive scan...", height=200)
    
    col_run, col_clear = st.columns([1, 5])
    run_btn = col_run.button("🚀 EXECUTE SCAN")

    if run_btn and input_text:
        # DATA LOGIC
        cleaned = clean_news(input_text)
        
        # MODEL PREDICTION
        probs = model.predict_proba([cleaned])[0]
        p_fake, p_real = probs[0] * 100, probs[1] * 100
        
        # NLP TOOLS
        blob = TextBlob(input_text)
        sentiment = (blob.sentiment.polarity + 1) * 50
        words = input_text.split()
        avg_word_len = sum(len(word) for word in words) / len(words) if words else 0

        # RESULTS TABS
        t1, t2, t3 = st.tabs(["📊 Probability Scan", "🧠 Writing Style", "🏷️ Named Entities"])

        with t1:
            st.markdown("### Authenticity Gauge")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = p_real,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#DD2476"},
                    'steps': [
                        {'range': [0, 45], 'color': "#ffccd5"},
                        {'range': [45, 75], 'color': "#fff4cc"},
                        {'range': [75, 100], 'color': "#d1e7dd"}]
                }
            ))
            fig.update_layout(height=350, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.markdown("### Linguistic DNA")
            c1, c2, c3 = st.columns(3)
            c1.metric("Emotional Bias", f"{sentiment:.1f}%")
            c2.metric("Word Density", f"{avg_word_len:.1f} avg len")
            c3.metric("Complexity", "Professional" if avg_word_len > 5.2 else "Casual")
            
            # WordCloud
            wc = WordCloud(background_color="white", colormap="magma", width=800, height=400).generate(cleaned)
            fig_wc, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)

        with t3:
            st.markdown("### Key Subjects Detected")
            # Using Noun Phrases as entities
            try:
                entities = list(set(blob.noun_phrases))
                if entities:
                    st.write("The AI identified the following key subjects:")
                    st.info(", ".join([e.title() for e in entities[:12]]))
                else:
                    st.write("No distinct subjects detected. The text might be too short.")
            except:
                st.warning("Linguistic analysis is warming up. Please try again in a moment.")

        # FINAL VERDICT
        st.markdown("---")
        # I adjusted the threshold to 45% to help 'Real' news pass more easily
        if p_real > 45:
            st.success(f"✅ VERDICT: LIKELY AUTHENTIC ({p_real:.1f}% Confidence)")
            st.balloons()
        else:
            st.error(f"⚠️ VERDICT: SUSPICIOUS CONTENT ({p_fake:.1f}% Misinformation Risk)")

    st.markdown('</div>', unsafe_allow_html=True)

# 7. FOOTER
st.markdown("<br><div style='text-align: center; color: #94a3b8; font-weight: bold;'>Forensic Suite | Intelligence by ISHA ❤️ 2026</div>", unsafe_allow_html=True)
