import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import string

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens Forensic | Isha", layout="wide", page_icon="🕵️")

# 2. THE CLEANER
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
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 4. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    return model

model = load_assets()

# 5. HEADER
st.markdown('<h1 class="isha-header">VERILENS FORENSIC</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Advanced Content Verification Suite by Isha</p>", unsafe_allow_html=True)

# 6. MAIN INTERFACE
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    input_text = st.text_area("📄 Paste Article Text", placeholder="Input news content for deep-dive scan...", height=200)
    
    col_run, col_clear = st.columns([1, 5])
    run_btn = col_run.button("🚀 START SCAN")

    if run_btn and input_text:
        # DATA LOGIC
        cleaned = clean_news(input_text)
        probs = model.predict_proba([cleaned])[0]
        p_fake, p_real = probs[0] * 100, probs[1] * 100
        
        # NLP TOOLS
        blob = TextBlob(input_text)
        sentiment = (blob.sentiment.polarity + 1) * 50
        words = input_text.split()
        avg_word_len = sum(len(word) for word in words) / len(words) if words else 0

        # RESULTS TABS
        t1, t2, t3 = st.tabs(["📊 Probability Scan", "🧠 Linguistic DNA", "🏷️ Entity Map"])

        with t1:
            st.markdown("### Forensic Probability")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = p_real,
                title = {'text': "Authenticity Score (%)", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#2575fc"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ff9a9e"},
                        {'range': [40, 70], 'color': "#f6d365"},
                        {'range': [70, 100], 'color': "#a1c4fd"}]
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.markdown("### Writing Style Analysis")
            c1, c2, c3 = st.columns(3)
            c1.metric("Emotional Bias", f"{sentiment:.1f}%")
            c2.metric("Lexical Density", f"{avg_word_len:.1f} chars/word")
            c3.metric("Complexity", "High" if avg_word_len > 5.5 else "Standard")
            
            # WordCloud
            wc = WordCloud(background_color="white", colormap="plasma", width=800, height=400).generate(cleaned)
            fig_wc, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig_wc)

        with t3:
            st.markdown("### Key Subjects Detected")
            # Simple Entity Extraction via Noun Phrases
            entities = list(set(blob.noun_phrases))
            if entities:
                st.write("The AI identified the following key subjects in this text:")
                st.write(", ".join(entities[:10]))
            else:
                st.write("No distinct named entities detected.")

        # FINAL VERDICT
        st.markdown("---")
        if p_real > 50:
            st.success(f"✅ VERDICT: High probability of AUTHENTICITY ({p_real:.1f}%)")
            st.info("**Next Step:** This article follows standard reporting patterns. Feel free to use it as a reference.")
        else:
            st.error(f"⚠️ VERDICT: SUSPICIOUS CONTENT detected ({p_fake:.1f}% Risk)")
            st.warning("**Action Required:** This text uses patterns common in misinformation. Check for 'Source Attribution' before sharing.")

    st.markdown('</div>', unsafe_allow_html=True)

# 7. FOOTER
st.markdown("<br><div style='text-align: center; color: #94a3b8;'>PRO Forensic Suite | Built by Isha ❤️ 2026</div>", unsafe_allow_html=True)
