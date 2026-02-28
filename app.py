import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens AI Pro | Isha", layout="wide", page_icon="🛡️")

# 2. PRO DESIGNER CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #00d2ff);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        color: white;
    }
    .feature-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 4. TITLE SECTION
st.markdown("<h1 style='text-align: center; color: white;'>🛡️ VERILENS <span style='color: #00d2ff;'>PRO</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a1c4fd;'>Advanced Forensic Text Analysis by Isha</p>", unsafe_allow_html=True)

# 5. INPUT AREA
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    input_text = st.text_area("📄 Input Article for Deep Analysis", placeholder="Paste text here...", height=200)
    
    col_btn, col_clear = st.columns([1, 4])
    analyze = col_btn.button("✨ DEEP SCAN")

    if analyze and input_text:
        # LOGIC
        probs = model.predict_proba([input_text])[0]
        p_fake, p_real = probs[0] * 100, probs[1] * 100
        confidence = max(p_fake, p_real)

        # ROW 1: GAUGES
        st.markdown("### 📊 Primary Indicators")
        c1, c2, c3 = st.columns([2, 2, 1])
        
        with c1:
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number", value=p_real,
                title={'text': "Credibility", 'font': {'color': 'white'}},
                gauge={'bar': {'color': "#00ffcc"}, 'axis': {'range': [0, 100]}}))
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250)
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number", value=p_fake,
                title={'text': "Fake Risk", 'font': {'color': 'white'}},
                gauge={'bar': {'color': "#ff3366"}, 'axis': {'range': [0, 100]}}))
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250)
            st.plotly_chart(fig2, use_container_width=True)
        
        with c3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.metric("AI Confidence", f"{confidence:.1f}%")
            st.progress(confidence / 100)

        # ROW 2: EXTRA FEATURES
        st.markdown("---")
        st.markdown("### 🧠 Deep Insights")
        feat1, feat2 = st.columns(2)

        with feat1:
            st.markdown("<div class='feature-box'><strong>🔠 Key Terms Visualized</strong>", unsafe_allow_html=True)
            wc = WordCloud(background_color=None, mode="RGBA", width=400, height=200, colormap='ice').generate(input_text)
            st.image(wc.to_array(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with feat2:
            st.markdown("<div class='feature-box'><strong>📝 Text Statistics</strong>", unsafe_allow_html=True)
            words = len(input_text.split())
            chars = len(input_text)
            complexity = "High" if words > 100 else "Low"
            st.write(f"Word Count: {words}")
            st.write(f"Character Count: {chars}")
            st.write(f"Linguistic Complexity: {complexity}")
            st.markdown("</div>", unsafe_allow_html=True)

        # FINAL VERDICT
        if p_real > 50:
            st.success("✅ SCAN COMPLETE: The content appears to be factually consistent.")
        else:
            st.error("⚠️ SCAN COMPLETE: Highly likely to be misinformation or biased content.")

    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("<br><p style='text-align: center; color: grey;'>PRO VERSION DEVELOPED BY ISHA ❤️ 2026</p>", unsafe_allow_html=True)
