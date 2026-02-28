import streamlit as st
import joblib
import pandas as pd
import re
import plotly.graph_objects as go

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens AI | Isha", layout="wide", page_icon="🛡️")

# 2. DESIGNER CSS (Glassmorphism + Neon Glow)
st.markdown("""
<style>
    /* Animated Dynamic Background */
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

    /* Floating Main Card */
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 50px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin-top: 20px;
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* Neon Button */
    div.stButton > button {
        background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.4);
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 25px rgba(79, 172, 254, 0.8);
        transform: scale(1.02);
    }

    /* Footer Style */
    .isha-footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 14px;
        letter-spacing: 3px;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 4. HEADER
st.markdown("<h1 style='text-align: center; color: white; font-size: 60px; font-weight: 800;'>VERILENS <span style='color: #00d2ff;'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a1c4fd; font-size: 20px;'>Trusted Fact-Checking Powered by Neural Intelligence</p>", unsafe_allow_html=True)

# 5. CONTENT
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Modern Input Area
    input_text = st.text_area("📄 News Content Analysis", placeholder="Paste your article here to verify its authenticity...", height=250)
    
    # Analyze Button
    if st.button("✨ ANALYZE CREDIBILITY"):
        if input_text:
            # Prediction Logic
            probs = model.predict_proba([input_text])[0]
            p_fake, p_real = probs[0] * 100, probs[1] * 100

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            # Gauge Charts
            with col1:
                fig1 = go.Figure(go.Indicator(
                    mode="gauge+number", value=p_real,
                    gauge={'axis': {'range': [0, 100], 'tickcolor': "white"}, 'bar': {'color': "#00ffcc"}},
                    title={'text': "Credibility", 'font': {'size': 24, 'color': 'white'}}))
                fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number", value=p_fake,
                    gauge={'axis': {'range': [0, 100], 'tickcolor': "white"}, 'bar': {'color': "#ff3366"}},
                    title={'text': "Fake Risk", 'font': {'size': 24, 'color': 'white'}}))
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
                st.plotly_chart(fig2, use_container_width=True)

            # Final Verdict Banner
            if p_real > 50:
                st.markdown(f"<div style='background: rgba(0,255,204,0.2); padding: 20px; border-radius: 15px; border: 1px solid #00ffcc; text-align: center; color: #00ffcc; font-size: 20px; font-weight: bold;'>✅ VERDICT: LIKELY AUTHENTIC NEWS</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background: rgba(255,51,102,0.2); padding: 20px; border-radius: 15px; border: 1px solid #ff3366; text-align: center; color: #ff3366; font-size: 20px; font-weight: bold;'>⚠️ VERDICT: POTENTIAL MISINFORMATION</div>", unsafe_allow_html=True)
        else:
            st.info("Please enter a news article to start.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# 6. SIGNATURE FOOTER
st.markdown("<div class='isha-footer'>ENGINEERED BY ISHA ❤️ 2026</div>", unsafe_allow_html=True)
