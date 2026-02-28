import streamlit as st
import joblib
import pandas as pd
import re
import plotly.graph_objects as go

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens AI | Isha", layout="wide", page_icon="🛡️")

# 2. ADVANCED CSS (Clean Design)
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #6a11cb, #2575fc, #00d2ff, #3a7bd5);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Hide the top Streamlit menu for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Modern Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        color: white;
        margin-bottom: 20px;
    }

    /* Isha's Footer */
    .isha-footer {
        text-align: center;
        color: rgba(255,255,255,0.8);
        padding: 20px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 4. HEADER
st.markdown("<h1 style='text-align: center; color: white;'>🛡️ VeriLens AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; opacity: 0.8;'>Advanced Neural Fact-Checking System</p>", unsafe_allow_html=True)

# 5. MAIN INTERFACE
# This single container prevents "extra boxes" from appearing
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    news_input = st.text_area("Analyze News Article", placeholder="Paste article text here...", height=200)
    
    if st.button("🚀 VERIFY NOW"):
        if news_input.strip():
            # Prediction Logic
            prediction = model.predict([news_input])[0]
            probs = model.predict_proba([news_input])[0]
            
            # Assuming 0: Fake, 1: Real
            p_fake, p_real = probs[0] * 100, probs[1] * 100

            # Dashboard Columns
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = go.Figure(go.Indicator(
                    mode="gauge+number", value=p_real,
                    title={'text': "Credibility Score", 'font': {'color': 'white'}},
                    gauge={'axis': {'range': [0, 100], 'tickcolor': "white"}, 'bar': {'color': "#00ffcc"}}))
                fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number", value=p_fake,
                    title={'text': "Fake Risk", 'font': {'color': 'white'}},
                    gauge={'axis': {'range': [0, 100], 'tickcolor': "white"}, 'bar': {'color': "#ff3366"}}))
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250)
                st.plotly_chart(fig2, use_container_width=True)

            # Final Verdict
            if p_real > 50:
                st.success(f"✅ VERDICT: This news appears to be AUTHENTIC.")
            else:
                st.error(f"⚠️ VERDICT: High probability of MISINFORMATION detected.")
        else:
            st.info("Please enter text to begin analysis.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# 6. FOOTER
st.markdown("<div class='isha-footer'>Developed by Isha ❤️ VeriLens AI 2026</div>", unsafe_allow_html=True)
