import streamlit as st
import joblib
import pandas as pd
import re
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="VeriLens AI by Isha", layout="wide", page_icon="🔍")

# --------------------------------------------------
# ENHANCED STYLING (Glassmorphism + Animations)
# --------------------------------------------------
st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass Effect Card */
    .main-card {
        background: rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 40px;
        margin-top: 20px;
        color: white;
    }

    /* Custom Footer for Isha */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        color: white;
        text-align: center;
        padding: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 2px;
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# --------------------------------------------------
# APP UI
# --------------------------------------------------
st.title("🔍 VeriLens AI")
st.subheader("Smart Fake News Detection System")

with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    news_text = st.text_area("Paste the news article text below to analyze its credibility:", height=250)
    
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyze_btn = st.button("✨ RUN ANALYSIS")

    if analyze_btn:
        if news_text.strip() == "":
            st.warning("Please enter some text first!")
        else:
            # Predict
            prediction = model.predict([news_text])[0]
            probs = model.predict_proba([news_text])[0]
            
            # 0=Fake, 1=Real
            prob_fake = probs[0] * 100
            prob_real = probs[1] * 100

            # Results Section
            st.markdown("---")
            c1, c2 = st.columns(2)

            with c1:
                fig_real = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_real,
                    title={'text': "Credibility Score", 'font': {'color': "white"}},
                    gauge={'axis': {'range': [0, 100], 'tickcolor': "white"},
                           'bar': {'color': "#00ff88"},
                           'steps': [{'range': [0, 50], 'color': "rgba(255,0,0,0.1)"}]}
                ))
                fig_real.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_real, use_container_width=True)

            with c2:
                fig_fake = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_fake,
                    title={'text': "Fake Risk Score", 'font': {'color': "white"}},
                    gauge={'axis': {'range': [0, 100], 'tickcolor': "white"},
                           'bar': {'color': "#ff3333"}}
                ))
                fig_fake.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_fake, use_container_width=True)

            if prob_real > 50:
                st.success(f"✅ THIS ARTICLE SEEMS AUTHENTIC")
            else:
                st.error(f"⚠️ HIGH RISK: THIS ARTICLE MAY BE FAKE")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER WITH YOUR NAME
# --------------------------------------------------
st.markdown("""
    <div class="footer">
        Developed with ❤️ by <b>ISHA</b> | VeriLens AI © 2026
    </div>
""", unsafe_allow_html=True)
