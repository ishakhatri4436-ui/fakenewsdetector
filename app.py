import streamlit as st
import joblib
import pandas as pd
import re
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="VeriLens AI", layout="wide")

# Custom CSS for the beautiful background and cards
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f6d365, #fda085, #a1c4fd, #c2e9fb);
        background-size: 400% 400%;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    # This loads your model.pkl file
    return joblib.load('model.pkl')

model = load_model()

# --------------------------------------------------
# APP UI
# --------------------------------------------------
st.title("🔍 VeriLens AI: News Credibility Detector")
st.markdown("### Verify the authenticity of news articles instantly.")

with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    news_text = st.text_area("Paste the news article text here:", height=200)
    
    if st.button("Analyze Article"):
        if news_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Predict
            prediction = model.predict([news_text])[0]
            probs = model.predict_proba([news_text])[0]
            
            # Assuming 0 is FAKE and 1 is REAL based on your training script
            prob_fake = probs[0] * 100
            prob_real = probs[1] * 100

            # Show Results with Gauges
            col1, col2 = st.columns(2)

            with col1:
                fig1 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_real,
                    title={'text': "Credibility Score (%)"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00c853"}}
                ))
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_fake,
                    title={'text': "Fake Risk Score (%)"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ff4b4b"}}
                ))
                st.plotly_chart(fig2, use_container_width=True)

            if prediction == "FAKE":
                st.error("⚠️ ALERT: This article shows high indicators of being Fake News.")
            else:
                st.success("✅ VERIFIED: This article appears to be Credible News.")
                
    st.markdown("</div>", unsafe_allow_html=True)
