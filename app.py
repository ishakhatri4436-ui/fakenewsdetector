import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens AI Ultra | Isha", layout="wide", page_icon="🛡️")

# 2. DESIGNER CSS
st.markdown("""
<style>
    .stApp { 
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #00d2ff);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white; 
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 4. APP HEADER
st.markdown("<h1 style='text-align: center; color: white;'>🛡️ VERILENS <span style='color: #00d2ff;'>ULTRA</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a1c4fd;'>Neural-Powered Forensic Analysis by Isha</p>", unsafe_allow_html=True)

# 5. INPUT SECTION
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    input_text = st.text_area("📄 Paste Article Text", placeholder="Enter news here...", height=200)
    
    if st.button("🚀 RUN FORENSIC SCAN"):
        if input_text.strip():
            # A. PREDICTION LOGIC
            probs = model.predict_proba([input_text])[0]
            p_fake, p_real = probs[0] * 100, probs[1] * 100
            
            # B. SENTIMENT ANALYSIS
            analysis = TextBlob(input_text)
            sentiment_score = (analysis.sentiment.polarity + 1) * 50 
            
            # C. TOP METRICS
            st.markdown("### 🔍 Scan Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Credibility Score", f"{p_real:.1f}%")
            m2.metric("Emotional Bias", f"{sentiment_score:.1f}%")
            m3.metric("Verdict", "Authentic" if p_real > 50 else "Fake")

            # D. VISUAL CHARTS
            st.markdown("---")
            col_chart, col_cloud = st.columns(2)
            
            with col_chart:
                fig = go.Figure(go.Bar(
                    x=['Real Pattern', 'Fake Pattern', 'Bias'],
                    y=[p_real, p_fake, sentiment_score],
                    marker_color=['#00ffcc', '#ff3366', '#4facfe']
                ))
                fig.update_layout(title="Linguistic Signature", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col_cloud:
                # CHANGED 'ice' to 'viridis' to fix the error!
                wc = WordCloud(background_color=None, mode="RGBA", width=400, height=250, colormap='viridis').generate(input_text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig_wc)

            # E. FINAL VERDICT
            if p_real > 50:
                st.success(f"✅ VERDICT: Likely Authentic News")
            else:
                st.error(f"⚠️ VERDICT: High Risk of Misinformation")

        else:
            st.warning("Please enter text to analyze.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: grey;'>DEVELOPED BY ISHA ❤️ 2026</p>", unsafe_allow_html=True)
