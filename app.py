# app.py
import streamlit as st
import joblib
import plotly.graph_objects as go
import time
import random
import pandas as pd

st.set_page_config(page_title="VeriLens Quantum AI", layout="wide")

# ---------- CREATIVE ANIMATED BACKGROUND ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #f0f8ff, #e0ffff, #f5f5dc, #faf0e6);
    background-size: 400% 400%;
    animation: gradient 20s ease infinite;
}
@keyframes gradient {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
.glass-card {
    background: rgba(255,255,255,0.5);
    backdrop-filter: blur(25px);
    padding: 40px;
    border-radius: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}
.title {
    text-align:center;
    font-size:64px;
    font-weight:900;
    color:#3a7bd5;
    text-shadow:0 0 20px rgba(58,123,213,0.5);
}
.stButton>button {
    background: linear-gradient(90deg,#ff9a9e,#fad0c4);
    border:none;
    padding:12px 30px;
    border-radius:30px;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🔮 VeriLens Quantum AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#333;font-size:18px;'>High-Fidelity Real vs Fake News Detector</p>", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
# Make sure you have a trained TF-IDF + LogisticRegression pipeline saved as model.pkl
model = joblib.load("model.pkl")

# ---------- INPUT ----------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
text = st.text_area("📰 Paste News Article Here", height=300, placeholder="Paste real or fake news article...")

if st.button("⚡ Verify News"):
    if not text.strip():
        st.warning("Please enter news text to analyze!")
    else:
        with st.spinner("Analyzing the news quantum matrix..."):
            time.sleep(1.5)

            # ---------- SLIDE ANALYSIS ----------
            slides = [p.strip() for p in text.split("\n") if p.strip()]
            if not slides:
                slides = [text]

            slide_probs = []
            for slide in slides:
                probs = model.predict_proba([slide])[0]
                slide_probs.append(probs)

            avg_fake = sum([p[0] for p in slide_probs]) / len(slide_probs) * 100
            avg_real = sum([p[1] for p in slide_probs]) / len(slide_probs) * 100

            # ---------- GAUGE CHART ----------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_real,
                title={'text': "Real News Probability"},
                gauge={
                    'axis': {'range':[0,100]},
                    'bar': {'color':'#3a7bd5'},
                    'steps':[
                        {'range':[0,40], 'color':'#ff9999'},
                        {'range':[40,70], 'color':'#ffe699'},
                        {'range':[70,100], 'color':'#b3ffcc'}
                    ],
                }
            ))
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color':'#333'})
            st.plotly_chart(fig, use_container_width=True)

            # ---------- METRICS ----------
            col1, col2 = st.columns(2)
            col1.metric("Real News %", f"{avg_real:.1f}%")
            col2.metric("Fake News %", f"{avg_fake:.1f}%")

            # ---------- VERDICT ----------
            if avg_real > avg_fake:
                st.success(f"✅ Likely REAL news ({avg_real:.1f}%)")
                st.balloons()
            else:
                st.error(f"🚨 Possibly FAKE news ({avg_fake:.1f}%)")

            # ---------- SLIDE-BY-SLIDE ANALYSIS ----------
            st.markdown("### 📄 Slide-by-Slide Analysis:")
            for i, probs in enumerate(slide_probs):
                color = "#3a7bd5" if probs[1]>probs[0] else "#ff4c4c"
                st.markdown(f"<p style='color:{color};'><b>Slide {i+1}:</b> Real {probs[1]*100:.1f}% | Fake {probs[0]*100:.1f}%</p>", unsafe_allow_html=True)
                st.write(slides[i])
                st.markdown("---")

            # ---------- FUN QUOTE ----------
            quotes = [
                "Always verify before sharing!",
                "Truth is the first casualty of misinformation.",
                "AI can help detect, but humans must think.",
                "Misinformation spreads faster than fact."
            ]
            st.info(random.choice(quotes))

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;margin-top:30px;'>VeriLens Quantum AI • 2026 • Powered by Isha Intelligence Labs</p>", unsafe_allow_html=True)
