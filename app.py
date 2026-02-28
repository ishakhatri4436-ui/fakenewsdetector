# app.py
import streamlit as st
import joblib
import plotly.graph_objects as go
import time
import random

st.set_page_config(page_title="TruthLens AI", layout="wide")

# ---------- ANIMATED LIGHT BACKGROUND ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #fefefe, #f0f8ff, #fffaf0, #f5f5f5);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
@keyframes gradient {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

.glass-card {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(20px);
    padding: 40px;
    border-radius: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.title {
    text-align:center;
    font-size:60px;
    font-weight:900;
    color:#3a7bd5;
    text-shadow:0 0 15px rgba(58,123,213,0.3);
}

.stButton>button {
    background: linear-gradient(90deg,#ff9a9e,#fad0c4);
    border:none;
    padding:12px 30px;
    border-radius:30px;
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🔮 TruthLens AI</div>", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
# Make sure you have model.pkl (TF-IDF + LogisticRegression pipeline)
model = joblib.load("model.pkl")

# ---------- INPUT ----------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
text = st.text_area("📰 Paste News Article Here", height=250, placeholder="Paste real or fake news article...")

if st.button("⚡ Analyze News"):
    if not text.strip():
        st.warning("Please enter news text to analyze!")
    else:
        with st.spinner("Scanning Reality Matrix..."):
            time.sleep(1.5)  # simulate analysis time

            # ---------- PREDICTION ----------
            probs = model.predict_proba([text])[0]
            fake_prob = probs[0] * 100
            real_prob = probs[1] * 100

            # ---------- GAUGE CHART ----------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=real_prob,
                title={'text': "Real News Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3a7bd5"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffb3b3"},
                        {'range': [40, 70], 'color': "#ffe699"},
                        {'range': [70, 100], 'color': "#b3ffcc"}
                    ],
                }
            ))
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color': "#333"})
            st.plotly_chart(fig, use_container_width=True)

            # ---------- METRICS ----------
            col1, col2 = st.columns(2)
            col1.metric("Real News %", f"{real_prob:.1f}%")
            col2.metric("Fake News %", f"{fake_prob:.1f}%")

            # ---------- VERDICT ----------
            if real_prob > fake_prob:
                st.success(f"✅ Verdict: Likely REAL news ({real_prob:.1f}%)")
                st.balloons()
            else:
                st.error(f"🚨 Verdict: Possibly FAKE news ({fake_prob:.1f}%)")

            # ---------- FUN QUOTE ----------
            quotes = [
                "Always verify before sharing!",
                "Truth is the first casualty of misinformation.",
                "AI can help detect, but humans must think.",
                "Misinformation spreads faster than fact."
            ]
            st.info(random.choice(quotes))

st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<p style='text-align:center; color:#666; margin-top:30px;'>TruthLens AI • 2026 • Powered by Isha Intelligence Labs</p>", unsafe_allow_html=True)
