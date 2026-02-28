import streamlit as st
import joblib
import plotly.graph_objects as go
import time
import random

st.set_page_config(page_title="TruthLens AI", layout="wide")

# ---------- LIGHT ANIMATED BACKGROUND ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #fdfbfb, #ebedee, #e3f2fd, #fff1f8);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}
@keyframes gradient {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

.glass {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(15px);
    padding: 30px;
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
    padding:12px 25px;
    border-radius:30px;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🔮 TruthLens AI</div>", unsafe_allow_html=True)

model = joblib.load("model.pkl")

st.markdown("<div class='glass'>", unsafe_allow_html=True)

text = st.text_area("📰 Paste News Article", height=250)

if st.button("⚡ Analyze News"):
    if text.strip() == "":
        st.warning("Please enter news text")
    else:
        with st.spinner("Scanning Reality Matrix..."):
            time.sleep(1.5)

            probs = model.predict_proba([text])[0]
            fake_prob = probs[0] * 100
            real_prob = probs[1] * 100

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=real_prob,
                title={'text': "Real News Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffb3b3"},
                        {'range': [40, 70], 'color': "#ffe699"},
                        {'range': [70, 100], 'color': "#b3ffcc"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric("Real News %", f"{real_prob:.1f}%")
            col2.metric("Fake News %", f"{fake_prob:.1f}%")

            if real_prob > fake_prob:
                st.success("✅ Looks like REAL news")
                st.balloons()
            else:
                st.error("🚨 Possibly FAKE news")

            # Fun creativity line
            quotes = [
                "Truth is the first casualty of misinformation.",
                "Always verify before you share.",
                "AI can help, but humans must think."
            ]
            st.info(random.choice(quotes))

st.markdown("</div>", unsafe_allow_html=True)
