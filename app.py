import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import os

# --- Helper function to create or load model ---
def load_or_train_model():
    if os.path.exists('news_model.pkl'):
        model = joblib.load('news_model.pkl')
        return model
    else:
        st.info("No trained model found. Training a new one...")

        # Replace this dummy data with your real balanced dataset
        data = pd.DataFrame({
            'text': [
                "The Bank of Italy warned the public about fraudulent deepfake content",
                "NASA discovers water on Mars",
                "Aliens control the government secretly",
                "COVID vaccines are safe and effective",
                "Bitcoin price will crash tomorrow"
            ],
            'label': ["FAKE", "REAL", "FAKE", "REAL", "FAKE"]
        })

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(data['text'], data['label'])
        joblib.dump(pipeline, 'news_model.pkl')
        st.success("Model trained and saved!")
        return pipeline

# Load or train model
model = load_or_train_model()

# --- Streamlit UI ---
st.set_page_config(page_title="🔮 TruthLens AI", layout="wide")
st.title("🔮 TruthLens AI — Real vs Fake News Detector")
st.write("Paste a news article below and check if it is REAL or FAKE.")

news_text = st.text_area("📰 Enter news article here", height=150)

if st.button("Analyze"):
    if news_text.strip() == "":
        st.warning("Please enter some news text!")
    else:
        pred = model.predict([news_text])[0]
        prob = model.predict_proba([news_text])[0]

        real_prob = prob[list(model.classes_).index("REAL")] * 100
        fake_prob = prob[list(model.classes_).index("FAKE")] * 100

        verdict_color = "green" if pred == "REAL" else "red"
        st.markdown(f"### 🚨 Verdict: **{pred}**", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:{verdict_color}'>REAL: {real_prob:.1f}% | FAKE: {fake_prob:.1f}%</h3>", unsafe_allow_html=True)

        st.subheader("📊 Probability Analysis")
        fig, ax = plt.subplots()
        ax.bar(["REAL", "FAKE"], [real_prob, fake_prob], color=["green", "red"])
        ax.set_ylim([0, 100])
        ax.set_ylabel("Probability (%)")
        for i, v in enumerate([real_prob, fake_prob]):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
        st.pyplot(fig)
