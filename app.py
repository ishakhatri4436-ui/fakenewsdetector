import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import string

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="VeriLens AI | Isha", layout="wide", page_icon="🎨")

# 2. THE ANALYZER (Cleaning function)
def clean_news(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W"," ",text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

# 3. VIBRANT UI CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); }
    
    .isha-brand {
        font-weight: 900;
        font-size: 3.5rem;
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }

    .card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #ffffff;
    }

    /* Colourful Metric Labels */
    [data-testid="stMetricLabel"] { font-size: 1.1rem !important; color: #4A5568 !important; }
</style>
""", unsafe_allow_html=True)

# 4. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 5. FRONTEND HEADER
st.markdown('<h1 class="isha-brand">VERILENS AI</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #718096; margin-bottom: 2rem;'>The Colourful Investigative Suite by Isha</p>", unsafe_allow_html=True)

# 6. MAIN TOOLBOX
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    input_text = st.text_area("📄 Paste content for deep-dive analysis:", placeholder="Paste here...", height=180)
    
    if st.button("🚀 UNLEASH TOOLS"):
        if input_text.strip():
            # Core Logic
            cleaned = clean_news(input_text)
            probs = model.predict_proba([cleaned])[0]
            p_fake, p_real = probs[0] * 100, probs[1] * 100
            
            # Sentiment Tool
            blob = TextBlob(input_text)
            sentiment = (blob.sentiment.polarity + 1) * 50
            subjectivity = blob.sentiment.subjectivity * 100

            # TOOLBOX TABS
            tab1, tab2, tab3 = st.tabs(["📊 Forensic Report", "🧠 Linguistic Insights", "🌈 Word Map"])

            with tab1:
                st.markdown("### Primary Scanner")
                m1, m2, m3 = st.columns(3)
                m1.metric("Authenticity", f"{p_real:.1f}%", delta="High" if p_real > 70 else "Low")
                m2.metric("Emotional Heat", f"{sentiment:.1f}%", delta="Intense" if sentiment > 70 else "Neutral")
                m3.metric("Bias Level", f"{subjectivity:.1f}%")

                fig = go.Figure(go.Bar(
                    x=['Fact Signal', 'Fake Signal', 'Sentiment'],
                    y=[p_real, p_fake, sentiment],
                    marker=dict(color=['#00dbde', '#fc00ff', '#ff0844'], line=dict(color='white', width=2))
                ))
                fig.update_layout(template="plotly_white", height=300, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.markdown("### Text DNA")
                words = input_text.split()
                unique_words = len(set(words))
                density = (unique_words / len(words)) * 100 if len(words) > 0 else 0
                
                c1, c2 = st.columns(2)
                c1.write(f"**Word Count:** {len(words)}")
                c1.write(f"**Lexical Diversity:** {density:.1f}%")
                c2.write(f"**Subjectivity Score:** {subjectivity:.1f}%")
                c2.write(f"**Reading Ease:** {'Professional' if len(words) > 50 else 'Casual'}")

            with tab3:
                # Colorful WordCloud
                wc = WordCloud(background_color="white", width=600, height=300, colormap='spring').generate(cleaned)
                fig_wc, ax = plt.subplots(facecolor='white')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)

            if p_real > 50:
                st.balloons()
                st.success("✨ Verdict: Content appears to be Authentic!")
            else:
                st.error("🚨 Warning: Potential Misinformation Detected.")
        else:
            st.warning("Please enter text to activate the tools.")
    st.markdown('</div>', unsafe_allow_html=True)

# 7. FOOTER
st.markdown("<div style='text-align: center; margin-top: 3rem; color: #A0AEC0;'>Designed with ❤️ by <b>ISHA</b> | 2026</div>", unsafe_allow_html=True)
