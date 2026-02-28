import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens AI | Isha", layout="wide", page_icon="🛡️")

# 2. LIGHT MODE CSS (High Readability)
st.markdown("""
<style>
    /* Soft Light Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #1a1a1b; /* Dark text for visibility */
    }

    /* Main Card - Frosted Glass Light Version */
    .main-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 40px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        color: #1a1a1b;
    }

    /* Input Box Styling */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1a1a1b !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
    }

    /* Metric Boxes */
    [data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-weight: 800 !important;
    }

    /* Custom Footer */
    .isha-footer {
        text-align: center;
        color: #4b5563;
        font-weight: 600;
        padding: 20px;
        letter-spacing: 1px;
    }
    
    h1, h2, h3, p {
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 4. APP HEADER
st.markdown("<h1 style='text-align: center;'>🛡️ VERILENS <span style='color: #2563eb;'>ULTRA</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Neural-Powered Forensic Analysis by Isha</p>", unsafe_allow_html=True)

# 5. INPUT SECTION
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    input_text = st.text_area("📄 Paste Article Text Below", placeholder="Analyze news content here...", height=200)
    
    if st.button("🚀 RUN SCAN"):
        if input_text.strip():
            # A. PREDICTION LOGIC
            probs = model.predict_proba([input_text])[0]
            p_fake, p_real = probs[0] * 100, probs[1] * 100
            
            # B. SENTIMENT ANALYSIS
            analysis = TextBlob(input_text)
            sentiment_score = (analysis.sentiment.polarity + 1) * 50 
            
            # C. TOP METRICS
            st.markdown("### 🔍 Analysis Results")
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
                    marker_color=['#10b981', '#ef4444', '#3b82f6']
                ))
                fig.update_layout(
                    title="Linguistic Signature", 
                    template="plotly_white", 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_cloud:
                # Use a darker colormap like 'plasma' or 'magma' for better visibility on white
                wc = WordCloud(background_color="white", width=400, height=250, colormap='magma').generate(input_text)
                fig_wc, ax = plt.subplots(facecolor='white')
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig_wc)

            # E. FINAL VERDICT
            if p_real > 50:
                st.success(f"✅ VERDICT: Likely Authentic Content")
            else:
                st.error(f"⚠️ VERDICT: High Risk of Misinformation")

        else:
            st.warning("Please enter some text to begin the analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

# 6. SIGNATURE FOOTER
st.markdown("<div class='isha-footer'>DEVELOPED BY ISHA ❤️ 2026</div>", unsafe_allow_html=True)
