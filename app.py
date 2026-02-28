import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens AI | Isha", layout="wide", page_icon="🛡️")

# 2. ULTRA-READABLE LIGHT CSS
st.markdown("""
<style>
    /* Force Light Background and Dark Text everywhere */
    .stApp {
        background: #f0f2f6 !important;
        color: #111827 !important;
    }

    /* Target every possible text element to be DARK GREY/BLACK */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #111827 !important;
    }

    /* Fix the 'Extra Box' and Card Visibility */
    .main-card {
        background: white !important;
        border-radius: 15px;
        border: 1px solid #d1d5db;
        padding: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Input Box Visibility */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 2px solid #3b82f6 !important;
    }

    /* Metric boxes text fix */
    [data-testid="stMetricLabel"] p {
        color: #374151 !important;
        font-weight: bold !important;
    }
    
    /* Footer contrast */
    .isha-footer {
        text-align: center;
        padding: 20px;
        color: #6b7280 !important;
        font-weight: bold;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 4. HEADER
st.markdown("<h1 style='text-align: center;'>🛡️ VERILENS <span style='color: #2563eb;'>ULTRA</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-weight: bold;'>Neural-Powered Forensic Analysis by Isha</p>", unsafe_allow_html=True)

# 5. MAIN APP
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    input_text = st.text_area("Paste Article Text Below:", placeholder="Enter news content...", height=200)
    
    if st.button("🚀 RUN FORENSIC SCAN"):
        if input_text.strip():
            # Logic
            probs = model.predict_proba([input_text])[0]
            p_fake, p_real = probs[0] * 100, probs[1] * 100
            analysis = TextBlob(input_text)
            sentiment_score = (analysis.sentiment.polarity + 1) * 50 
            
            # Results Header
            st.markdown("### 🔍 Analysis Overview")
            m1, m2, m3 = st.columns(3)
            m1.metric("Credibility Score", f"{p_real:.1f}%")
            m2.metric("Emotional Bias", f"{sentiment_score:.1f}%")
            m3.metric("Status", "Authentic" if p_real > 50 else "Flagged")

            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(go.Bar(
                    x=['Authentic', 'Fake', 'Bias'],
                    y=[p_real, p_fake, sentiment_score],
                    marker_color=['#10b981', '#ef4444', '#3b82f6']
                ))
                fig.update_layout(template="plotly_white", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                wc = WordCloud(background_color="white", width=400, height=250, colormap='plasma').generate(input_text)
                fig_wc, ax = plt.subplots(facecolor='white')
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig_wc)

            if p_real > 50:
                st.success("✅ This content appears to be AUTHENTIC.")
            else:
                st.error("⚠️ HIGH RISK: Potential Misinformation Detected.")
        else:
            st.warning("Please provide text to analyze.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# 6. SIGNATURE
st.markdown("<div class='isha-footer'>DEVELOPED BY ISHA ❤️ 2026</div>", unsafe_allow_html=True)
