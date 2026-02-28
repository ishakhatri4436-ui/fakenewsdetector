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
st.set_page_config(page_title="VeriLens AI | Isha", layout="wide", page_icon="🛡️")

# 2. THE ANALYZER (Cleaning function to improve model accuracy)
def clean_news(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W"," ",text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    return text

# 3. ADVANCED UI CSS (Custom Frontend)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #F8FAFC !important;
    }
    
    /* Header Bar */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Elegant Isha Signature */
    .isha-brand {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
    }

    /* Content Cards */
    .card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }

    /* Professional Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E40AF !important;
    }
    
    /* Styled Input Area */
    .stTextArea textarea {
        border: 2px solid #E2E8F0 !important;
        border-radius: 12px !important;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 5. FRONTEND HEADER
st.markdown("""
    <div class="main-header">
        <h1 class="isha-brand">VERILENS AI</h1>
        <p style="color: #64748B; font-weight: 500;">Intelligence by <span style="color: #3B82F6;">Isha</span></p>
    </div>
""", unsafe_allow_html=True)

# 6. MAIN INTERFACE
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Input Label
    st.markdown("<h3 style='margin-bottom:1rem; font-size:1.2rem;'>Article Forensics</h3>", unsafe_allow_html=True)
    news_text = st.text_area("", placeholder="Paste the news text here for deep analysis...", height=220, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("🚀 SCAN CONTENT")

    if analyze_btn:
        if news_text.strip():
            # Data Processing
            cleaned_text = clean_news(news_text)
            probs = model.predict_proba([cleaned_text])[0]
            p_fake, p_real = probs[0] * 100, probs[1] * 100
            sentiment = (TextBlob(news_text).sentiment.polarity + 1) * 50
            
            # RESULTS DASHBOARD
            st.markdown("<br><h3>Investigative Results</h3>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Authenticity", f"{p_real:.1f}%")
            m2.metric("Emotional Bias", f"{sentiment:.1f}%")
            m3.metric("Status", "VERIFIED" if p_real > 50 else "SUSPECT")
            
            st.markdown("---")
            
            # Visual Analytics
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(
                    x=['Fact Signal', 'Fake Signal', 'Bias'],
                    y=[p_real, p_fake, sentiment],
                    marker_color=['#10B981', '#EF4444', '#3B82F6'],
                    bordercolor="white",
                ))
                fig.update_layout(
                    template="plotly_white", 
                    height=350, 
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                # Wordcloud for context
                wc = WordCloud(background_color="white", width=500, height=350, colormap='Blues').generate(cleaned_text)
                fig_wc, ax = plt.subplots(facecolor='white')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)

            # Final Verdict Banner
            if p_real > 50:
                st.success(f"✅ HIGH CONFIDENCE: This report matches authentic news patterns.")
            else:
                st.error(f"⚠️ LOW CONFIDENCE: High probability of misinformation or biased reporting.")
        else:
            st.warning("Input required for analysis.")

    st.markdown('</div>', unsafe_allow_html=True)

# 7. FOOTER
st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #94A3B8;">
        VeriLens AI Ultra &copy; 2026 | Built by <b style="color: #64748B;">ISHA</b>
    </div>
""", unsafe_allow_html=True)
