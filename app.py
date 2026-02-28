import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens AI Ultra | Isha", layout="wide", page_icon="🛡️")

# 2. DESIGNER CSS (Modern Dark Mode with Mesh Background)
st.markdown("""
<style>
    .stApp { 
        background-color: #080a12;
        background-image: 
            radial-gradient(at 0% 0%, rgba(26, 35, 126, 0.4) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(0, 210, 255, 0.2) 0px, transparent 50%);
        background-attachment: fixed;
        color: white; 
    }
    .main-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    }
    .stMetric { 
        background: rgba(255,255,255,0.05); 
        padding: 15px; 
        border-radius: 15px; 
        border-left: 5px solid #00d2ff; 
    }
    /* Custom Button Glow */
    div.stButton > button {
        background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.6);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 4. APP HEADER
st.markdown("<h1 style='text-align: center; color: white; font-size: 50px;'>🛡️ VERILENS <span style='color: #00d2ff;'>ULTRA</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a1c4fd; font-size: 18px;'>Neural-Powered Fact Checking & Forensic Analysis</p>", unsafe_allow_html=True)

# 5. INPUT SECTION
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    input_text = st.text_area("📄 Analysis Input", placeholder="Paste news article or suspicious text here...", height=200)
    
    if st.button("🚀 EXECUTE FORENSIC SCAN"):
        if input_text.strip():
            # A. PREDICTION LOGIC
            probs = model.predict_proba([input_text])[0]
            p_fake, p_real = probs[0] * 100, probs[1] * 100
            
            # B. SENTIMENT ANALYSIS
            analysis = TextBlob(input_text)
            sentiment_score = (analysis.sentiment.polarity + 1) * 50 # 0 to 100 scale
            
            # C. TOP METRICS
            st.markdown("### 🔍 Forensic Overview")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Credibility Score", f"{p_real:.1f}%")
            with m2:
                st.metric("Emotional Bias", f"{sentiment_score:.1f}%")
            with m3:
                st.metric("Authenticity", "High" if p_real > 70 else "Suspicious")

            # D. VISUALIZATIONS
            st.markdown("---")
            col_chart, col_cloud = st.columns(2)
            
            with col_chart:
                fig = go.Figure(go.Bar(
                    x=['Real Pattern', 'Fake Pattern', 'Bias Level'],
                    y=[p_real, p_fake, sentiment_score],
                    marker_color=['#00ffcc', '#ff3366', '#4facfe']
                ))
                fig.update_layout(
                    title="Linguistic Signature",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_cloud:
                # Fixed WordCloud with standard colormap
                wc = WordCloud(
                    background_color=None, 
                    mode="RGBA", 
                    width=500, 
                    height=350, 
                    colormap='viridis' 
                ).generate(input_text)
                
                fig_wc, ax = plt.subplots(facecolor='none')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)

            # E. FINAL VERDICT
            st.markdown("---")
            if p_real > 50:
                st.success(f"✅ ANALYSIS COMPLETE: Content likely AUTHENTIC ({p_real:.1f}% confidence).")
            else:
                st.error(f"⚠️ ANALYSIS COMPLETE: Content flagged as MISINFORMATION ({p_fake:.1f}% risk).")
            
            # F. DOWNLOADABLE REPORT
            report = f"VeriLens AI Forensic Report\nLead Engineer: Isha\n\nDecision: {'AUTHENTIC' if p_real > 50 else 'FAKE'}\nReal Score: {p_real:.1f}%\nFake Risk: {p_fake:.1f}%\nBias Score: {sentiment_score:.1f}%"
            st.download_button("📩 Download Analysis Report", report, file_name="VeriLens_Report.txt")

        else:
            st.warning("Please enter text to begin scanning.")
    st.markdown('</div>', unsafe_allow_html=True)

# 6. SIGNATURE FOOTER
st.markdown("<br><div style='text-align: center; color: rgba(255,255,255,0.4); letter-spacing: 2px;'>DEVELOPED BY ISHA ❤️ 2026</div>", unsafe_allow_html=True)
