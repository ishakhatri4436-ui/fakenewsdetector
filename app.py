import streamlit as st
import joblib
import plotly.graph_objects as go
from textblob import TextBlob
import re
import string
import nltk

# --- LINGUISTIC ENGINE INITIALIZATION ---
@st.cache_resource
def init_nlp():
    for pkg in ['punkt', 'brown', 'punkt_tab']:
        try:
            nltk.download(pkg)
        except:
            pass

init_nlp()

# 1. PAGE CONFIG
st.set_page_config(page_title="VeriLens 3D Pro | Isha", layout="wide", page_icon="💎")

# 2. LIGHT GRADIENT UI (Vibrant & Clean)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 25px;
        padding: 40px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid #ffffff;
    }
    .isha-brand {
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 5px;
    }
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 3. ADVANCED LOGIC: LINGUISTIC PATTERN REFINER
def calculate_authenticity(text, model):
    # Get raw probability from the ML model
    raw_prob = model.predict_proba([text.lower()])[0][1] * 100
    
    # Professional Pattern Detection (To fix the 'Real as Fake' error)
    # Real news uses specific professional indicators
    real_indicators = [
        'minister', 'inaugurated', 'facility', 'partnership', 'agreement', 
        'launched', 'pivotal', 'government', 'spokesperson', 'confirmed'
    ]
    
    boost = 0
    words = text.lower().split()
    for word in real_indicators:
        if word in words:
            boost += 5 # 5% boost for every professional indicator found
            
    # Fake news indicators (to counteract false boosts)
    fake_triggers = ['shocking', 'miracle', 'secret', 'leaked', 'conspiracy', 'emergency']
    for word in fake_triggers:
        if word in words:
            boost -= 7
            
    final_score = max(5, min(99.9, raw_prob + boost))
    return final_score

# 4. ASSET LOADING
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 5. HEADER
st.markdown('<h1 class="isha-brand">VERILENS 3D PRO</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #576574; font-weight: 600;'>AI Content Verification Suite v3.0</p>", unsafe_allow_html=True)

# 6. APP LAYOUT
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    input_text = st.text_area("📄 Paste News Article:", placeholder="Input the text you wish to verify...", height=200)
    
    col_btn, _ = st.columns([1, 4])
    if col_btn.button("🔍 ANALYZE CONTENT"):
        if input_text.strip():
            # Analysis
            auth_score = calculate_authenticity(input_text, model)
            blob = TextBlob(input_text)
            
            # 3D GAUGE DISPLAY
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = auth_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Authenticity Meter", 'font': {'size': 24, 'color': '#2f3542'}},
                delta = {'reference': 50, 'increasing': {'color': "#2ed573"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#1e90ff", 'thickness': 0.2},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#ced6e0",
                    'steps': [
                        {'range': [0, 45], 'color': '#ff6b6b'},
                        {'range': [45, 75], 'color': '#feca57'},
                        {'range': [75, 100], 'color': '#1dd1a1'}
                    ],
                    'threshold': {
                        'line': {'color': "gold", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=400, margin=dict(t=50, b=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            # EXTRA ELEMENTS
            st.markdown("### 🔬 Forensic Insights")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.write("**Emotional Bias**")
                st.subheader(f"{blob.sentiment.subjectivity*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.write("**Sentiment Mood**")
                mood = "Positive" if blob.sentiment.polarity > 0 else "Neutral/Alert"
                st.subheader(mood)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.write("**Context Quality**")
                complexity = "High" if len(input_text.split()) > 100 else "Standard"
                st.subheader(complexity)
                st.markdown('</div>', unsafe_allow_html=True)

            # FINAL ACTIONABLE VERDICT
            st.markdown("---")
            if auth_score > 55:
                st.success(f"✅ **VERDICT: TRUSTED CONTENT.** Patterns indicate verified reporting (Score: {auth_score:.1f}%)")
            else:
                st.error(f"⚠️ **VERDICT: SUSPICIOUS CONTENT.** Low linguistic reliability (Risk: {100-auth_score:.1f}%)")
        else:
            st.warning("Please provide an article for scanning.")
    st.markdown('</div>', unsafe_allow_html=True)

# 7. FOOTER
st.markdown("<br><div style='text-align: center; color: #57606f;'>© 2026 Isha Forensic Labs | Powered by VeriLens 3D</div>", unsafe_allow_html=True)
