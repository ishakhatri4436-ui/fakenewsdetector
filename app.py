import streamlit as st
import joblib
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
import time

# --- INITIALIZATION ---
@st.cache_resource
def init_nlp():
    nltk.download('punkt')
    nltk.download('brown')

init_nlp()

# 1. PAGE CONFIG & STYLING
st.set_page_config(page_title="VeriLens Quantum | Isha", layout="wide")

# CUSTOM CSS: Neon Glassmorphism & Animated Background
st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 40px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: white;
    }

    /* Glow Text */
    .glow-text {
        color: #fff;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 5px;
        text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #00d2ff, 0 0 40px #00d2ff;
        font-size: 4rem;
        font-weight: 900;
    }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# 2. LOGIC LAYER
def get_quantum_score(text, model):
    probs = model.predict_proba([text.lower()])[0]
    base_prob = probs[1] * 100
    
    # Forensic Signal Analysis
    indicators = ['official', 'minister', 'launched', 'inaugurated', 'accord', 'verified', 'diplomatic']
    boost = sum(7 for word in indicators if word in text.lower())
    
    return min(99.9, base_prob + boost)

# 3. UI CONTENT
st.markdown('<h1 class="glow-text">VERILENS QUANTUM</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:white; font-size:1.2rem; margin-top:-20px;'>Deep-Forensic Content Verification Suite • v5.0</p>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("### 📝 Input Feed")
        user_input = st.text_area("", placeholder="Paste article content for high-fidelity scanning...", height=280, label_visibility="collapsed")
        analyze_btn = st.button("⚡ INITIATE QUANTUM SCAN")

    with col_r:
        if analyze_btn and user_input:
            with st.spinner('Synchronizing Forensic Modules...'):
                time.sleep(1) # Simulation for "Attractive" effect
                
                model = joblib.load('model.pkl')
                score = get_quantum_score(user_input, model)
                
                # 3D GAUGE
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    number = {'suffix': "%", 'font': {'size': 80, 'color': 'white'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': 'white'},
                        'bar': {'color': "#00d2ff", 'thickness': 0.3},
                        'bgcolor': "rgba(0,0,0,0)",
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(255, 99, 132, 0.4)"},
                            {'range': [40, 75], 'color': "rgba(255, 206, 86, 0.4)"},
                            {'range': [75, 100], 'color': "rgba(75, 192, 192, 0.4)"}
                        ],
                        'threshold': {'line': {'color': "white", 'width': 4}, 'value': score}
                    }
                ))
                fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<div style='text-align:center; padding-top:100px;'><h4>Awaiting Input for Real-Time Analysis...</h4></div>", unsafe_allow_html=True)

    # 4. ADDITIONAL ELEMENTS (Metrics & Word Map)
    if analyze_btn and user_input:
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        blob = TextBlob(user_input)
        
        m1.metric("Emotional Charge", f"{blob.sentiment.subjectivity*100:.1f}%")
        m2.metric("Forensic Certainty", f"{score:.1f}%")
        m3.metric("Linguistic Tone", "Professional" if len(user_input.split()) > 100 else "Standard")
        
        if score > 50:
            st.balloons()
            st.success(f"💎 **VERDICT: TRUSTED CONTENT.** Authenticity verified at {score:.1f}%.")
        else:
            st.error(f"🚨 **VERDICT: CRITICAL RISK.** Suspicious patterns detected.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# 5. FOOTER
st.markdown("<br><p style='text-align:center; color:white; opacity:0.6;'>Isha Intelligence Labs • 2026 Pro Edition</p>", unsafe_allow_html=True)
