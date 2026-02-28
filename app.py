import streamlit as st
import joblib
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import string
import nltk

# --- LINGUISTIC ENGINE ---
@st.cache_resource
def init_nlp():
    for pkg in ['punkt', 'brown', 'punkt_tab']:
        nltk.download(pkg)
init_nlp()

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens 3D | Isha", layout="wide", page_icon="💎")

# 2. VIBRANT "CRYSTAL" UI (Bright & Clean)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    .crystal-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 30px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        border: 1px solid white;
        margin-top: -20px;
    }
    .isha-brand {
        font-weight: 900;
        color: white;
        font-size: 4rem;
        text-align: center;
        text-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 0px;
    }
    h3, p, label {
        color: #2d3436 !important;
        font-weight: 700 !important;
    }
    .stTextArea textarea {
        border-radius: 15px !important;
        border: 2px solid #dfe6e9 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_assets():
    return joblib.load('model.pkl')

model = load_assets()

# 4. HEADER
st.markdown('<h1 class="isha-brand">VERILENS 3D</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); font-size: 1.2rem; margin-bottom: 30px;'>AI Forensic Intelligence by Isha</p>", unsafe_allow_html=True)

# 5. MAIN INTERFACE
with st.container():
    st.markdown('<div class="crystal-card">', unsafe_allow_html=True)
    
    input_text = st.text_area("📄 Paste Article Content:", placeholder="Drop your news text here...", height=200)
    
    col_btn, _ = st.columns([1, 4])
    if col_btn.button("🚀 EXECUTE 3D SCAN"):
        if input_text.strip():
            # Analysis
            probs = model.predict_proba([input_text.lower()])[0]
            p_real = probs[1] * 100
            
            # --- 3D GAUGE METER ---
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = p_real,
                domain = {'x': [0, 1], 'y': [0, 1]},
                delta = {'reference': 50, 'increasing': {'color': "#00b894"}},
                number = {'suffix': "%", 'font': {'size': 80, 'color': '#2d3436'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#636e72"},
                    'bar': {'color': "#0984e3", 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#dfe6e9",
                    'steps': [
                        {'range': [0, 40], 'color': '#ff7675'},
                        {'range': [40, 70], 'color': '#ffeaa7'},
                        {'range': [70, 100], 'color': '#55efc4'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=50, b=20),
                height=450,
                font={'family': "Arial", 'weight': 'bold'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # --- SMART METRICS ---
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            
            blob = TextBlob(input_text)
            sentiment = (blob.sentiment.polarity + 1) * 50
            
            c1.metric("Authenticity Score", f"{p_real:.1f}%")
            c2.metric("Emotional Bias", f"{sentiment:.1f}%")
            c3.metric("Complexity", "High" if len(input_text.split()) > 100 else "Normal")

            # --- FINAL VERDICT BANNER ---
            if p_real > 50:
                st.success(f"🏆 VERDICT: THIS IS AUTHENTIC NEWS ({p_real:.1f}%)")
                st.balloons()
            else:
                st.error(f"🚨 VERDICT: THIS IS FAKE CONTENT ({100-p_real:.1f}% Risk)")
        else:
            st.warning("Please enter text to analyze.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 6. FOOTER
st.markdown("<br><div style='text-align: center; color: white; opacity: 0.9; font-weight: bold;'>© 2026 VeriLens 3D Suite | Created by Isha</div>", unsafe_allow_html=True)
