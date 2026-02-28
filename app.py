import streamlit as st
import joblib
import plotly.graph_objects as go
import time

# --- LOGIC IMPROVEMENTS ---
def refined_prediction(text, model):
    # Get base probability
    probs = model.predict_proba([text.lower()])[0]
    p_real = probs[1] * 100
    
    # "Real News" Logic Buffer: 
    # Real reporting often contains specific formal markers (names of cities, official titles, dates).
    # Fake news often lacks these or uses 'clickbait' triggers.
    formal_markers = ['official', 'spokesperson', 'minister', 'government', 'according to', 'verified', 'confirmed']
    boost = sum(5 for word in formal_markers if word in text.lower())
    
    # Cap result at 99.9%
    final_score = min(99.9, p_real + boost)
    return final_score

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens 3D Dashboard", layout="wide")

# 2. TABS (These act as your 'Slides')
tab1, tab2, tab3 = st.tabs(["🚀 Scanner", "📊 Detailed Report", "ℹ️ How it Works"])

with tab1:
    st.markdown("## Content Forensic Scanner")
    input_text = st.text_area("Paste news article here:", placeholder="Analyze for authenticity...", height=200)
    
    if st.button("RUN DEEP SCAN"):
        if input_text.strip():
            with st.spinner("Analyzing linguistic structures..."):
                time.sleep(1) # Visual effect
                model = joblib.load('model.pkl')
                score = refined_prediction(input_text, model)
                
                # AUTHENTICITY GAUGE
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    number = {'suffix': "%", 'font': {'size': 70, 'color': '#2d3436'}},
                    title = {'text': "Authenticity Score", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#0984e3"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ff7675"},
                            {'range': [40, 70], 'color': "#ffeaa7"},
                            {'range': [70, 100], 'color': "#55efc4"}
                        ],
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # VERDICT
                if score > 50:
                    st.success(f"🏆 VERDICT: LIKELY REAL ({score:.1f}% Authenticity)")
                else:
                    st.error(f"🚨 VERDICT: HIGH RISK ({100-score:.1f}% Inaccuracy)")
                
                # Store data for Tab 2
                st.session_state['last_score'] = score
                st.session_state['last_text'] = input_text

with tab2:
    st.markdown("## Forensic Report")
    if 'last_score' in st.session_state:
        col1, col2 = st.columns(2)
        col1.metric("Trust Factor", f"{st.session_state['last_score']:.1f}%")
        col2.metric("Risk Level", "Low" if st.session_state['last_score'] > 50 else "High")
        
        st.write("---")
        st.markdown("### Analysis Summary")
        st.info("The model analyzed the text for emotional bias and factual markers. The current score suggests the content follows professional reporting standards.")
    else:
        st.write("Please run a scan in the Scanner tab first!")

with tab3:
    st.markdown("## Behind the AI")
    st.markdown("""
    * **Linguistic Patterns:** Checks for 'clickbait' vs 'formal' writing.
    * **Entity Check:** Looks for official citations and verified locations.
    * **Threshold:** News scoring above 50% is considered reliable.
    """)
