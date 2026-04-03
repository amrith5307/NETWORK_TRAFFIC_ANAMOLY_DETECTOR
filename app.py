import streamlit as st
import pandas as pd
import os
import time
from dotenv import load_dotenv
from google import genai  
from google.genai import types

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Network Shield AI", layout="wide", page_icon="🛡️")

# --- INITIALIZE SESSION STATE ---
if 'run_completed' not in st.session_state:
    st.session_state['run_completed'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.title("🛡️ Network Shield AI")
st.markdown("### *Weighted Ensemble Defense System*")

# --- SIDEBAR ---
st.sidebar.header("🕹️ Control Center")
if st.sidebar.button("🚀 Run Detection Pipeline"):
    st.session_state['run_completed'] = False
    import main 
    with st.spinner('AI is analyzing traffic patterns...'):
        try:
            main.main()
            # Trigger SHAP explicitly if it's not part of your main.main()
            from src.shap_explainer import run_shap_analysis
            run_shap_analysis() 
            
            time.sleep(1)
            st.session_state['run_completed'] = True
            st.sidebar.success("Analysis Complete!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# --- DASHBOARD ---
results_path = os.path.join("results", "latest_results.csv")

if st.session_state['run_completed'] and os.path.exists(results_path):
    res_df = pd.read_csv(results_path)

    # Dynamic Metric Columns - Fixes index out of range
    cols = st.columns(len(res_df))
    for i, (idx, row) in enumerate(res_df.iterrows()):
        with cols[i]:
            st.metric(label=f"⭐ {row['MODEL']}", value=f"{row['ACCURACY']*100:.1f}%", delta=f"F1: {row['F1-SCORE']:.2f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Verdict", "📈 Full Data", "📍 Visuals", "🔍 SHAP", "🤖 Gemini AI"])

    with tab1:
        st.success("Analysis of 30,000 traffic packets complete.")
        st.info("The system uses Isolation Forest + Elliptic Envelope for anomaly detection.")

    with tab2:
        st.dataframe(res_df, use_container_width=True)

    with tab3:
        comp_path = os.path.join("results", "plots", "final_metrics_comparison.png")
        if os.path.exists(comp_path): 
            st.image(comp_path, caption="Model Performance Comparison")

    with tab4:
        st.subheader("🔍 SHAP Explanation")
        # CRITICAL FIX: Filename must match what explainer.py saves
        shap_path = os.path.join("results", "plots", "shap_summary.png")
        
        if os.path.exists(shap_path): 
            st.image(shap_path, caption="Feature Importance (Mapped from PCA)")
            st.write("This plot shows which original network features had the most influence on the decision.")
        else: 
            st.warning("SHAP plot not found. Ensure 'run_shap_analysis()' is generating 'shap_summary.png' in results/plots/.")

    # --- TAB 5: GEMINI CHATBOT ---
    with tab5:
        st.subheader("🤖 Gemini Security Assistant")
        
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            GEMINI_API_KEY = st.text_input("Enter Gemini API Key", type="password")
        
        if GEMINI_API_KEY:
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                results_context = res_df.to_string()
                sys_instruct = f"You are a network security expert. Here are the project results: {results_context}. Help the user understand these anomalies."

                for msg in st.session_state['chat_history']:
                    with st.chat_message(msg['role']):
                        st.write(msg['content'])

                if prompt := st.chat_input("Ask about your NSL-KDD results..."):
                    st.session_state['chat_history'].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Gemini is analyzing..."):
                            response = client.models.generate_content(
                                model='gemini-2.5-flash', 
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=sys_instruct,
                                    temperature=0.7
                                )
                            )
                            reply = response.text
                            st.write(reply)
                            st.session_state['chat_history'].append({"role": "assistant", "content": reply})
            
            except Exception as e:
                st.error(f"Gemini Error: {e}")
        else:
            st.warning("Please provide an API Key.")

else:
    st.info("System Idle. Press 'Run' in the sidebar to begin.")