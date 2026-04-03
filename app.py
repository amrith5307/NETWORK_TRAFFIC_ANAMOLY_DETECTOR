import streamlit as st
import pandas as pd
import os
import time

st.set_page_config(page_title="Network Shield AI", layout="wide", page_icon="🛡️")

if 'run_completed' not in st.session_state:
    st.session_state['run_completed'] = False

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
            time.sleep(2) # Stability pause
            st.session_state['run_completed'] = True
            st.sidebar.success("Analysis Complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# --- DASHBOARD ---
results_path = os.path.join("results", "latest_results.csv")
if st.session_state['run_completed'] and os.path.exists(results_path):
    res_df = pd.read_csv(results_path)
    
    # ACCURACY METRICS ROW
    cols = st.columns(3)
    for i, row in res_df.iterrows():
        with cols[i]:
            st.metric(label=f"⭐ {row['MODEL']}", value=f"{row['ACCURACY']*100:.1f}%", delta=f"F1: {row['F1-SCORE']:.2f}")

    tab1, tab2, tab3 = st.tabs(["🏆 Verdict", "📈 Full Data", "📍 Visuals"])

    with tab1:
        ensemble_row = res_df[res_df['MODEL'] == 'ENSEMBLE'].iloc[0]
        st.success(f"### System Verdict: {ensemble_row['ACCURACY']*100:.1f}% Accuracy")
        st.info("The system has concluded its analysis of 30,000 traffic packets using weighted unsupervised voting.")

    with tab2:
        st.table(res_df) 

    with tab3:
        st.subheader("Visual Analysis")
        # Check for Comparison Plot
        comp_path = os.path.join("results", "plots", "final_metrics_comparison.png")
        if os.path.exists(comp_path):
            st.image(comp_path)
        
        st.markdown("---")
        
        # Check for Confusion Matrix
        cm_path = os.path.join("results", "plots", "ENSEMBLE_cm.png")
        if os.path.exists(cm_path):
            st.markdown("#### 🎯 Ensemble Confusion Matrix")
            st.image(cm_path, width=700)
        else:
            st.warning("Visuals are still being finalized. Please wait 5 seconds and refresh.")
else:
    st.info("System Idle. Press 'Run' in the sidebar to begin monitoring.")