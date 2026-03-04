import streamlit as st
import pandas as pd
import os
import time
import shutil

# 1. Page Config
st.set_page_config(page_title="Network Shield AI", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'run_completed' not in st.session_state:
    st.session_state['run_completed'] = False

st.title("🛡️ Network Anomaly Detection Dashboard")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("Control Center")

if st.sidebar.button("🚀 Execute Detection Pipeline"):
    # Reset state before running
    st.session_state['run_completed'] = False
    
    import main
    with st.spinner('🧠 AI Models are analyzing traffic patterns...'):
        try:
            main.main()
            # Give the computer a moment to finish saving the files
            time.sleep(2) 
            
            # Double check the file actually exists before switching pages
            if os.path.exists("results/latest_results.csv"):
                st.session_state['run_completed'] = True
                st.sidebar.success("Analysis Complete!")
                st.rerun()
            else:
                st.error("Results file not found. Check if main.py is saving correctly.")
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")

# --- MAIN DASHBOARD LOGIC ---
results_path = "results/latest_results.csv"

# We check BOTH the session state and the physical file
if st.session_state['run_completed'] and os.path.exists(results_path):
    
    # Load data
    res_df = pd.read_csv(results_path)
    
    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Accuracy", "🏆 Verdict", "📊 Metrics", "📍 Visuals"])

    with tab1:
        st.subheader("Model Confidence")
        cols = st.columns(4)
        for i, row in res_df.iterrows():
            with cols[i]:
                acc = float(row['ACCURACY'])
                st.markdown(f"### {row['MODEL']}")
                st.metric(label="Accuracy", value=f"{acc*100:.1f}%")
                st.progress(acc)

    with tab2:
        ensemble = res_df.iloc[-1]
        st.header(f"System Verdict: {float(ensemble['ACCURACY'])*100:.2f}% Confidence")
        st.success("✅ The network is currently being monitored and secured by the Ensemble AI.")

    with tab3:
        st.subheader("Detailed Performance Data")
        st.table(res_df)

    with tab4:
        st.subheader("Visual Analysis")
        g1, g2 = st.columns(2)
        metrics_plot = "results/plots/final_metrics_comparison.png"
        pca_plot = "results/plots/traffic_clusters_pca.png"
        
        if os.path.exists(metrics_plot):
            with g1: st.image(metrics_plot)
        if os.path.exists(pca_plot):
            with g2: st.image(pca_plot)

else:
    # --- CLEAN WELCOME VIEW ---
    st.subheader("Welcome to the Network Security Analysis Interface.")
    st.write("The system is currently idle. Please initiate the detection pipeline from the sidebar to begin monitoring.")