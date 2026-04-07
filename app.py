import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Custom modules
from src.data_loader import DataLoader
from src.preprocessing import preprocess_sequence
from src.optical_flow import DenseOpticalFlowRAFT
from src.physics_model import PhysicsModeling
from src.feature_extraction import FeatureExtractor
from src.ml_model import RiskPredictor

st.set_page_config(page_title="Visual Stress Detection", layout="wide")

st.title("Visual Stress Detection in Materials")
st.markdown('''
System using **Optical Flow (RAFT)**, **Physics-Based Modeling**, and **Machine Learning**
to detect micro-deformations and predict structural risk.
''')

# Sidebar configurations
st.sidebar.header("Configuration")
data_dir = st.sidebar.text_input("Data Directory", value="./data")
youngs_mod = st.sidebar.number_input("Young's Modulus (GPa)", value=250.0) * 1e9
poisson_r = st.sidebar.slider("Poisson's Ratio", min_value=0.1, max_value=0.5, value=0.3)

if st.sidebar.button("Run Full Pipeline"):
    with st.spinner("Loading Data and Preprocessing..."):
        loader = DataLoader(data_dir)
        dic_seq = loader.load_dic_sequence()
        processed_seq = preprocess_sequence(dic_seq)
    st.success(f"Loaded and preprocessed {len(processed_seq)} frames.")

    with st.spinner("Computing Dense Optical Flow (RAFT)..."):
        # For performance, only compute flow for the first two frames in this demo
        raft = DenseOpticalFlowRAFT()
        flow = raft.compute_flow(processed_seq[0], processed_seq[1])
    st.success("Dense displacement field computed.")

    with st.spinner("Calculating Physics-based Stress Models..."):
        physics = PhysicsModeling(youngs_modulus=youngs_mod, poisson_ratio=poisson_r)
        vm_stress = physics.process_displacement_to_stress(flow)
    st.success("Strain and Stress models calculated.")

    with st.spinner("Extracting Features and Predicting Risk..."):
        extractor = FeatureExtractor()
        # For demo purposes, we treat this single transition as temporal variation
        mock_seq = [vm_stress, vm_stress * 1.05, vm_stress * 1.1]
        features = extractor.extract_temporal_features(mock_seq)
        
        predictor = RiskPredictor()
        risk_probability = predictor.predict_risk(features)
    
    st.header("Results Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Base Frame")
        st.image(processed_seq[0], caption="Frame 1 (Normalized)", use_column_width=True, clamp=True)
    with col2:
        st.subheader("Dense Displacement (u-component)")
        fig, ax = plt.subplots()
        cax = ax.imshow(flow[..., 0], cmap='jet')
        fig.colorbar(cax)
        st.pyplot(fig)
    with col3:
        st.subheader("Von Mises Stress Heatmap")
        fig2, ax2 = plt.subplots()
        cax2 = ax2.imshow(vm_stress, cmap='hot')
        fig2.colorbar(cax2)
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Structural Risk Assessment")
    
    risk_percentage = risk_probability * 100
    if risk_percentage > 70:
        st.error(f"High Risk of Failure Detected: {risk_percentage:.2f}%")
    elif risk_percentage > 40:
        st.warning(f"Moderate Risk of Failure Detected: {risk_percentage:.2f}%")
    else:
        st.success(f"Low Risk (Structure Safe): {risk_percentage:.2f}%")
    
    st.write("Calculated based on extracted spatial and sequence temporal features.")

else:
    st.info("Configure parameters and click 'Run Full Pipeline' to start.")
