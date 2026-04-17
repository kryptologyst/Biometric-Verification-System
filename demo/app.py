"""Interactive Streamlit demo for biometric verification system."""

import logging
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.biometric_verifier import (
    FingerprintVerifier, FaceVerifier, VoiceVerifier, 
    MultiModalVerifier, generate_synthetic_dataset
)
from eval.biometric_metrics import BiometricEvaluator, create_leaderboard
from defenses.anti_spoofing import AntiSpoofingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Biometric Verification System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'verifiers' not in st.session_state:
    st.session_state.verifiers = {
        'fingerprint': FingerprintVerifier(),
        'face': FaceVerifier(),
        'voice': VoiceVerifier()
    }
    st.session_state.multimodal_verifier = MultiModalVerifier()
    st.session_state.anti_spoofing = AntiSpoofingSystem()
    st.session_state.evaluator = BiometricEvaluator()
    st.session_state.enrolled_users = set()
    st.session_state.test_results = {}

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔐 Biometric Verification System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research and Education Disclaimer</h4>
    <p>This biometric verification system is designed for research and educational purposes only. 
    It uses synthetic data and simplified models that may not reflect the complexity and security 
    requirements of actual biometric systems. This system should not be used for production 
    security operations or real-world authentication.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Enrollment", "Verification", "Evaluation", "Anti-Spoofing", "System Status"]
        )
        
        st.header("Settings")
        threshold = st.slider("Verification Threshold", 0.0, 1.0, 0.5, 0.01)
        
        # Update thresholds
        for verifier in st.session_state.verifiers.values():
            verifier.set_threshold(threshold)
    
    # Route to appropriate page
    if page == "Enrollment":
        enrollment_page()
    elif page == "Verification":
        verification_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Anti-Spoofing":
        anti_spoofing_page()
    elif page == "System Status":
        system_status_page()

def enrollment_page():
    """User enrollment page."""
    st.header("👤 User Enrollment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enroll New User")
        
        user_id = st.text_input("User ID", value="user_001")
        modality = st.selectbox("Biometric Modality", ["fingerprint", "face", "voice"])
        
        if st.button("Generate Synthetic Sample"):
            # Generate synthetic biometric data
            if modality == "fingerprint":
                raw_data = np.random.randn(100)
            elif modality == "face":
                raw_data = np.random.randn(200)
            elif modality == "voice":
                raw_data = np.random.randn(150)
            
            # Enroll user
            verifier = st.session_state.verifiers[modality]
            success = verifier.enroll(user_id, raw_data)
            
            if success:
                st.session_state.enrolled_users.add(user_id)
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Enrollment Successful</h4>
                <p>User <strong>{user_id}</strong> has been enrolled for <strong>{modality}</strong> verification.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                <h4>❌ Enrollment Failed</h4>
                <p>There was an error during enrollment. Please try again.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Enrolled Users")
        
        if st.session_state.enrolled_users:
            enrolled_df = pd.DataFrame({
                'User ID': list(st.session_state.enrolled_users),
                'Modalities': ['fingerprint, face, voice'] * len(st.session_state.enrolled_users)
            })
            st.dataframe(enrolled_df, use_container_width=True)
        else:
            st.info("No users enrolled yet.")

def verification_page():
    """User verification page."""
    st.header("🔍 User Verification")
    
    if not st.session_state.enrolled_users:
        st.warning("Please enroll users first before testing verification.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Single Modality Verification")
        
        user_id = st.selectbox("Select User", list(st.session_state.enrolled_users))
        modality = st.selectbox("Select Modality", ["fingerprint", "face", "voice"])
        
        if st.button("Test Verification"):
            # Generate test data
            if modality == "fingerprint":
                test_data = np.random.randn(100)
            elif modality == "face":
                test_data = np.random.randn(200)
            elif modality == "voice":
                test_data = np.random.randn(150)
            
            # Verify user
            verifier = st.session_state.verifiers[modality]
            is_verified, score = verifier.verify(user_id, test_data)
            
            # Display result
            if is_verified:
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Verification Successful</h4>
                <p>User <strong>{user_id}</strong> verified with <strong>{modality}</strong>.</p>
                <p>Similarity Score: <strong>{score:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                <h4>❌ Verification Failed</h4>
                <p>User <strong>{user_id}</strong> could not be verified with <strong>{modality}</strong>.</p>
                <p>Similarity Score: <strong>{score:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Multi-Modal Verification")
        
        user_id = st.selectbox("Select User (Multi-Modal)", list(st.session_state.enrolled_users))
        
        modalities_to_test = st.multiselect(
            "Select Modalities", 
            ["fingerprint", "face", "voice"],
            default=["fingerprint", "face"]
        )
        
        if st.button("Test Multi-Modal Verification") and modalities_to_test:
            # Generate test data for each modality
            biometric_data = {}
            for modality in modalities_to_test:
                if modality == "fingerprint":
                    biometric_data[modality] = np.random.randn(100)
                elif modality == "face":
                    biometric_data[modality] = np.random.randn(200)
                elif modality == "voice":
                    biometric_data[modality] = np.random.randn(150)
            
            # Multi-modal verification
            is_verified, fused_score = st.session_state.multimodal_verifier.verify_multimodal(
                user_id, biometric_data
            )
            
            # Display individual scores
            individual_scores = {}
            for modality, data in biometric_data.items():
                _, score = st.session_state.multimodal_verifier.verify(user_id, modality, data)
                individual_scores[modality] = score
            
            # Display result
            if is_verified:
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Multi-Modal Verification Successful</h4>
                <p>User <strong>{user_id}</strong> verified with multiple modalities.</p>
                <p>Fused Score: <strong>{fused_score:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                <h4>❌ Multi-Modal Verification Failed</h4>
                <p>User <strong>{user_id}</strong> could not be verified with multiple modalities.</p>
                <p>Fused Score: <strong>{fused_score:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show individual scores
            st.subheader("Individual Scores")
            scores_df = pd.DataFrame(list(individual_scores.items()), columns=['Modality', 'Score'])
            st.dataframe(scores_df, use_container_width=True)

def evaluation_page():
    """System evaluation page."""
    st.header("📊 System Evaluation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Generate Test Data")
        
        n_users = st.slider("Number of Users", 10, 100, 50)
        n_samples = st.slider("Samples per User", 5, 20, 10)
        
        if st.button("Generate Dataset"):
            with st.spinner("Generating synthetic dataset..."):
                dataset = generate_synthetic_dataset(
                    n_users=n_users,
                    n_samples_per_user=n_samples
                )
                st.session_state.test_dataset = dataset
                st.success(f"Generated dataset with {n_users} users and {n_samples} samples per user")
    
    with col2:
        st.subheader("Run Evaluation")
        
        if 'test_dataset' not in st.session_state:
            st.warning("Please generate test data first.")
        else:
            if st.button("Evaluate All Modalities"):
                with st.spinner("Running evaluation..."):
                    results = {}
                    
                    for modality, verifier in st.session_state.verifiers.items():
                        result = st.session_state.evaluator.evaluate_verifier(
                            verifier, st.session_state.test_dataset, modality
                        )
                        results[modality] = result
                    
                    st.session_state.evaluation_results = results
                    st.success("Evaluation completed!")
    
    # Display results
    if 'evaluation_results' in st.session_state:
        st.subheader("Evaluation Results")
        
        # Create metrics table
        metrics_data = []
        for modality, metrics in st.session_state.evaluation_results.items():
            metrics_data.append({
                'Modality': modality.capitalize(),
                'EER': f"{metrics['EER']:.4f}",
                'minDCF': f"{metrics['minDCF']:.4f}",
                'ROC AUC': f"{metrics['ROC_AUC']:.4f}",
                'PR AUC': f"{metrics['PR_AUC']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Create leaderboard
        st.subheader("Performance Leaderboard")
        leaderboard = create_leaderboard(st.session_state.evaluation_results)
        st.text(leaderboard)
        
        # Performance comparison chart
        st.subheader("Performance Comparison")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('EER (Lower is Better)', 'ROC AUC (Higher is Better)', 
                          'minDCF (Lower is Better)', 'PR AUC (Higher is Better)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        modalities = list(st.session_state.evaluation_results.keys())
        
        # EER
        eer_values = [st.session_state.evaluation_results[m]['EER'] for m in modalities]
        fig.add_trace(
            go.Bar(x=modalities, y=eer_values, name='EER', marker_color='red'),
            row=1, col=1
        )
        
        # ROC AUC
        roc_values = [st.session_state.evaluation_results[m]['ROC_AUC'] for m in modalities]
        fig.add_trace(
            go.Bar(x=modalities, y=roc_values, name='ROC AUC', marker_color='blue'),
            row=1, col=2
        )
        
        # minDCF
        mindcf_values = [st.session_state.evaluation_results[m]['minDCF'] for m in modalities]
        fig.add_trace(
            go.Bar(x=modalities, y=mindcf_values, name='minDCF', marker_color='green'),
            row=2, col=1
        )
        
        # PR AUC
        pr_values = [st.session_state.evaluation_results[m]['PR_AUC'] for m in modalities]
        fig.add_trace(
            go.Bar(x=modalities, y=pr_values, name='PR AUC', marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Biometric Verification Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)

def anti_spoofing_page():
    """Anti-spoofing demonstration page."""
    st.header("🛡️ Anti-Spoofing Detection")
    
    st.markdown("""
    This page demonstrates liveness detection capabilities to prevent spoofing attacks.
    The system can detect fake biometric samples using various techniques.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Train Liveness Detectors")
        
        modality = st.selectbox("Select Modality", ["fingerprint", "face", "voice"])
        n_samples = st.slider("Training Samples", 100, 1000, 500)
        
        if st.button("Train Detector"):
            with st.spinner("Training liveness detector..."):
                # Generate synthetic training data
                live_samples, spoof_samples = st.session_state.anti_spoofing.generate_synthetic_spoof_data(
                    modality, n_samples, n_samples
                )
                
                # Train detector
                st.session_state.anti_spoofing.train_detector(modality, live_samples, spoof_samples)
                
                st.success(f"Liveness detector trained for {modality}")
    
    with col2:
        st.subheader("Test Liveness Detection")
        
        if st.button("Test Live Sample"):
            # Generate test sample
            if modality == "fingerprint":
                test_data = np.random.randn(100) + np.random.normal(0, 0.1, 100)
            elif modality == "face":
                test_data = np.random.randn(200) + np.random.normal(0, 0.1, 200)
            elif modality == "voice":
                test_data = np.random.randn(150) + np.random.normal(0, 0.1, 150)
            
            # Check liveness
            is_live, confidence = st.session_state.anti_spoofing.check_liveness(modality, test_data)
            
            if is_live:
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Live Sample Detected</h4>
                <p>The sample appears to be from a live person.</p>
                <p>Confidence: <strong>{confidence:.3f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                <h4>⚠️ Potential Spoof Detected</h4>
                <p>The sample may be a spoofed/fake biometric.</p>
                <p>Confidence: <strong>{confidence:.3f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
        
        if st.button("Test Spoof Sample"):
            # Generate spoof sample
            if modality == "fingerprint":
                test_data = np.random.uniform(-1, 1, 100) + np.random.normal(0, 0.3, 100)
            elif modality == "face":
                test_data = np.random.uniform(-0.5, 0.5, 200) + np.random.normal(0, 0.2, 200)
            elif modality == "voice":
                test_data = np.random.uniform(-0.8, 0.8, 150) + np.random.normal(0, 0.25, 150)
            
            # Check liveness
            is_live, confidence = st.session_state.anti_spoofing.check_liveness(modality, test_data)
            
            if is_live:
                st.markdown(f"""
                <div class="error-box">
                <h4>❌ Spoof Not Detected</h4>
                <p>The spoofed sample was incorrectly classified as live.</p>
                <p>Confidence: <strong>{confidence:.3f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Spoof Detected</h4>
                <p>The spoofed sample was correctly identified as fake.</p>
                <p>Confidence: <strong>{confidence:.3f}</strong></p>
                </div>
                """, unsafe_allow_html=True)

def system_status_page():
    """System status and information page."""
    st.header("ℹ️ System Status")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("System Information")
        
        status_data = {
            'Component': ['Fingerprint Verifier', 'Face Verifier', 'Voice Verifier', 
                         'Multi-Modal Verifier', 'Anti-Spoofing System'],
            'Status': ['✅ Active', '✅ Active', '✅ Active', '✅ Active', '✅ Active'],
            'Users Enrolled': [len(st.session_state.enrolled_users)] * 5
        }
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True)
        
        st.subheader("Configuration")
        config_data = {
            'Setting': ['Verification Threshold', 'Liveness Threshold', 'Random Seed'],
            'Value': ['0.5', '0.7', '42']
        }
        
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True)
    
    with col2:
        st.subheader("Performance Metrics")
        
        if 'evaluation_results' in st.session_state:
            # Create performance summary
            performance_data = []
            for modality, metrics in st.session_state.evaluation_results.items():
                performance_data.append({
                    'Modality': modality.capitalize(),
                    'EER': metrics['EER'],
                    'ROC AUC': metrics['ROC_AUC'],
                    'Genuine Mean': metrics['Genuine_Mean'],
                    'Impostor Mean': metrics['Impostor_Mean']
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
        else:
            st.info("Run evaluation to see performance metrics.")
        
        st.subheader("System Health")
        
        # Simulate system health metrics
        health_data = {
            'Metric': ['CPU Usage', 'Memory Usage', 'Response Time', 'Error Rate'],
            'Value': ['15%', '45%', '120ms', '0.1%'],
            'Status': ['✅ Good', '✅ Good', '✅ Good', '✅ Good']
        }
        
        health_df = pd.DataFrame(health_data)
        st.dataframe(health_df, use_container_width=True)

if __name__ == "__main__":
    main()
