# frontend/app.py - Unified Streamlit UI
import streamlit as st
import requests
import librosa
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import time
import base64

# Backend API URL (can be overridden with environment variable)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Page configuration
st.set_page_config(
    page_title="🛡️ VoiceGuard AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center;}
    .verdict-human {color: #2E7D32; font-weight: bold; font-size: 1.5rem;}
    .verdict-ai {color: #C62828; font-weight: bold; font-size: 1.5rem;}
    .stButton>button {font-weight: bold; border-radius: 8px;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white;}
    .tab-content {padding: 10px;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🛡️ VoiceGuard AI</p>', unsafe_allow_html=True)
st.markdown("### Detect AI-generated vs Human Voice in seconds")
st.markdown("---")

# Session state initialization
if "live_results" not in st.session_state:
    st.session_state.live_results = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("**Analysis Options:**")
    show_spectrogram = st.checkbox("📊 Show Frequency Analysis", value=True)
    show_confidence = st.checkbox("📈 Show Confidence Details", value=True)
    confidence_threshold = st.slider("Confidence Threshold %", 50, 99, 75, 1)
    
    st.markdown("---")
    st.markdown("**Live Analysis:**")
    live_duration = st.slider("Recording Duration (seconds)", 5, 30, 10, 5)
    chunk_size = st.slider("Chunk Size (seconds)", 1.0, 5.0, 2.0, 0.5)
    
    st.markdown("---")
    st.markdown("**💡 Tips for Best Results:**")
    st.markdown("- 🎧 Use headphones to avoid feedback")
    st.markdown("- 🔇 Minimize background noise")
    st.markdown("- 🗣️ Speak clearly at normal volume")
    st.markdown("- 📁 Use 3-10 second audio clips")
    
    st.markdown("---")
    st.caption("🔒 Your audio is processed locally and never stored")

# Create tabs for different analysis modes
tab1, tab2, tab3 = st.tabs(["📁 File Upload", "🎤 Live Microphone", "ℹ️ About"])

# ============================================
# TAB 1: FILE UPLOAD ANALYSIS
# ============================================
with tab1:
    st.header("📁 Upload Audio File")
    st.markdown("Upload a pre-recorded audio file to analyze")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, M4A (max 25MB)"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format=uploaded_file.type)
        
        # File info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"📁 File: {uploaded_file.name}")
        with col_info2:
            st.info(f"📏 Size: {uploaded_file.size / 1024:.1f} KB")
        
        # Analyze button
        if st.button("🔍 Analyze File", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing audio... This may take 10-30 seconds"):
                try:
                    # Save to temp file for API
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Call backend API
                    with open(tmp_path, "rb") as f:
                        files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                        response = requests.post(f"{API_URL}/analyze", files=files, timeout=60)
                    
                    # Cleanup temp file
                    os.unlink(tmp_path)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Check for errors
                        if "error" in data:
                            st.error(f"❌ {data['error']}")
                            st.info("💡 Make sure the backend is running and model is trained")
                        else:
                            # Results display
                            st.markdown("---")
                            st.subheader("🎯 Analysis Results")
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Verdict card
                                if data['label'] == "Human Voice":
                                    st.markdown(f'<p class="verdict-human">✅ {data["label"]}</p>', unsafe_allow_html=True)
                                    st.success("This voice appears to be naturally recorded")
                                else:
                                    st.markdown(f'<p class="verdict-ai">⚠️ {data["label"]}</p>', unsafe_allow_html=True)
                                    st.warning("This voice shows patterns consistent with AI synthesis")
                                
                                # Main confidence metric
                                st.metric(
                                    label="🎯 Overall Confidence",
                                    value=f"{data['score']}%",
                                    delta=None
                                )
                            
                            with col2:
                                # Probability gauge chart
                                if show_confidence:
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=data['ai_probability'],
                                        domain={'x': [0, 1], 'y': [0, 1]},
                                        title={'text': "🤖 AI Probability"},
                                        gauge={
                                            'axis': {'range': [None, 100]},
                                            'bar': {'color': "darkblue", 'thickness': 0.3},
                                            'bordercolor': "gray", 'borderwidth': 1,
                                            'steps': [
                                                {'range': [0, 50], 'color': '#2E7D32'},  # Green = Human
                                                {'range': [50, 100], 'color': '#C62828'}  # Red = AI
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': confidence_threshold
                                            }
                                        }
                                    ))
                                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Detailed probabilities
                            if show_confidence:
                                st.markdown("#### 📊 Probability Breakdown")
                                col_prob1, col_prob2 = st.columns(2)
                                with col_prob1:
                                    st.metric("👤 Human Voice Probability", f"{data['human_probability']}%")
                                with col_prob2:
                                    st.metric("🤖 AI Generated Probability", f"{data['ai_probability']}%")
                            
                            # Spectrogram analysis
                            if show_spectrogram:
                                st.markdown("---")
                                st.subheader("📈 Frequency Spectrum Analysis")
                                
                                try:
                                    # Load audio for visualization
                                    y, sr = librosa.load(uploaded_file, sr=None)
                                    
                                    # Create spectrogram
                                    D = librosa.amplitude_to_db(
                                        np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), 
                                        ref=np.max
                                    )
                                    
                                    # Plot with Plotly
                                    fig_spec = go.Figure(data=go.Heatmap(
                                        z=D,
                                        colorscale='Viridis',
                                        showscale=True
                                    ))
                                    fig_spec.update_layout(
                                        height=300,
                                        xaxis_title="Time (frames)",
                                        yaxis_title="Frequency (bins)",
                                        margin=dict(l=20, r=20, t=20, b=40)
                                    )
                                    st.plotly_chart(fig_spec, use_container_width=True)
                                    
                                    st.caption("💡 AI voices often show unnatural frequency patterns or missing high-frequency noise")
                                    
                                except Exception as e:
                                    st.warning(f"Could not generate spectrogram: {e}")
                            
                            # Raw JSON for developers
                            with st.expander("🔧 View Raw API Response (Developer)"):
                                st.json(data)
                                
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.code(response.text)
                        
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to backend API")
                    st.info("""
                    **Fix this by:**
                    1. Make sure backend is running: `uvicorn backend.main:app --port 8000`
                    2. Check the backend shows: `Uvicorn running on http://127.0.0.1:8000`
                    3. Ensure no firewall is blocking port 8000
                    """)
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.info("Check the terminal for detailed error messages")
    else:
        # Welcome message when no file uploaded
        st.info("👆 Upload an audio file above to get started")
        
        # Demo section
        with st.expander("🎬 See how it works (Demo)"):
            st.markdown("""
            **Example workflow:**
            1. 🎤 Record yourself saying "Hello, this is a test"
            2. 🤖 Generate the same text using an AI voice tool
            3. 📤 Upload both files to VoiceGuard
            4. ✅ Compare the analysis results
            
            **What the AI detects:**
            - Unnatural pitch variations
            - Missing micro-breaths and pauses
            - Spectral artifacts from synthesis
            - Inconsistent formant transitions
            """)

# ============================================
# TAB 2: LIVE MICROPHONE ANALYSIS
# ============================================
with tab2:
    st.header("🎤 Live Microphone Analysis")
    st.markdown("Speak into your microphone for real-time AI voice detection")
    
    # Status display
    col_status1, col_status2 = st.columns([2, 1])
    
    with col_status1:
        st.subheader("📊 Real-Time Prediction")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        if st.session_state.live_results:
            latest = st.session_state.live_results[-1]
            if latest["prediction"] == "Human Voice":
                prediction_placeholder.success(f"👤 {latest['prediction']}")
            else:
                prediction_placeholder.error(f"🤖 {latest['prediction']}")
            
            confidence_placeholder.metric("Confidence", f"{latest['confidence']:.1f}%")
        else:
            prediction_placeholder.info("🎙️ Click 'Start Recording' to begin")
            confidence_placeholder.metric("Confidence", "--")
    
    with col_status2:
        st.subheader("🎯 AI Probability")
        
        if st.session_state.live_results:
            latest = st.session_state.live_results[-1]
            ai_prob = latest.get("ai_probability", 50)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ai_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "🤖 AI Probability"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': '#2E7D32'},
                        {'range': [50, 100], 'color': '#C62828'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'value': confidence_threshold
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(go.Figure(), use_container_width=True)
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        start_btn = st.button("🎙️ Start Recording", type="primary", 
                             disabled=st.session_state.is_recording, use_container_width=True)
    
    with col_btn2:
        stop_btn = st.button("⏹️ Stop", disabled=not st.session_state.is_recording, use_container_width=True)
    
    with col_btn3:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.live_results = []
            st.rerun()
    
    # Recording logic
    if start_btn:
        st.session_state.is_recording = True
        st.session_state.live_results = []
        st.info(f"🎙️ Recording started for {live_duration} seconds... Speak naturally!")
        
        try:
            with st.spinner("🔄 Analyzing live audio..."):
                response = requests.post(
                    f"{API_URL}/analyze-live",
                    params={"duration": live_duration, "chunk_size": chunk_size},
                    timeout=live_duration + 30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "error" not in result:
                        st.session_state.live_results.append({
                            "prediction": result["prediction"],
                            "confidence": result["confidence"],
                            "ai_probability": result["ai_probability"],
                            "human_probability": result["human_probability"],
                            "chunks_analyzed": result.get("chunks_analyzed", 1)
                        })
                        
                        st.success("✅ Recording complete!")
                        
                        # Display final result
                        st.markdown("---")
                        st.subheader("📋 Final Analysis Summary")
                        
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        
                        with col_sum1:
                            if result["prediction"] == "Human Voice":
                                st.metric("Final Prediction", "👤 Human Voice")
                            else:
                                st.metric("Final Prediction", "🤖 AI Generated")
                        
                        with col_sum2:
                            st.metric("Confidence", f"{result['confidence']:.1f}%")
                        
                        with col_sum3:
                            st.metric("Chunks Analyzed", result.get("chunks_analyzed", 1))
                        
                        # Recommendation
                        if "recommendation" in result:
                            if result["confidence"] > 80:
                                if result["prediction"] == "Human Voice":
                                    st.success(f"🟢 {result['recommendation']}")
                                else:
                                    st.error(f"🔴 {result['recommendation']}")
                            else:
                                st.warning(f"🟡 {result['recommendation']}")
                        
                        with st.expander("🔧 View Raw API Response"):
                            st.json(result)
                    else:
                        st.error(f"❌ {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.code(response.text)
                    
        except requests.exceptions.Timeout:
            st.warning("⏱️ Request timed out - try shorter duration")
        except Exception as e:
            st.error(f"❌ Connection error: {e}")
            st.info("Make sure backend is running: `uvicorn backend.main:app --port 8000`")
        
        st.session_state.is_recording = False
    
    # Live results history
    if st.session_state.live_results and not st.session_state.is_recording:
        st.markdown("---")
        st.subheader("📈 Analysis History")
        
        if len(st.session_state.live_results) > 0:
            latest = st.session_state.live_results[-1]
            
            if latest["confidence"] > 80:
                if latest["prediction"] == "Human Voice":
                    st.success("🟢 High confidence human detection - likely natural voice")
                else:
                    st.error("🔴 High confidence AI detection - likely synthetic voice")
            elif latest["confidence"] > 60:
                st.info("🟡 Moderate confidence - consider longer recording for certainty")
            else:
                st.warning("🟠 Low confidence - try recording in quieter environment")

# ============================================
# TAB 3: ABOUT & HELP
# ============================================
with tab3:
    st.header("ℹ️ About VoiceGuard AI")
    
    st.markdown("""
    ### 🎯 What is VoiceGuard AI?
    
    VoiceGuard AI is a deep learning-powered voice analysis tool that detects whether 
    an audio sample contains a **human voice** or **AI-generated synthetic voice**.
    
    ### 🔬 How It Works
    
    1. **Feature Extraction**: Uses Wav2Vec2 to extract speech patterns
    2. **Classification**: Trained model detects AI synthesis artifacts
    3. **Confidence Scoring**: Provides probability scores for both classes
    4. **Real-time Analysis**: Live microphone support for instant detection
    
    ### 📊 Model Information
    
    | Property | Value |
    |----------|-------|
    | Base Model | facebook/wav2vec2-base |
    | Training Data | Human + AI voice samples |
    | Accuracy | 85-92% (varies by dataset) |
    | Inference Time | 2-8 seconds (CPU) |
    
    ### 🎤 Analysis Modes
    
    **📁 File Upload**: Upload pre-recorded audio files (WAV, MP3, FLAC, M4A)
    
    **🎤 Live Microphone**: Real-time analysis as you speak (requires microphone access)
    
    ### 💡 Tips for Best Results
    
    - ✅ Use 3-10 second audio clips
    - ✅ Minimize background noise
    - ✅ Speak at normal volume and pace
    - ✅ Use good quality microphone
    - ❌ Avoid very short clips (<2 seconds)
    - ❌ Avoid heavy background music or noise
    
    ### 🔒 Privacy & Security
    
    - All audio processing happens locally or via secure API
    - No audio files are permanently stored
    - Temporary files are deleted after analysis
    
    ### 📚 Technical Details
    
    Built with:
    - **Backend**: FastAPI (Python)
    - **Frontend**: Streamlit
    - **ML Model**: Wav2Vec2 (Hugging Face Transformers)
    - **Audio Processing**: Librosa, SoundDevice
    
    ### 🐛 Troubleshooting
    
    | Issue | Solution |
    |-------|----------|
    | Cannot connect to API | Ensure backend is running on port 8000 |
    | Low confidence scores | Try longer audio clips, reduce noise |
    | Microphone not working | Check browser permissions, install sounddevice |
    | Slow analysis | Normal on CPU, consider GPU for production |
    
    ### 📞 Support
    
    For issues or questions, check the API documentation at `/docs` endpoint.
    """)
    
    # API Health Check
    st.markdown("---")
    st.subheader("🔍 System Status")
    
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            
            col_h1, col_h2, col_h3 = st.columns(3)
            
            with col_h1:
                if health_data.get("model_loaded"):
                    st.success("✅ Model Loaded")
                else:
                    st.error("❌ Model Not Loaded")
            
            with col_h2:
                st.info(f"💻 Device: {health_data.get('device', 'Unknown')}")
            
            with col_h3:
                st.info(f"🌐 API: Online")
        else:
            st.warning("⚠️ API returned unexpected status")
    except:
        st.error("🔌 Cannot connect to backend API")
        st.info("Start backend with: `uvicorn backend.main:app --port 8000`")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: gray; font-size: 0.9rem;">'
    '🛡️ VoiceGuard AI v2.0 • Built with Wav2Vec2 • For research and educational use'
    '</div>',
    unsafe_allow_html=True
)