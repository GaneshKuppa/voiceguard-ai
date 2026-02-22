# Recording Display Component for Live Analysis

def display_recording_proof(result):
    """Display recording proof in Streamlit UI"""
    import streamlit as st
    import os
    
    if result.get("recording_saved") and result.get("recording_path"):
        st.markdown("---")
        st.subheader("🎤 Recording Proof")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success("✅ Recording saved as proof of live analysis")
            st.info(f"**Recording ID:** `{result['recording_id']}`")
            
            # Audio player
            recording_path = result["recording_path"]
            if os.path.exists(recording_path):
                st.audio(recording_path, format="audio/wav")
        
        with col2:
            st.metric("Duration", f"{result.get('duration_seconds', 0):.1f}s")
            st.metric("Chunks Analyzed", result.get("chunks_analyzed", 0))
            
            # Download button
            if os.path.exists(recording_path):
                with open(recording_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Recording",
                        data=f.read(),
                        file_name=f"{result['recording_id']}.wav",
                        mime="audio/wav"
                    )
        
        # Metadata expander
        with st.expander("📋 View Recording Metadata"):
            st.json({
                "recording_id": result.get("recording_id"),
                "duration_seconds": result.get("duration_seconds"),
                "chunks_analyzed": result.get("chunks_analyzed"),
                "silent_chunks": result.get("silent_chunks", 0),
                "sample_rate": 16000,
                "format": "WAV (16-bit, Mono)"
            })
