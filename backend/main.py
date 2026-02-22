# backend/main.py - Unified FastAPI application
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.inference import analyzer
from backend.live_analyzer import live_analyzer
import shutil
import os
import uuid
import tempfile
import asyncio
import time
import numpy as np
import base64
import json
from loguru import logger

# Configure logging
logger.add("logs/voiceguard.log", rotation="10 MB", level="INFO")

app = FastAPI(
    title="🛡️ VoiceGuard AI",
    description="Detect AI-generated vs Human voice (File Upload + Live Microphone)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary upload directory
UPLOAD_DIR = tempfile.mkdtemp(prefix="voiceguard_")

@app.on_event("shutdown")
def cleanup():
    """Clean up temporary files on shutdown"""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    logger.info("🧹 Cleanup complete")


# ============================================
# 📁 FILE UPLOAD ENDPOINT
# ============================================
@app.post("/analyze", summary="Analyze uploaded audio file")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (WAV, MP3, FLAC, M4A) to detect if it's human or AI-generated.
    
    Returns:
        dict with prediction, confidence, and probabilities
    """
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Use WAV, MP3, FLAC, or M4A."
        )
    
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] or ".wav"
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run prediction
        result = analyzer.predict(file_path)
        
        # Add metadata
        result["filename"] = file.filename
        result["file_size_kb"] = round(os.path.getsize(file_path) / 1024, 2)
        result["analysis_type"] = "file_upload"
        
        logger.info(f"📁 Analyzed: {file.filename} → {result['label']} ({result['score']}%)")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


# ============================================
# 🎤 LIVE MICROPHONE ENDPOINT
# ============================================
@app.post("/analyze-live", summary="Analyze live microphone input")
async def analyze_live(duration: int = 10, chunk_size: float = 2.0):
    """
    Record from microphone and analyze in real-time.
    
    Params:
        duration: Total seconds to record (5-60)
        chunk_size: Seconds per inference chunk (1-5)
        
    Returns:
        dict with aggregated prediction after recording completes
    """
    try:
        # Update chunk size
        live_analyzer.chunk_samples = int(live_analyzer.sample_rate * chunk_size)
        
        # Run analysis in thread pool (blocking I/O)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await asyncio.get_event_loop().run_in_executor(
                pool, 
                lambda: live_analyzer.analyze_live(duration=duration)
            )
        
        result["analysis_type"] = "live_microphone"
        logger.info(f"🎤 Live analysis: {result.get('prediction', 'N/A')} ({result.get('confidence', 0):.1f}%)")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Live analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Live analysis failed: {str(e)}")


@app.websocket("/ws/analyze-stream")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming analysis.
    
    Client sends audio chunks as base64, server streams back predictions.
    
    Frontend usage:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/analyze-stream');
    ws.onmessage = (event) => {
        const result = JSON.parse(event.data);
        updateUI(result.prediction, result.confidence);
    };
    // Send audio:
    ws.send(JSON.stringify({audio: base64Chunk, sampleRate: 16000}));
    ```
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connected for live analysis")
    
    try:
        audio_buffer = []
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if "audio" in message:
                # Decode base64 audio chunk
                audio_bytes = base64.b64decode(message["audio"])
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.extend(audio_np.tolist())
                
                # Analyze when we have enough samples
                if len(audio_buffer) >= live_analyzer.chunk_samples:
                    chunk = np.array(audio_buffer[:live_analyzer.chunk_samples])
                    pred = live_analyzer._predict_chunk(chunk)
                    smoothed = live_analyzer._smooth_prediction(pred)
                    
                    # Send result back
                    await websocket.send_json({
                        "prediction": smoothed["prediction"],
                        "confidence": smoothed["confidence"],
                        "ai_probability": smoothed["ai_probability"],
                        "human_probability": smoothed["human_probability"],
                        "timestamp": time.time()
                    })
                    
                    # Keep leftover samples for next chunk
                    audio_buffer = audio_buffer[live_analyzer.chunk_samples:]
            
            elif "stop" in message:
                logger.info("🔌 WebSocket stop received")
                break
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


# ============================================
# 🔍 HEALTH & INFO ENDPOINTS
# ============================================
@app.get("/health", summary="Health check")
async def health_check():
    """Check if the API and model are operational"""
    return {
        "status": "operational",
        "model_loaded": analyzer.ready,
        "device": str(analyzer.device) if analyzer.ready else None,
        "live_analysis_ready": live_analyzer.is_ready,
        "version": "2.0.0"
    }


@app.get("/", summary="API info")
async def root():
    """API root endpoint with documentation"""
    return {
        "message": "🛡️ VoiceGuard AI - Unified Voice Detection API",
        "endpoints": {
            "file_upload": "POST /analyze",
            "live_microphone": "POST /analyze-live", 
            "websocket_stream": "WS /ws/analyze-stream",
            "health": "GET /health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "version": "2.0.0",
        "github": "https://github.com/yourusername/voiceguard-ai"
    }


@app.get("/model-info", summary="Get model information")
async def model_info():
    """Get information about the loaded model"""
    if not analyzer.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": analyzer.model_path,
        "device": str(analyzer.device),
        "ready": analyzer.ready,
        "training_config": getattr(analyzer, 'training_config', None)
    }

# ============================================
# 🎤 RECORDING ENDPOINTS (NEW)
# ============================================
@app.get("/recordings", summary="List all recordings")
async def list_recordings():
    """List all saved recordings with metadata"""
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        return {"recordings": []}
    
    recordings = []
    for file in os.listdir(recordings_dir):
        if file.endswith('.wav'):
            filepath = os.path.join(recordings_dir, file)
            metadata_file = filepath.replace('.wav', '.json')
            
            recording_info = {
                "filename": file,
                "size_kb": round(os.path.getsize(filepath) / 1024, 2),
                "created": os.path.getctime(filepath)
            }
            
            if os.path.exists(metadata_file):
                with open(metadata_file) as f:
                    recording_info["metadata"] = json.load(f)
            
            recordings.append(recording_info)
    
    # Sort by newest first
    recordings.sort(key=lambda x: x["created"], reverse=True)
    
    return {"recordings": recordings, "count": len(recordings)}


@app.get("/recordings/{recording_id}", summary="Download recording")
async def get_recording(recording_id: str):
    """Download a specific recording by ID"""
    filepath = os.path.join("recordings", f"{recording_id}.wav")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Recording not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=f"{recording_id}.wav"
    )


@app.delete("/recordings", summary="Delete all recordings")
async def delete_recordings():
    """Delete all saved recordings (cleanup)"""
    recordings_dir = "recordings"
    if os.path.exists(recordings_dir):
        import shutil
        shutil.rmtree(recordings_dir)
        os.makedirs(recordings_dir)
        return {"message": "All recordings deleted"}
    return {"message": "No recordings to delete"}
