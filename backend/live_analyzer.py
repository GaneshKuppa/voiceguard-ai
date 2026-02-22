# backend/live_analyzer.py - WITH RECORDING SAVE FEATURE
import torch
import numpy as np
from collections import deque
import time
import os
import uuid
import wave
import json

try:
    from .inference import analyzer
except ImportError:
    from inference import analyzer


class LiveVoiceAnalyzer:
    def __init__(self, chunk_duration=2.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device = analyzer.device if hasattr(analyzer, 'device') else torch.device('cpu')
        self.audio_buffer = deque(maxlen=self.chunk_samples)
        self.prediction_history = deque(maxlen=5)
        self.smoothing_weight = 0.7
        self.vad_threshold = 0.00001  # Mac-optimized
        self.is_ready = analyzer.ready
        self.input_device = self._find_best_input_device()
        
        # ✅ Recordings storage
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        print(f"🎤 Live analyzer ready")
        print(f"   Input Device: {self.input_device}")
        print(f"   VAD Threshold: {self.vad_threshold}")
        print(f"   Recordings saved to: {self.recordings_dir}/")

    def _find_best_input_device(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    if 'microphone' in dev['name'].lower():
                        return i
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    return i
            return None
        except Exception as e:
            print(f"   ⚠️  Device detection failed: {e}")
            return None

    def _record_chunk(self, duration=None, save_recording=False):
        """Record audio chunk from microphone"""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed")
        
        record_duration = min(2.0, duration) if duration else 2.0
        total_samples = int(self.sample_rate * record_duration)
        
        print(f"🎙️  Listening for {record_duration}s... (SPEAK NOW!)", end=" ", flush=True)
        
        try:
            if self.input_device is not None:
                sd.default.device = (self.input_device, None)
            
            recording = sd.rec(
                frames=total_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocking=True,
                device=self.input_device
            )
            sd.wait()
            print("✅")
            
            audio_np = np.array(recording).squeeze()
            
            # Debug info
            energy = np.mean(audio_np ** 2)
            max_amp = np.max(np.abs(audio_np))
            print(f"   📊 Energy: {energy:.6f} | Max Amp: {max_amp:.6f}")
            
            return audio_np
            
        except Exception as e:
            print(f"\n❌ Audio error: {e}")
            raise

    def _save_recording(self, audio_data, recording_id):
        """Save recording to WAV file"""
        try:
            filename = f"{recording_id}.wav"
            filepath = os.path.join(self.recordings_dir, filename)
            
            # Convert float32 to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Save metadata
            metadata = {
                "recording_id": recording_id,
                "sample_rate": self.sample_rate,
                "duration_seconds": len(audio_data) / self.sample_rate,
                "samples": len(audio_data),
                "timestamp": time.time()
            }
            
            metadata_path = os.path.join(self.recordings_dir, f"{recording_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"   💾 Recording saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"   ⚠️  Could not save recording: {e}")
            return None

    def _is_voice_present(self, audio_chunk):
        """Voice activity detection"""
        energy = np.mean(audio_chunk ** 2)
        return energy > self.vad_threshold

    def _predict_chunk(self, audio_chunk):
        """Run model inference on audio chunk"""
        if not analyzer.ready:
            return {"error": "Model not ready"}
        return analyzer.predict_from_array(audio_chunk, sample_rate=self.sample_rate)

    def _smooth_prediction(self, new_pred):
        """Apply temporal smoothing"""
        if "error" in new_pred:
            return new_pred
            
        self.prediction_history.append(new_pred["ai_probability"] / 100)
        
        if len(self.prediction_history) < 3:
            return new_pred
        
        smoothed_ai = np.mean(list(self.prediction_history)[-3:])
        smoothed_human = 1 - smoothed_ai
        
        return {
            "ai_probability": smoothed_ai * 100,
            "human_probability": smoothed_human * 100,
            "prediction": "AI Generated" if smoothed_ai > 0.5 else "Human Voice",
            "confidence": max(smoothed_ai, smoothed_human) * 100,
            "raw_ai": new_pred["ai_probability"],
            "smoothed": True
        }

    def analyze_live(self, duration=10, chunk_size=2.0, save_recording=True):
        """Analyze live microphone input with optional recording save"""
        if not analyzer.ready:
            return {"error": "Model not trained. Run training first."}
        
        # Generate unique ID for this recording session
        recording_id = f"recording_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        print(f"🎯 Starting live analysis for {duration} seconds...")
        print(f"💾 Recording ID: {recording_id}")
        print("💡 SPEAK CLEARLY and CONTINUOUSLY\n")
        
        all_chunks = []
        results = []
        start_time = time.time()
        silent_chunks = 0
        
        while time.time() - start_time < duration:
            remaining = duration - (time.time() - start_time)
            if remaining < 1:
                break
                
            try:
                chunk = self._record_chunk(duration=min(chunk_size, remaining))
                all_chunks.append(chunk)
                
                if not self._is_voice_present(chunk):
                    silent_chunks += 1
                    if silent_chunks >= 2:
                        print("⚠️  Low audio detected. Please speak louder.")
                
                pred = self._predict_chunk(chunk)
                if "error" in pred:
                    print(f"❌ Prediction error: {pred['error']}")
                    continue
                
                smoothed = self._smooth_prediction(pred)
                results.append(smoothed)
                
                status = "🤖 AI" if smoothed["prediction"] == "AI Generated" else "👤 Human"
                conf = smoothed["confidence"]
                print(f"📊 [{time.time()-start_time:.1f}s] {status}: {conf:.1f}%")
                
            except Exception as e:
                print(f"⚠️  Chunk error: {e}")
                break
        
        # Combine all chunks into one recording
        if all_chunks and save_recording:
            combined_audio = np.concatenate(all_chunks)
            saved_path = self._save_recording(combined_audio, recording_id)
        else:
            saved_path = None
        
        if not results:
            return {
                "error": "No voice detected during recording",
                "recording_id": recording_id,
                "recording_saved": saved_path is not None,
                "recording_path": saved_path,
                "details": f"Recorded {duration}s but all chunks were silent",
                "solution": "Speak louder or check microphone permissions"
            }
        
        # Calculate final result
        weights = np.linspace(0.5, 1.0, len(results))
        final_ai = np.average([r["ai_probability"] for r in results], weights=weights)
        final_human = 100 - final_ai
        
        return {
            "prediction": "AI Generated" if final_ai > 50 else "Human Voice",
            "confidence": max(final_ai, final_human),
            "ai_probability": final_ai,
            "human_probability": final_human,
            "chunks_analyzed": len(results),
            "silent_chunks": silent_chunks,
            "duration_seconds": duration,
            "recording_id": recording_id,
            "recording_saved": saved_path is not None,
            "recording_path": saved_path,
            "recommendation": self._get_recommendation(final_ai, len(results))
        }

    def _get_recommendation(self, ai_prob, chunks):
        conf = max(ai_prob, 100 - ai_prob)
        if conf < 60:
            return "⚠️ Low confidence - speak clearer"
        elif ai_prob > 80:
            return "🔴 High AI probability - likely synthetic voice"
        elif ai_prob < 20:
            return "🟢 High human probability - likely natural voice"
        else:
            return "🟡 Moderate confidence"


live_analyzer = LiveVoiceAnalyzer()
