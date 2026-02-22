# backend/inference.py - Model loading and prediction
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import os
import json

class VoiceAnalyzer:
    """Unified voice analyzer for file and live analysis"""
    
    def __init__(self, model_path="models/voice_guard_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.ready = False
        self.temperature = 0.8  # For confidence calibration
        
        # Check if model exists
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                print(f"📥 Loading model from: {model_path}")
                
                # Load processor
                self.processor = Wav2Vec2Processor.from_pretrained(model_path)
                
                # Load model (handles both .bin and .safetensors)
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32 if self.device.type == 'cpu' else None
                )
                
                self.model.to(self.device)
                self.model.eval()
                self.ready = True
                
                # Load training config if available
                config_file = os.path.join(model_path, "training_config.json")
                if os.path.exists(config_file):
                    with open(config_file) as f:
                        self.training_config = json.load(f)
                    print(f"✅ Model loaded | Trained with: {self.training_config.get('class_distribution', {})}")
                else:
                    print(f"✅ Model loaded on {self.device}")
                    
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.ready = False
        else:
            print(f"⚠️ Model config not found at {model_path}")
            print("💡 Run 'python training/train.py' first to train the model")
            self.ready = False

    def predict(self, audio_path, return_probs=True):
        """
        Analyze audio file and return prediction
        
        Args:
            audio_path: Path to audio file
            return_probs: Whether to return probability scores
            
        Returns:
            dict with prediction, confidence, and probabilities
        """
        if not self.ready:
            return {
                "score": 0,
                "label": "Model Not Ready",
                "ai_probability": 0,
                "human_probability": 0,
                "error": "Model not trained. Run 'python training/train.py' first."
            }

        try:
            # Load and preprocess audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono and resample to 16kHz
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = waveform.squeeze().numpy()
            
            # Process through Wav2Vec2
            inputs = self.processor(
                waveform, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True,
                max_length=64000,  # 4 seconds at 16kHz
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                
                # Apply temperature scaling for calibrated confidence
                probabilities = torch.softmax(logits / self.temperature, dim=-1)[0]
                
                ai_score = probabilities[1].item()
                human_score = probabilities[0].item()
                
                prediction = "AI Generated" if ai_score > human_score else "Human Voice"
                confidence = max(ai_score, human_score)
                
                result = {
                    "score": round(confidence * 100, 2),
                    "label": prediction,
                    "ai_probability": round(ai_score * 100, 2),
                    "human_probability": round(human_score * 100, 2)
                }
                
                if return_probs:
                    result["raw_probabilities"] = {
                        "human": human_score,
                        "ai": ai_score
                    }
                
                return result
                
        except Exception as e:
            return {
                "score": 0,
                "label": "Error",
                "ai_probability": 0,
                "human_probability": 0,
                "error": str(e)
            }

    def predict_from_array(self, audio_array, sample_rate=16000):
        """Analyze audio from numpy array (for live analysis)"""
        if not self.ready:
            return {"error": "Model not ready"}
        
        try:
            # Ensure correct shape and type
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            
            # Process through model
            inputs = self.processor(
                audio_array, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True,
                max_length=64000,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits / self.temperature, dim=-1)[0]
                
                ai_score = probabilities[1].item()
                human_score = probabilities[0].item()
                
                return {
                    "prediction": "AI Generated" if ai_score > human_score else "Human Voice",
                    "confidence": max(ai_score, human_score) * 100,
                    "ai_probability": ai_score * 100,
                    "human_probability": human_score * 100
                }
                
        except Exception as e:
            return {"error": str(e)}


# Global instance for API use
analyzer = VoiceAnalyzer()