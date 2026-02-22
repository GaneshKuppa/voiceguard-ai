#!/usr/bin/env python3
"""Convert model to ONNX format for mobile deployment (future use)"""
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import os

def convert_model(model_path="models/voice_guard_model", output_path="models/voice_guard_model.onnx"):
    """Convert PyTorch model to ONNX format"""
    print(f"🔄 Converting {model_path} to ONNX...")
    
    # Load model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Create dummy input (4 seconds of audio at 16kHz)
    dummy_input = torch.randn(1, 64000)  # batch_size=1, sequence_length=64000
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_values'],
        output_names=['logits'],
        dynamic_axes={
            'input_values': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model saved to {output_path}")
    print(f"📦 Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return output_path

if __name__ == "__main__":
    convert_model()