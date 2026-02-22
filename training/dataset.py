# training/dataset.py - Data loading with augmentation
import torch
import torchaudio
from torch.utils.data import Dataset, WeightedRandomSampler
import random
import os
import numpy as np
from collections import Counter

class VoiceDataset(Dataset):
    """PyTorch Dataset for voice classification with augmentation"""
    
    def __init__(self, file_paths, labels, sample_rate=16000, augment=True, chunk_duration=4.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.augment = augment
        self.chunk_samples = int(sample_rate * chunk_duration)

    def __len__(self):
        return len(self.file_paths)

    def augment_audio(self, waveform):
        """Apply aggressive augmentation for small datasets"""
        if not self.augment:
            return waveform
        
        # 1. Add random noise
        noise_level = random.uniform(0.001, 0.01)
        noise = torch.randn_like(waveform) * noise_level
        waveform += noise
        
        # 2. Random gain adjustment
        gain = random.uniform(0.8, 1.2)
        waveform *= gain
        
        # 3. Time masking (simulate dropouts)
        if random.random() > 0.5:
            mask_len = random.randint(100, 2000)
            start = random.randint(0, max(0, len(waveform) - mask_len))
            waveform[start:start+mask_len] = 0
        
        # 4. Small pitch shift (±5%)
        if random.random() > 0.7:
            factor = random.uniform(0.95, 1.05)
            waveform = torchaudio.functional.resample(
                waveform, self.sample_rate, int(self.sample_rate * factor)
            )
            waveform = torchaudio.functional.resample(
                waveform, int(self.sample_rate * factor), self.sample_rate
            )
        
        return waveform

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            # Return silent audio as fallback
            waveform = torch.zeros(1, self.sample_rate)
            sr = self.sample_rate
        
        # Convert to mono and resample to target rate
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Apply augmentations
        waveform = self.augment_audio(waveform)
        waveform = waveform.squeeze()
        
        # Standardize length: pad or random crop to chunk_duration
        if waveform.shape[0] > self.chunk_samples:
            # Random crop for augmentation
            start = random.randint(0, waveform.shape[0] - self.chunk_samples)
            waveform = waveform[start:start+self.chunk_samples]
        else:
            # Zero-pad
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        return {
            "input_values": waveform,
            "label": torch.tensor(label, dtype=torch.long)
        }


def create_balanced_loader(file_paths, labels, batch_size=4, sample_rate=16000, augment=True):
    """Create DataLoader with weighted sampling for imbalanced datasets"""
    dataset = VoiceDataset(file_paths, labels, sample_rate, augment)
    
    # Calculate class weights: inverse frequency for balance
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    
    # Create sample weights for WeightedRandomSampler
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )