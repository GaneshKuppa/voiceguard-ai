# training/train.py - MEMORY OPTIMIZED VERSION
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from torch.utils.data import DataLoader
from dataset import VoiceDataset, create_balanced_loader
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score, classification_report
import gc

# ✅ OPTIMIZED FOR LOW MEMORY
MODEL_NAME = "facebook/wav2vec2-base"
NUM_LABELS = 2
BATCH_SIZE = 1  # Reduced for low memory
EPOCHS = 10     # Reduced for faster training
LEARNING_RATE = 2e-5
EARLY_STOP_PATIENCE = 3
CHUNK_DURATION = 3.0  # Shorter audio chunks = less memory

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def prepare_data(data_dir):
    paths, labels = [], []
    for label, folder_name in enumerate(['human', 'fake']):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path): 
            print(f"⚠️ Warning: {folder_path} not found.")
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.wav', '.flac', '.mp3', '.m4a')):
                paths.append(os.path.join(folder_path, file))
                labels.append(label)
    
    print(f"📊 Dataset: {Counter(labels)}")
    
    if len(paths) < 10:
        print("❌ Too few samples. Need at least 10 total files.")
        return None, None, None, None
        
    return train_test_split(paths, labels, test_size=0.25, random_state=42, stratify=labels)

def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device} | Model: {MODEL_NAME}")
    print(f"💾 Memory optimized: Batch Size={BATCH_SIZE}, Chunk Duration={CHUNK_DURATION}s")

    result = prepare_data('data')
    if result[0] is None:
        return
        
    train_paths, test_paths, train_labels, test_labels = result
    print(f"📚 Train: {len(train_paths)} | Test: {len(test_paths)}")

    train_loader = create_balanced_loader(train_paths, train_labels, batch_size=BATCH_SIZE, augment=True, sample_rate=16000)
    test_loader = create_balanced_loader(test_paths, test_labels, batch_size=BATCH_SIZE, augment=False, sample_rate=16000)

    print(f"📥 Loading {MODEL_NAME}... (first time: ~5 min download)")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    # Freeze early layers to reduce memory
    for name, param in model.wav2vec2.named_parameters():
        if "encoder.layers" in name:
            try:
                layer_num = int(name.split(".")[2])
                if layer_num < 10:
                    param.requires_grad = False
            except (IndexError, ValueError):
                pass
    
    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    best_f1 = 0.0
    patience_counter = 0
    os.makedirs('models', exist_ok=True)

    print(f"\n🎯 Training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)

            inputs = processor(input_values.tolist(), sampling_rate=16000, return_tensors="pt", padding=True, max_length=int(16000 * CHUNK_DURATION), truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            # Clear memory after each batch
            del input_values, labels, inputs, outputs, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                input_values = batch["input_values"].to(device)
                labels_batch = batch["label"].to(device)
                
                inputs = processor(input_values.tolist(), sampling_rate=16000, return_tensors="pt", padding=True, max_length=int(16000 * CHUNK_DURATION), truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                
                del input_values, labels_batch, inputs, outputs, probs, preds
                gc.collect()
        
        acc = accuracy_score(all_labels, all_preds)
        human_f1 = f1_score(all_labels, all_preds, labels=[0], zero_division=0)
        fake_f1 = f1_score(all_labels, all_preds, labels=[1], zero_division=0)
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc*100:.1f}% | Human F1: {human_f1*100:.1f}% | Fake F1: {fake_f1*100:.1f}%")

        if human_f1 > best_f1:
            best_f1 = human_f1
            patience_counter = 0
            save_path = "models/voice_guard_model"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"💾 ✅ Saved! Human F1: {human_f1*100:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"⏹️ Early stopping (no improvement for {patience_counter} epochs)")
                break

    print(f"\n📊 Final Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Human', 'Fake']))
    print(f"\n🎉 Training complete! Best Human F1: {best_f1*100:.1f}%")
    print(f"📁 Model saved to: {save_path}/")
    print(f"💡 Next: Run 'docker compose up' to start application")

if __name__ == "__main__":
    train()
