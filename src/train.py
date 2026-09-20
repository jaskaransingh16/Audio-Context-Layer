"""
Training Script for Audio Context Layer Model (AudioContextTransformerQA)

Loads SoundContextQA dataset, builds vocabulary, trains model with AdamW optimizer and cosine LR scheduler,
saves best model checkpoint, and outputs loss curves.
"""

import os
import json
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.model import Vocabulary, AudioContextTransformerQA, AudioSpectrogramEncoder

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 5
MAX_AUDIO_LEN = 128000  # 8 seconds at 16kHz
MAX_QUESTION_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_TO_ID = {
    "perceptual": 0,
    "counting": 1,
    "temporal": 2,
    "causal": 3
}


class SoundQADataset(Dataset):
    """PyTorch Dataset loading audio samples and encoded QA pairs with pre-computed Mel-Spectrograms."""
    
    def __init__(self, data_json_path: str, audio_dir: str, vocab: Vocabulary, max_audio_len: int = MAX_AUDIO_LEN):
        self.audio_dir = audio_dir
        self.vocab = vocab
        self.max_audio_len = max_audio_len
        self.samples = []
        self.spec_cache: Dict[str, torch.Tensor] = {}
        
        with open(data_json_path, "r", encoding="utf-8") as f:
            raw_metadata = json.load(f)
            
        encoder_helper = AudioSpectrogramEncoder(d_model=256)
        
        # Pre-compute Mel-Spectrograms into RAM
        print(f"[Dataset] Pre-computing Mel-Spectrograms from '{audio_dir}'...", flush=True)
        unique_audio_files = set(sample["audio_file"] for sample in raw_metadata)
        
        with torch.no_grad():
            for fname in unique_audio_files:
                audio_path = os.path.join(self.audio_dir, fname)
                sr, audio_int16 = wavfile.read(audio_path)
                audio_float = (audio_int16.astype(np.float32) / 32767.0)
                if len(audio_float) < self.max_audio_len:
                    audio_float = np.pad(audio_float, (0, self.max_audio_len - len(audio_float)))
                else:
                    audio_float = audio_float[:self.max_audio_len]
                
                wave_tensor = torch.tensor(audio_float, dtype=torch.float32).unsqueeze(0)
                spec = encoder_helper.extract_mel_spectrogram(wave_tensor)[0]  # (1, n_mels, T)
                self.spec_cache[fname] = spec
            
        for sample in raw_metadata:
            audio_file = sample["audio_file"]
            for qa in sample["qa_pairs"]:
                self.samples.append({
                    "audio_file": audio_file,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "category": qa["category"],
                    "answer_type": qa["answer_type"]
                })

        if len(self.samples) > 1500:
            random.seed(42)
            random.shuffle(self.samples)
            self.samples = self.samples[:1500]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        
        # Retrieve pre-computed mel-spectrogram
        mel_spec = self.spec_cache[item["audio_file"]]
            
        # Encode question tokens
        q_ids = self.vocab.encode(item["question"], max_len=MAX_QUESTION_LEN)
        
        # Encode target answer: map single word/first word or answer string
        ans_clean = item["answer"].lower().strip()
        ans_word = self.vocab.tokenize(ans_clean)[0] if self.vocab.tokenize(ans_clean) else "<unk>"
        ans_id = self.vocab.word2idx.get(ans_word, self.vocab.word2idx["<unk>"])
        
        cat_id = CATEGORY_TO_ID.get(item["category"], 0)
        
        return {
            "mel_spec": mel_spec,
            "question_ids": torch.tensor(q_ids, dtype=torch.long),
            "answer_id": torch.tensor(ans_id, dtype=torch.long),
            "cat_id": torch.tensor(cat_id, dtype=torch.long),
            "raw_question": item["question"],
            "raw_answer": item["answer"],
            "category": item["category"]
        }


def build_vocabulary_from_dataset(dataset_dir: str) -> Vocabulary:
    """Builds unified vocabulary mapping from train metadata."""
    vocab = Vocabulary()
    train_path = os.path.join(dataset_dir, "train.json")
    
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for sample in data:
        for qa in sample["qa_pairs"]:
            for word in vocab.tokenize(qa["question"]):
                vocab.add_word(word)
            for word in vocab.tokenize(qa["answer"]):
                vocab.add_word(word)
                
    print(f"[Vocabulary] Built vocabulary with {len(vocab)} unique tokens.")
    return vocab


def train_model(
    dataset_dir: str = "data/sound_context_qa",
    artifacts_dir: str = "artifacts",
    epochs: int = EPOCHS
):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"[Training] Using Device: {DEVICE}")
    
    # 1. Build & Save Vocabulary
    vocab = build_vocabulary_from_dataset(dataset_dir)
    vocab_path = os.path.join(artifacts_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"word2idx": vocab.word2idx, "idx2word": vocab.idx2word}, f, indent=2)
    print(f"[Training] Saved vocabulary to '{vocab_path}'.")

    # 2. Prepare DataLoaders
    audio_dir = os.path.join(dataset_dir, "audio")
    train_ds = SoundQADataset(os.path.join(dataset_dir, "train.json"), audio_dir, vocab)
    val_ds = SoundQADataset(os.path.join(dataset_dir, "val.json"), audio_dir, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Instantiate Model
    model = AudioContextTransformerQA(vocab_size=len(vocab), d_model=256, nhead=4).to(DEVICE)
    
    criterion_ans = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    criterion_cat = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(artifacts_dir, "best_model.pt")

    print(f"[Training] Starting model training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            mel_specs = batch["mel_spec"].to(DEVICE)
            q_ids = batch["question_ids"].to(DEVICE)
            target_ans = batch["answer_id"].to(DEVICE)
            target_cat = batch["cat_id"].to(DEVICE)

            optimizer.zero_grad()
            out = model(mel_specs, q_ids)

            loss_ans = criterion_ans(out["logits"], target_ans)
            loss_cat = criterion_cat(out["qtype_logits"], target_cat)
            loss = loss_ans + 0.2 * loss_cat

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(mel_specs)

        scheduler.step()

        train_loss = running_loss / len(train_ds)

        # Validation Loop
        model.eval()
        val_running_loss = 0.0
        correct_ans = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                mel_specs = batch["mel_spec"].to(DEVICE)
                q_ids = batch["question_ids"].to(DEVICE)
                target_ans = batch["answer_id"].to(DEVICE)
                target_cat = batch["cat_id"].to(DEVICE)

                out = model(mel_specs, q_ids)
                loss_ans = criterion_ans(out["logits"], target_ans)
                loss_cat = criterion_cat(out["qtype_logits"], target_cat)
                loss = loss_ans + 0.2 * loss_cat

                val_running_loss += loss.item() * len(mel_specs)
                preds = torch.argmax(out["logits"], dim=-1)
                correct_ans += (preds == target_ans).sum().item()
                total_val += len(target_ans)

        val_loss = val_running_loss / len(val_ds)
        val_acc = (correct_ans / total_val) * 100.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab_size": len(vocab),
                "val_acc": val_acc
            }, best_model_path)
            print(f" -> Saved new best checkpoint with Val Acc {val_acc:.2f}% to '{best_model_path}'", flush=True)

    total_time = time.time() - start_time
    print(f"[Training] Training complete in {total_time/60:.2f} mins. Best Val Accuracy: {best_val_acc:.2f}%", flush=True)

    # Save Loss Curve Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, epochs + 1)
    
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#3b82f6", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#ef4444", linewidth=2)
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["val_accuracy"], label="Val Accuracy (%)", color="#10b981", linewidth=2)
    ax2.set_title("Validation Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_curve_path = os.path.join(artifacts_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"[Training] Loss curve plot saved to '{loss_curve_path}'.")

    with open(os.path.join(artifacts_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train_model(epochs=5)
