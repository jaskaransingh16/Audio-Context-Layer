"""
Standalone Inference Engine for Audio Context Layer System

Processes any input audio file and question string, returning natural language answer,
question classification, mel-spectrogram data, and temporal cross-attention heatmaps.
"""

import os
import json
import torch
import numpy as np
from scipy.io import wavfile
from typing import Dict, Any, Tuple

from src.model import Vocabulary, AudioContextTransformerQA
from src.train import MAX_QUESTION_LEN

MAX_AUDIO_LEN = 128000  # 8 seconds at 16kHz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID_TO_CATEGORY = {
    0: "perceptual",
    1: "counting",
    2: "temporal",
    3: "causal"
}


class AudioContextInferenceEngine:
    """Inference wrapper for trained AudioContextTransformerQA model."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        vocab_path = os.path.join(artifacts_dir, "vocab.json")
        model_path = os.path.join(artifacts_dir, "best_model.pt")
        
        if not os.path.exists(vocab_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model or vocabulary in '{artifacts_dir}'. Please train the model first.")

        # Load Vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
        self.vocab = Vocabulary()
        self.vocab.word2idx = v_data["word2idx"]
        self.vocab.idx2word = {int(k): v for k, v in v_data["idx2word"].items()}

        # Load Model
        self.model = AudioContextTransformerQA(vocab_size=len(self.vocab), d_model=256, nhead=4)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(DEVICE)
        self.model.eval()
        
        print(f"[Inference Engine] Loaded model checkpoint from '{model_path}' on DEVICE: {DEVICE}")

    def predict(self, audio_path: str, question: str) -> Dict[str, Any]:
        """
        Executes QA inference over an audio file and question.
        Returns predicted answer, category, spectrogram data, and temporal attention map.
        """
        # Load WAV file
        sr, audio_int16 = wavfile.read(audio_path)
        audio_float = (audio_int16.astype(np.float32) / 32767.0)
        actual_duration = len(audio_float) / sr
        
        # Pad / Crop audio
        if len(audio_float) < MAX_AUDIO_LEN:
            padded_audio = np.pad(audio_float, (0, MAX_AUDIO_LEN - len(audio_float)))
        else:
            padded_audio = audio_float[:MAX_AUDIO_LEN]
            
        waveform_tensor = torch.tensor(padded_audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Encode question
        q_ids = self.vocab.encode(question, max_len=MAX_QUESTION_LEN)
        q_tensor = torch.tensor(q_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mel_spec = self.model.audio_encoder.extract_mel_spectrogram(waveform_tensor)
            out = self.model(mel_spec, q_tensor)
            
            # Predict answer masking special tokens (<pad>=0, <unk>=1, <bos>=2, <eos>=3)
            logits = out["logits"][0].clone()
            for s_idx in [0, 1, 2, 3]:
                logits[s_idx] = -1e9
                
            pred_id = torch.argmax(logits, dim=-1).item()
            pred_token = self.vocab.idx2word.get(pred_id, "<unk>")
            
            # Predict question category
            qtype_id = torch.argmax(out["qtype_logits"], dim=-1).item()
            predicted_cat = ID_TO_CATEGORY.get(qtype_id, "perceptual")
            
            # Extract temporal attention weights
            # attn_weights shape: (1, L_question, T_audio)
            attn_weights = out["attn_weights"][0].cpu().numpy()  # (L_question, T_audio)
            frame_attention = np.mean(attn_weights, axis=0)      # (T_audio,)
            
            # Normalize attention to [0, 1] range
            attn_min, attn_max = frame_attention.min(), frame_attention.max()
            if attn_max > attn_min:
                norm_attention = (frame_attention - attn_min) / (attn_max - attn_min)
            else:
                norm_attention = frame_attention

            # Extract mel-spectrogram for visualization
            spec_matrix = mel_spec[0, 0].cpu().numpy()  # (n_mels=80, T_spec)

        # Generate timestamps for frames
        num_frames = len(norm_attention)
        timestamps = np.linspace(0.0, 8.0, num_frames).tolist()

        return {
            "question": question,
            "answer": pred_token,
            "question_category": predicted_cat,
            "audio_duration_sec": round(float(actual_duration), 2),
            "temporal_attention": {
                "timestamps": [round(t, 2) for t in timestamps],
                "weights": [round(float(w), 4) for w in norm_attention]
            },
            "spectrogram": {
                "n_mels": spec_matrix.shape[0],
                "num_frames": spec_matrix.shape[1],
                "matrix": spec_matrix.tolist()
            }
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio Context QA Inference CLI")
    parser.add_argument("--audio", type=str, default="data/sound_context_qa/audio/sample_0000.wav", help="Path to input audio file (.wav)")
    parser.add_argument("--question", type=str, default="How many beep sounds occur in the audio?", help="Natural language question string")
    args = parser.parse_args()

    engine = AudioContextInferenceEngine()
    if os.path.exists(args.audio):
        res = engine.predict(args.audio, args.question)
        print(f"Audio Path      : {args.audio}")
        print(f"Sample Question : {res['question']}")
        print(f"Predicted Answer: {res['answer']}")
        print(f"Category        : {res['question_category']}")
    else:
        print(f"Error: Audio file not found at '{args.audio}'")
