"""
Audio Context Layer Model Architecture: AudioContextTransformerQA

Combines a Mel-Spectrogram Conv stem, a Temporal Audio Transformer Encoder, 
a Text Query Encoder, a Multi-Head Audio-Text Cross-Attention Fusion module,
and a Multi-Task Decoder Answering Head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class Vocabulary:
    """Tokenizer and vocabulary mapper for natural language questions and answers."""
    
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
    
    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        
        for token in self.SPECIAL_TOKENS:
            self.add_word(token)
            
    def add_word(self, word: str) -> int:
        word = word.lower().strip()
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def tokenize(self, text: str) -> List[str]:
        # Simple punctuation-aware tokenizer
        text = text.lower().replace("?", " ? ").replace(".", " . ").replace(",", " , ")
        return [w for w in text.split() if w]

    def encode(self, text: str, max_len: int = 32, add_bos_eos: bool = False) -> List[int]:
        tokens = self.tokenize(text)
        if add_bos_eos:
            tokens = ["<bos>"] + tokens + ["<eos>"]
        
        ids = [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]
        if len(ids) < max_len:
            ids = ids + [self.word2idx["<pad>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            word = self.idx2word.get(idx, "<unk>")
            if word in ["<pad>", "<bos>", "<eos>"]:
                if word == "<eos>":
                    break
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for temporal frames and text sequences."""
    
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


class AudioSpectrogramEncoder(nn.Module):
    """
    Computes Mel-Spectrogram features from raw 16kHz audio, projects through 2D Conv Stem,
    and contextualizes temporal frame embeddings using a 4-layer Audio Transformer.
    """
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80, d_model: int = 256, nhead: int = 4, num_layers: int = 4):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.d_model = d_model
        
        # Conv2D Stem for frequency-time downsampling (16x temporal stride)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 4), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Linear projection to d_model
        self.proj = nn.Linear(128 * 10, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def extract_mel_spectrogram(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Calculates log magnitude spectrogram directly in PyTorch using STFT across full 0-8000Hz range.
        waveforms shape: (B, N_samples)
        returns shape: (B, 1, 513, T)
        """
        n_fft = 1024
        hop_length = 320  # 20ms frame shift at 16kHz
        win_length = 800  # 50ms window
        
        window = torch.hann_window(win_length, device=waveforms.device)
        stft = torch.stft(
            waveforms, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True
        )
        magnitude = torch.abs(stft)  # (B, F_stft=513, T)
        log_spec = torch.log(magnitude + 1e-6)
        
        # Resize frequency axis from 513 to 80 bins using bilinear interpolation across full 0-8000Hz spectrum
        log_spec_4d = log_spec.unsqueeze(1)  # (B, 1, 513, T)
        mel_spec = F.interpolate(log_spec_4d, size=(self.n_mels, log_spec.size(-1)), mode='bilinear', align_corners=False)
        return mel_spec  # (B, 1, 80, T)

    def forward(self, mel_specs: torch.Tensor) -> torch.Tensor:
        # mel_specs: (B, 1, n_mels, T)
        x = self.conv_stem(mel_specs)  # (B, 128, F', T')
        B, C, F_prime, T_prime = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T_prime, C * F_prime)  # (B, T', C*F')
        
        x = self.proj(x)  # (B, T', d_model)
        x = self.pos_encoder(x)
        
        audio_features = self.transformer_encoder(x)  # (B, T', d_model)
        return audio_features


class ContextualAudioFusion(nn.Module):
    """
    Multi-Head Cross-Attention module querying temporal audio frame representations 
    with text question tokens. Outputs fused context and frame attention weights.
    """
    
    def __init__(self, d_model: int = 256, nhead: int = 4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self, query_feats: torch.Tensor, audio_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_feats: (B, L_text, d_model)
        audio_feats: (B, T_audio, d_model)
        Returns:
            fused_context: (B, L_text, d_model)
            attn_weights: (B, L_text, T_audio)
        """
        B, L, _ = query_feats.shape
        _, T, _ = audio_feats.shape
        
        Q = self.q_proj(query_feats).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # (B, H, L, d_k)
        K = self.k_proj(audio_feats).view(B, T, self.nhead, self.head_dim).transpose(1, 2)   # (B, H, T, d_k)
        V = self.v_proj(audio_feats).view(B, T, self.nhead, self.head_dim).transpose(1, 2)   # (B, H, T, d_k)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, T)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, T)
        
        context = torch.matmul(attn_weights, V)  # (B, H, L, d_k)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B, L, d_model)
        fused_context = self.out_proj(context)
        
        # Average attention weights across heads for visualization
        attn_mean = attn_weights.mean(dim=1)  # (B, L, T)
        return fused_context, attn_mean


class AudioContextTransformerQA(nn.Module):
    """
    End-to-End Audio Context Layer Model for Natural Language Question Answering over Audio.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_classes: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 1. Audio Spectrogram Encoder
        self.audio_encoder = AudioSpectrogramEncoder(d_model=d_model, nhead=nhead)
        
        # 2. Text Query Encoder
        self.text_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.text_pos = PositionalEncoding(d_model)
        text_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(text_encoder_layer, num_layers=3)
        
        # 3. Contextual Audio-Text Cross-Attention Fusion
        self.fusion = ContextualAudioFusion(d_model=d_model, nhead=nhead)
        
        # 4. Multi-Task Answering Heads
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, vocab_size)
        )
        
        # Auxiliary question-type head (perceptual, counting, temporal, causal)
        self.question_type_head = nn.Linear(d_model, 4)

    def forward(
        self, mel_specs: torch.Tensor, question_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        mel_specs: (B, 1, n_mels, T_spec)
        question_ids: (B, L_question)
        """
        # Encode audio timeline
        audio_feats = self.audio_encoder(mel_specs)  # (B, T_audio, d_model)
        
        # Encode text question
        text_x = self.text_embed(question_ids)
        text_x = self.text_pos(text_x)
        query_feats = self.text_encoder(text_x)  # (B, L_question, d_model)
        
        # Contextual Cross-Attention Fusion
        fused_context, attn_weights = self.fusion(query_feats, audio_feats)  # (B, L_question, d_model), (B, L_question, T_audio)
        
        # Pool query and fused representation
        pooled_query = query_feats.mean(dim=1)      # (B, d_model)
        pooled_fused = fused_context.mean(dim=1)    # (B, d_model)
        combined = torch.cat([pooled_query, pooled_fused], dim=-1)  # (B, d_model * 2)
        
        # Predict Answer Logits over Vocabulary
        logits = self.classifier_head(combined)  # (B, vocab_size)
        
        # Predict Question Type (0: perceptual, 1: counting, 2: temporal, 3: causal)
        qtype_logits = self.question_type_head(pooled_query)
        
        return {
            "logits": logits,
            "qtype_logits": qtype_logits,
            "attn_weights": attn_weights,
            "pooled_fused": pooled_fused
        }
