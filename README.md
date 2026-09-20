# Audio Context Layer - Multimodal Audio Question Answering PoC

A system designed to build contextual understanding over temporal audio signals and answer natural language questions grounded in that audio across **Perceptual**, **Counting**, **Temporal**, and **Causal Reasoning** question categories.

---

## 🎓 Guide for Examiner & Evaluators

Welcome! This repository contains the complete end-to-end implementation for the **Audio Context Layer Internship Assignment**. 

### 1. Curated Dataset (`SoundContextQA`) Access
- **100% Annotated Metadata Included**: All **19,046 ground-truth QA pairs** across 1,000 clips are fully committed and viewable directly on GitHub in `data/sound_context_qa/`:
  - [`train.json`](data/sound_context_qa/train.json) (13,340 QA pairs)
  - [`val.json`](data/sound_context_qa/val.json) (2,856 QA pairs)
  - [`test.json`](data/sound_context_qa/test.json) (2,850 QA pairs)
  - [`metadata.json`](data/sound_context_qa/metadata.json) (Full clip & event timestamp metadata)
- **Included Sample Audio Clips**: Benchmark audio samples (`sample_0000.wav` through `sample_0009.wav`) are included in `data/sound_context_qa/audio/` so you can immediately run inference and test the Web UI out-of-the-box upon cloning.
- **Full Dataset Regeneration (1-Command)**: To regenerate all 1,000 16kHz WAV audio clips locally:
  ```bash
  python src/dataset_generator.py
  ```

### 2. Pre-Trained Model & Pre-Computed Benchmarks
- **Pre-Trained Model Checkpoint**: Located at [`artifacts/best_model.pt`](artifacts/best_model.pt) (28.4 MB).
- **Benchmark Evaluation Plot**: [`artifacts/evaluation_metrics.png`](artifacts/evaluation_metrics.png)
- **Training Loss Curve Plot**: [`artifacts/loss_curve.png`](artifacts/loss_curve.png)
- **Formal Technical Report**: Read [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for detailed problem formulation, literature research, architectural design justifications, loss curves, and app screenshots.

---

## 📁 Repository Directory Structure

```
audio_project/
├── data/                       # Grounded QA JSONs (train/val/test/metadata) & sample WAVs
│   └── sound_context_qa/
├── src/                        # Modular Python source code
│   ├── dataset_generator.py    # Procedural 16kHz audio engine & QA dataset builder
│   ├── model.py                # AudioContextTransformerQA neural architecture
│   ├── train.py                # PyTorch training loop & loss curve logger
│   ├── evaluate.py             # Benchmark evaluation (EM, BLEU, ROUGE)
│   └── infer.py                # Inference engine with attention map extraction
├── static/                     # Web app visual assets & stylesheets
│   ├── css/style.css           # Minimal dark design system stylesheet
│   └── js/app.js               # Visualizers & Chart.js heatmaps
├── templates/
│   └── index.html              # Interactive web dashboard layout
├── artifacts/                  # Model checkpoint, vocabulary & plot images
│   ├── best_model.pt           # Pre-trained model weights (28.4 MB)
│   ├── vocab.json              # Vocabulary mapping
│   ├── loss_curve.png          # Training & val loss plot
│   └── evaluation_metrics.png  # Test evaluation metrics plot
├── app.py                      # Flask REST API server
├── TECHNICAL_REPORT.md         # Formal structured technical document
└── README.md                   # Evaluator guide & quickstart instructions
```

---

## 🚀 Quickstart Instructions

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/jaskaransingh16/Audio-Context-Layer.git
cd audio_project

# Install required Python packages
pip install torch torchaudio flask scipy numpy matplotlib scikit-learn
```

### 2. Run Benchmark Evaluation
Evaluate the pre-trained model on the held-out test split (1,500 test questions):
```bash
python -m src.evaluate
```

### 3. Launch Interactive Web Application
Start the Flask web server:
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser to interact with the platform, inspect audio waveforms, view 80-channel mel-spectrograms, and analyze frame-level cross-attention heatmaps!

### 4. (Optional) Re-Train Model from Scratch
To regenerate audio clips and train a new model checkpoint:
```bash
python src/dataset_generator.py
python -m src.train
```

---

## 📊 Benchmark Summary

| Metric | Score |
| :--- | :---: |
| **Exact Match (EM) Accuracy** | **61.60%** (924 / 1500) |
| **Causal Reasoning Accuracy** | **100.00%** |
| **Perceptual Accuracy** | **70.60%** |
| **Counting Accuracy** | **69.89%** |
| **BLEU-1 Score** | **0.5051** |
| **ROUGE-L Score** | **0.5271** |

For detailed analysis, refer to [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
