"""
Flask Web Server for Interactive Audio Context Layer PoC Application
"""

import os
import json
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from src.infer import AudioContextInferenceEngine

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configuration
UPLOAD_FOLDER = os.path.join("data", "uploads")
DATASET_DIR = os.path.join("data", "sound_context_qa")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

# Initialize Inference Engine
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = AudioContextInferenceEngine()
    return engine


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples")
def get_samples():
    """Returns test dataset samples for benchmark exploration."""
    test_json = os.path.join(DATASET_DIR, "test.json")
    if not os.path.exists(test_json):
        return jsonify({"error": "Test dataset not found"}), 440
        
    with open(test_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    # Return light metadata for first 30 samples
    light_samples = []
    for s in samples[:30]:
        light_samples.append({
            "sample_id": s["sample_id"],
            "audio_file": s["audio_file"],
            "duration": s["duration"],
            "environment": s["environment"],
            "event_count": len(s["events"]),
            "events": s["events"],
            "qa_count": len(s["qa_pairs"]),
            "qa_pairs": s["qa_pairs"]
        })
    return jsonify({"samples": light_samples})


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serves audio WAV files from dataset or uploads."""
    if os.path.exists(os.path.join(AUDIO_DIR, filename)):
        return send_from_directory(AUDIO_DIR, filename)
    elif os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
        return send_from_directory(UPLOAD_FOLDER, filename)
    else:
        return "Audio file not found", 404


@app.route("/api/predict", methods=["POST"])
def predict_qa():
    """Handles audio QA inference from dataset sample ID or uploaded file."""
    try:
        engine = get_engine()
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or request.form.get("question", "")).strip()
        sample_id = (data.get("sample_id") or request.form.get("sample_id", "")).strip()
        
        if not question:
            return jsonify({"error": "Question prompt is required"}), 400

        audio_path = None

        if "audio_file" in request.files and request.files["audio_file"].filename != "":
            file = request.files["audio_file"]
            filename = secure_filename(file.filename)
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(audio_path)
        elif sample_id:
            for split_name in ["test.json", "train.json", "val.json"]:
                split_json = os.path.join(DATASET_DIR, split_name)
                if os.path.exists(split_json):
                    with open(split_json, "r", encoding="utf-8") as f:
                        samples = json.load(f)
                    target_sample = next((s for s in samples if s["sample_id"] == sample_id), None)
                    if target_sample:
                        audio_path = os.path.join(AUDIO_DIR, target_sample["audio_file"])
                        break

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"error": "Valid audio file or sample ID is required"}), 400

        # Run inference
        result = engine.predict(audio_path, question)
        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark_metrics")
def get_benchmark_metrics():
    """Returns benchmark evaluation metrics and breakdown."""
    eval_json = os.path.join("artifacts", "evaluation_results.json")
    if os.path.exists(eval_json):
        with open(eval_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Benchmark metrics not found"}), 404


if __name__ == "__main__":
    print("[Web Server] Starting Audio Context Layer Application on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
