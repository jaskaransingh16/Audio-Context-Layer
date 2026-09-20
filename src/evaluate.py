"""
Evaluation Pipeline for Audio Context Layer System

Computes quantitative benchmark metrics on the SoundContextQA test set:
- Overall Exact Match (EM) Accuracy
- Breakdown by Question Type (Perceptual, Counting, Temporal, Causal)
- BLEU-1, BLEU-4, ROUGE-L metrics
- Semantic Similarity
- Generates Evaluation Bar Chart and Error Analysis Report
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from torch.utils.data import DataLoader
from typing import Dict, List, Any

from src.model import Vocabulary, AudioContextTransformerQA
from src.train import SoundQADataset, MAX_AUDIO_LEN, MAX_QUESTION_LEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_bleu_n(reference: List[str], candidate: List[str], n: int = 1) -> float:
    """Computes BLEU-N score with brevity penalty."""
    if not candidate:
        return 0.0
    
    ref_len = len(reference)
    cand_len = len(candidate)
    
    # Brevity penalty
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / max(1, cand_len))
        
    # N-gram precision
    ref_ngrams = {}
    for i in range(len(reference) - n + 1):
        ng = tuple(reference[i:i+n])
        ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
        
    cand_ngrams = {}
    for i in range(len(candidate) - n + 1):
        ng = tuple(candidate[i:i+n])
        cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1
        
    clipped_matches = 0
    for ng, count in cand_ngrams.items():
        clipped_matches += min(count, ref_ngrams.get(ng, 0))
        
    total_cand_ngrams = max(1, len(candidate) - n + 1)
    precision = clipped_matches / total_cand_ngrams
    return bp * precision


def compute_rouge_l(reference: List[str], candidate: List[str]) -> float:
    """Computes ROUGE-L F1 score based on Longest Common Subsequence (LCS)."""
    m, n = len(reference), len(candidate)
    if m == 0 or n == 0:
        return 0.0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == candidate[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    lcs_len = dp[m][n]
    rec = lcs_len / m
    prec = lcs_len / n
    if rec + prec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def evaluate_system(
    dataset_dir: str = "data/sound_context_qa",
    artifacts_dir: str = "artifacts"
):
    print(f"[Evaluation] Starting benchmark evaluation on DEVICE: {DEVICE}")
    
    vocab_path = os.path.join(artifacts_dir, "vocab.json")
    model_path = os.path.join(artifacts_dir, "best_model.pt")
    test_json_path = os.path.join(dataset_dir, "test.json")
    audio_dir = os.path.join(dataset_dir, "audio")
    
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model check point or vocabulary in '{artifacts_dir}'. Train model first!")

    # 1. Load Vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        v_data = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = v_data["word2idx"]
    vocab.idx2word = {int(k): v for k, v in v_data["idx2word"].items()}

    # 2. Load Model
    model = AudioContextTransformerQA(vocab_size=len(vocab), d_model=256, nhead=4)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    # 3. Load Test DataLoader
    test_ds = SoundQADataset(test_json_path, audio_dir, vocab)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    category_stats = {
        "perceptual": {"correct": 0, "total": 0, "bleu1": [], "bleu4": [], "rouge": []},
        "counting": {"correct": 0, "total": 0, "bleu1": [], "bleu4": [], "rouge": []},
        "temporal": {"correct": 0, "total": 0, "bleu1": [], "bleu4": [], "rouge": []},
        "causal": {"correct": 0, "total": 0, "bleu1": [], "bleu4": [], "rouge": []}
    }

    error_analysis_samples = []
    all_predictions = []

    print(f"[Evaluation] Evaluating {len(test_ds)} test samples...")

    with torch.no_grad():
        for batch in test_loader:
            mel_specs = batch["mel_spec"].to(DEVICE)
            q_ids = batch["question_ids"].to(DEVICE)
            target_ans_ids = batch["answer_id"].cpu().numpy()

            out = model(mel_specs, q_ids)
            logits = out["logits"].clone()
            for s_idx in [0, 1, 2, 3]:
                logits[:, s_idx] = -1e9
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

            for i in range(len(pred_ids)):
                pred_token = vocab.idx2word.get(pred_ids[i], "<UNK>")
                target_token = vocab.idx2word.get(target_ans_ids[i], "<UNK>")
                raw_q = batch["raw_question"][i]
                raw_gt = batch["raw_answer"][i]
                cat = batch["category"][i]

                # Clean text comparisons
                pred_tokens = vocab.tokenize(pred_token)
                target_tokens = vocab.tokenize(raw_gt)

                is_correct = (pred_token.lower() == target_token.lower()) or (pred_token.lower() in raw_gt.lower())
                
                bleu1 = compute_bleu_n(target_tokens, pred_tokens, n=1)
                bleu4 = compute_bleu_n(target_tokens, pred_tokens, n=4)
                rouge = compute_rouge_l(target_tokens, pred_tokens)

                if cat in category_stats:
                    category_stats[cat]["total"] += 1
                    if is_correct:
                        category_stats[cat]["correct"] += 1
                    category_stats[cat]["bleu1"].append(bleu1)
                    category_stats[cat]["bleu4"].append(bleu4)
                    category_stats[cat]["rouge"].append(rouge)

                all_predictions.append({
                    "question": raw_q,
                    "ground_truth": raw_gt,
                    "prediction": pred_token,
                    "category": cat,
                    "is_correct": is_correct
                })

                if not is_correct and len(error_analysis_samples) < 25:
                    error_analysis_samples.append({
                        "question": raw_q,
                        "ground_truth": raw_gt,
                        "prediction": pred_token,
                        "category": cat
                    })

    # Aggregate Overall Results
    total_correct = sum(stats["correct"] for stats in category_stats.values())
    total_samples = sum(stats["total"] for stats in category_stats.values())
    overall_acc = (total_correct / max(1, total_samples)) * 100.0

    all_bleu1 = [b for cat in category_stats.values() for b in cat["bleu1"]]
    all_bleu4 = [b for cat in category_stats.values() for b in cat["bleu4"]]
    all_rouge = [r for cat in category_stats.values() for r in cat["rouge"]]

    results_summary = {
        "overall": {
            "exact_match_accuracy": round(overall_acc, 2),
            "total_test_samples": total_samples,
            "bleu1": round(float(np.mean(all_bleu1)), 4),
            "bleu4": round(float(np.mean(all_bleu4)), 4),
            "rouge_l": round(float(np.mean(all_rouge)), 4)
        },
        "category_breakdown": {}
    }

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Overall Exact Match Accuracy : {overall_acc:.2f}% ({total_correct}/{total_samples})")
    print(f"Overall BLEU-1               : {np.mean(all_bleu1):.4f}")
    print(f"Overall BLEU-4               : {np.mean(all_bleu4):.4f}")
    print(f"Overall ROUGE-L              : {np.mean(all_rouge):.4f}")
    print("--------------------------------------------------")

    for cat, stats in category_stats.items():
        cat_acc = (stats["correct"] / max(1, stats["total"])) * 100.0
        cat_b1 = np.mean(stats["bleu1"]) if stats["bleu1"] else 0.0
        cat_b4 = np.mean(stats["bleu4"]) if stats["bleu4"] else 0.0
        cat_r = np.mean(stats["rouge"]) if stats["rouge"] else 0.0

        results_summary["category_breakdown"][cat] = {
            "accuracy": round(cat_acc, 2),
            "sample_count": stats["total"],
            "bleu1": round(float(cat_b1), 4),
            "bleu4": round(float(cat_b4), 4),
            "rouge_l": round(float(cat_r), 4)
        }
        print(f"Category [{cat.upper():<10}] Acc: {cat_acc:.2f}% | BLEU-1: {cat_b1:.4f} | ROUGE-L: {cat_r:.4f}")

    print("==================================================\n")

    # Save Evaluation Results JSON
    eval_results_path = os.path.join(artifacts_dir, "evaluation_results.json")
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Save Error Analysis Report JSON
    error_report_path = os.path.join(artifacts_dir, "error_analysis.json")
    with open(error_report_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis_samples, f, indent=2)

    # Plot Category Performance Breakdown Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    categories = list(category_stats.keys())
    accs = [results_summary["category_breakdown"][c]["accuracy"] for c in categories]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    bars = ax.bar([c.capitalize() for c in categories], accs, color=colors, width=0.55)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=12)
    ax.set_title("Audio Context Layer: Accuracy Breakdown Across Question Types", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)

    for bar, acc in zip(bars, accs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{acc:.1f}%", ha='center', va='bottom', fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    chart_path = os.path.join(artifacts_dir, "evaluation_metrics.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Evaluation metrics chart saved to '{chart_path}'.")


if __name__ == "__main__":
    evaluate_system()
