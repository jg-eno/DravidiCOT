#!/usr/bin/env python3
"""Measure question-embedding alignment before and after a Qwen3.5-2B adapter.

Run from the repository root with a CUDA-enabled environment containing torch,
transformers, peft, bitsandbytes, accelerate, datasets, numpy, scikit-learn, and
matplotlib. The same twenty complete held-out source rows are encoded in all
five forms before and after loading the adapter. This script does not train a
model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from sklearn.manifold import TSNE
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig


FIELDS = {
    "English": "question_en",
    "Tamil": "question_ta",
    "Malayalam": "question_ml",
    "Romanized Tamil": "question_ta_roman",
    "Romanized Malayalam": "question_ml_roman",
}
COLORS = {
    "English": "#182d49",
    "Tamil": "#b44a51",
    "Malayalam": "#176b77",
    "Romanized Tamil": "#df8e36",
    "Romanized Malayalam": "#6a5aad",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--adapter", required=True, help="Path or Hub ID for the exact trained adapter")
    parser.add_argument("--dataset", default="jg-eno/xcot-ta_ml-gsm8k")
    parser.add_argument("--output", type=Path, default=Path("reports/tsne/output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def select_rows(dataset, seed: int) -> list[dict]:
    usable = [
        i for i, row in enumerate(dataset)
        if str(row.get("answer_en") or "").strip()
        and any(str(row.get(f) or "").strip() for f in FIELDS.values())
    ]
    shuffled = usable.copy()
    random.Random(seed).shuffle(shuffled)
    validation = shuffled[:max(1, round(len(shuffled) * .05))]
    complete = [i for i in validation if all(str(dataset[i].get(f) or "").strip() for f in FIELDS.values())]
    if len(complete) < 20:
        raise RuntimeError(f"Only {len(complete)} held-out rows have all five forms; need twenty.")
    chosen = random.Random(seed + 1).sample(complete, 20)
    # Every row contains both languages. The two focus tags satisfy the requested
    # 10 + 10 sampling view; they do not imply separate language-only corpora.
    return [
        {
            "source_index": index,
            "focus": "Tamil" if position < 10 else "Malayalam",
            "questions": {name: dataset[index][field] for name, field in FIELDS.items()},
        }
        for position, index in enumerate(chosen)
    ]


def last_hidden(outputs) -> torch.Tensor:
    states = getattr(outputs, "hidden_states", None)
    if states is None and hasattr(outputs, "model_output"):
        states = getattr(outputs.model_output, "hidden_states", None)
    if not states:
        raise RuntimeError("Model did not return text hidden states; inspect its output API.")
    return states[-1]


def extract(model, tokenizer, rows: list[dict], max_length: int) -> np.ndarray:
    vectors = []
    model.eval()
    device = model.get_input_embeddings().weight.device
    for number, row in enumerate(rows, 1):
        for name, question in row["questions"].items():
            inputs = tokenizer(
                question, return_tensors="pt", truncation=True, max_length=max_length,
                add_special_tokens=True,
            ).to(device)
            with torch.inference_mode():
                output = model(
                    **inputs, output_hidden_states=True, return_dict=True,
                    use_cache=False, logits_to_keep=1,
                )
                hidden = last_hidden(output)[0]
                mask = inputs["attention_mask"][0].to(hidden.device).bool()
                pooled = hidden[mask].float().mean(dim=0)
                pooled = torch.nn.functional.normalize(pooled, dim=0)
            vectors.append(pooled.cpu().numpy())
        print(f"Embedded source {number}/20", flush=True)
    return np.stack(vectors)


def alignment_metrics(vectors: np.ndarray) -> dict:
    groups = vectors.reshape(20, 5, -1)
    by_pair = {}
    per_record_pairs = []
    names = list(FIELDS)
    for i in range(5):
        for j in range(i + 1, 5):
            similarities = np.sum(groups[:, i] * groups[:, j], axis=-1)
            by_pair[f"{names[i]} / {names[j]}"] = float(similarities.mean())
            per_record_pairs.append(similarities)
    return {
        "mean_all_pairs": float(np.mean(list(by_pair.values()))),
        "mean_english_to_other": float(np.mean([by_pair[f"English / {name}"] for name in names[1:]])),
        "per_record_mean_all_pairs": np.stack(per_record_pairs).mean(axis=0).tolist(),
        "pairwise_cosine_similarity": by_pair,
    }


def make_plot(before: np.ndarray, after: np.ndarray, rows: list[dict], output: Path, seed: int) -> None:
    combined = np.concatenate([before, after], axis=0)
    coords = TSNE(
        n_components=2, perplexity=20, init="pca", learning_rate="auto",
        metric="cosine", random_state=seed,
    ).fit_transform(combined)
    np.save(output / "tsne_coordinates.npy", coords)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.1), sharex=True, sharey=True, constrained_layout=True)
    labels = list(FIELDS)
    for stage, ax, xy in zip(("Before · base", "After · adapter"), axes, (coords[:100], coords[100:])):
        ax.set_title(stage, fontsize=17, fontweight="bold", loc="left", pad=14)
        for index, label in enumerate(labels):
            points = xy[index::5]
            ax.scatter(points[:, 0], points[:, 1], s=39, color=COLORS[label],
                       marker="o" if index < 3 else "s", alpha=.9,
                       edgecolors="white", linewidths=.55, label=label, zorder=2)
        ax.grid(color="#e7edf3", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    fig.savefig(output / "alignment_tsne.svg", bbox_inches="tight")
    fig.savefig(output / "alignment_tsne.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run outside the restricted sandbox on the host GPU.")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, split="train")
    rows = select_rows(dataset, args.seed)
    (args.output / "selected_records.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base, quantization_config=quantization,
        device_map="auto", dtype=torch.float16, low_cpu_mem_usage=True,
    )
    before = extract(model, tokenizer, rows, args.max_length)
    np.save(args.output / "embeddings_before.npy", before)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    after = extract(model, tokenizer, rows, args.max_length)
    np.save(args.output / "embeddings_after.npy", after)
    metrics = {
        "base_model": args.base,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "dataset_fingerprint": dataset._fingerprint,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "sampling": "20 complete rows from the source-level 5% validation split (seed 42); 10 tagged Tamil focus, 10 Malayalam focus; every row has all five forms",
        "source_indices": [row["source_index"] for row in rows],
        "max_length": args.max_length,
        "pooling": "mean of final hidden states over non-padding question tokens",
        "projection": "joint t-SNE; cosine metric; perplexity 20; PCA initialization",
        "before": alignment_metrics(before),
        "after": alignment_metrics(after),
    }
    deltas = np.array(metrics["after"]["per_record_mean_all_pairs"]) - np.array(metrics["before"]["per_record_mean_all_pairs"])
    bootstrap = np.random.default_rng(args.seed).choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    metrics["change_mean_all_pairs"] = float(deltas.mean())
    metrics["change_bootstrap_95pct_interval"] = [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])]
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    make_plot(before, after, rows, args.output, args.seed)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
