# Embedding alignment analysis for slide 5

The figure in `output/alignment_tsne.svg` compares the base `Qwen/Qwen3.5-2B`
with the adapter identified by the project owner,
`jg-eno/Qwen-3.5-sft-ta-ml`. This is the published **xCOT-style SFT adapter**;
the repository code does not establish that xICL, online CoT, mSampling, or
teacher-based xDistill contributed to these weights.

## Sample and method

- Recreated the source-level 5% validation split in `xcot_train.py` with seed
  42. All 66 validation records have the five required question fields.
- Sampled 20 distinct validation source records. The first 10 and second 10
  are tagged Tamil focus and Malayalam focus for the requested 10+10 view;
  every selected record contains **both** languages and all five forms.
- Encoded 100 raw question texts with each model stage, in the same order.
  Each vector is the mean of the final hidden states over its non-padding
  question tokens, normalized to unit length. Inputs were capped at 256 tokens.
- Loaded the base in 4-bit NF4 on the local RTX 3050 (4 GB), then attached the
  LoRA adapter without changing the base quantization. A single cosine-metric
  t-SNE fit over all 200 vectors provides comparable before/after panels.
- Used cosine similarity in the original 2,048-dimensional space for the
  numerical comparison. The interval below is a paired bootstrap over 20
  source records (10,000 resamples), not an interval on t-SNE distances.

| Matched-pair measure | Base | Adapter | Change |
|---|---:|---:|---:|
| Mean across all ten language pairs | 0.242 | 0.327 | +0.085 |
| English–Tamil | 0.066 | 0.228 | +0.162 |
| English–Malayalam | 0.028 | 0.182 | +0.154 |
| English–Romanized Tamil | 0.411 | 0.335 | −0.076 |
| English–Romanized Malayalam | 0.455 | 0.374 | −0.081 |

Paired bootstrap 95% interval for the all-pair mean change: **+0.074 to
+0.096**. The native-script pairs move closer to English in the original
embedding space, while both English–Romanized pairs move farther apart. The
t-SNE visualization still separates points strongly by script; it should not
be read as proof of semantic alignment or a quality gain on unseen problems.

## Reproduce

Use a CUDA-capable environment with PyTorch, Transformers, PEFT, bitsandbytes,
Accelerate, Datasets, NumPy, scikit-learn, and Matplotlib. The run used
Transformers 5.5.0 and PyTorch 2.11.0+cu130. From the repository root:

```bash
python reports/tsne/compute_alignment.py \
  --base Qwen/Qwen3.5-2B \
  --adapter jg-eno/Qwen-3.5-sft-ta-ml
```

Model revisions used in this run:

- Base: `15852e8c16360a2fea060d615a32b45270f8a8fc`
- Adapter: `f1e7812902f4e716b4e8c5e0850f65fad9bb7d26`
- Dataset fingerprint: `a6483374010d11d9`

`output/selected_records.json` records the exact source indices and question
texts. `output/embeddings_before.npy`, `output/embeddings_after.npy`, and
`output/tsne_coordinates.npy` preserve the computed arrays. `output/metrics.json`
contains all pairwise values, the sample indices, and environment details.
