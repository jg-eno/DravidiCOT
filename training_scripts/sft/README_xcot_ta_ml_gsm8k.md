# xCOT-Style Multilingual SFT for Qwen3.5-2B

This repository contains a standalone training script extracted from the notebook
`xcot-ta-ml-gsm8k-fixed(1).ipynb`. It fine-tunes a LoRA adapter for
`Qwen/Qwen3.5-2B` using aligned English, Tamil, Malayalam, and Romanized
question variants from `jg-eno/xcot-ta_ml-gsm8k`.

> **Scope:** this is a multilingual supervised fine-tuning (SFT) baseline inspired
> by the xCOT direction. It is **not yet a complete implementation of the xCOT
> method**. xICL, Random Online CoT, mSampling, and token-level xDistill are not
> implemented in this pipeline.

## 1. Files

- `xcot_ta_ml_gsm8k_train.py` — standalone training script.
- `requirements_xcot.txt` — Python dependencies.
- `xcot_qwen35_lora/` — created during training; contains checkpoints, metrics,
  Trainer state, and the final LoRA adapter.

## 2. What the current pipeline does

1. **Loads the dataset** from `jg-eno/xcot-ta_ml-gsm8k`, using its `train` split.
2. **Builds language variants** for each source problem from:
   - English (`question_en`)
   - Tamil (`question_ta`)
   - Malayalam (`question_ml`)
   - Romanized Tamil (`question_ta_roman`)
   - Romanized Malayalam (`question_ml_roman`)
3. **Splits by source problem before expansion.** With seed `42` and a 5%
   validation fraction, all variants of a source problem stay together. This
   avoids leakage where one language version is in training and another
   version of the same problem is in validation.
4. **Uses English answer/reasoning targets** from `answer_en` for all question
   languages. The system prompt asks the model to reason in English and end
   with `####` followed by the final numeric answer.
5. **Renders chat templates as text, then tokenizes.** This avoids the nested
   token-ID issue encountered when using `apply_chat_template(..., tokenize=True)`
   in the notebook environment.
6. **Masks prompt tokens in the labels** with `-100`, so the supervised loss is
   calculated on the assistant answer rather than the system/user prompt.
7. **Filters invalid examples** and checks that token arrays are flat integer
   lists of matching lengths.
8. **Loads Qwen3.5-2B** using `AutoModelForMultimodalLM` and selects BF16 when
   supported, otherwise FP16. A CUDA GPU is required by the current script.
9. **Adds LoRA adapters** with rank `16`, alpha `32`, dropout `0.05`, and
   `target_modules="all-linear"`. The base model weights are not fully fine-tuned.
10. **Uses gradient checkpointing** to reduce activation memory use.
11. **Pads batches with a custom collator**, padding `input_ids`, `attention_mask`,
    and `labels` consistently; padded label positions use `-100`.
12. **Trains with Hugging Face Trainer**, evaluates on the held-out validation
    split each epoch when supported by the installed Transformers API, and saves
    checkpoints and training metrics.
13. **Saves the final adapter and processor/tokenizer files** under
    `xcot_qwen35_lora/final_adapter`.

## 3. Current training configuration

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` |
| Dataset | `jg-eno/xcot-ta_ml-gsm8k` (`train` split) |
| Languages/input variants | English, Tamil, Malayalam, Romanized Tamil, Romanized Malayalam |
| Target | English reasoning and answer (`answer_en`) |
| Seed | `42` |
| Validation fraction | `0.05` of source problems |
| Maximum sequence length | `1024` tokens |
| Fine-tuning method | LoRA / PEFT |
| LoRA rank (`r`) | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| LoRA target modules | `all-linear` |
| Epochs | `3` |
| Per-device train batch size | `1` |
| Gradient accumulation | `8` steps |
| Effective batch size | `8` examples per device before considering distributed training |
| Learning rate | `1e-4` |
| Scheduler | Cosine |
| Warmup | `3%` |
| Weight decay | `0.01` |
| Precision | BF16 if supported; otherwise FP16 |
| Gradient checkpointing | Enabled |
| Checkpointing | Save each epoch; retain up to 2 checkpoints |
| Logging | Every 10 steps, plus first step |
| External reporting | Disabled (`report_to="none"`) |

The number of expanded examples depends on which language fields are present in
the current Hugging Face dataset revision. The script prints source counts and
expanded train/validation counts at runtime; treat those logs as authoritative
for a particular run.

## 4. Environment setup

Use a fresh Python environment with a CUDA-enabled PyTorch build compatible with
the machine's NVIDIA driver. Install PyTorch using the official selector for
your CUDA/runtime setup if the default pip build does not provide CUDA support.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_xcot.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Check that CUDA is available before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

The script deliberately raises an error if CUDA is unavailable. It is designed
for GPU training, not CPU-only training. Install a Transformers version that
supports `Qwen/Qwen3.5-2B` and exposes `AutoModelForMultimodalLM`. Since model
support and APIs evolve, pin the exact package versions after a successful
reproducible run.

## 5. Run training

From the directory containing the script:

```bash
python xcot_ta_ml_gsm8k_train.py
```

The script downloads the dataset and model weights from Hugging Face on the
first run, so network access and sufficient disk space are needed. The default
output directory is `xcot_qwen35_lora/`.

Expected artifacts include:

- `checkpoint-*` directories — intermediate training checkpoints.
- `train_results.json` / `train_metrics` artifacts where produced by Trainer.
- `trainer_state.json` — training state and logged history.
- `final_adapter/` — final LoRA adapter and saved processor/tokenizer files.

The exact Trainer artifact names can vary slightly with Transformers version.
Back up the final adapter and logs after a successful run.

## 6. What has been addressed in the notebook

- Split the dataset by original source problem before creating language variants.
- Changed chat-template handling to render text first and tokenize it separately.
- Ensured `input_ids`, `attention_mask`, and `labels` are flat integer lists.
- Masked prompt tokens and padded labels with `-100`.
- Added a custom collator with length and type checks.
- Added gradient checkpointing and automatic BF16/FP16 selection.
- Added compatibility checks for `TrainingArguments` and `Trainer` parameter names.
- Checked a training batch before starting the full run.
- Saved the adapter, processor, metrics, and Trainer state.

These are workflow/code fixes present in the notebook; they should not be
interpreted as proof that a complete training run finished successfully. Confirm
completion from the actual logs and the presence of the final adapter.

## 7. Work still to do

### A. Verify and reproduce the baseline

- Run the script end-to-end and retain the complete logs.
- Confirm the final adapter directory is created and reloads successfully.
- Record exact versions of Python, PyTorch, Transformers, PEFT, Accelerate,
  Datasets, CUDA runtime, GPU model, and the Hugging Face dataset revision.
- Pin tested package versions in a lockfile or version-pinned requirements file.
- Record the actual number of source problems, expanded examples, trainable
  parameters, training duration, peak GPU memory, train loss, and validation loss.

### B. Evaluate the adapter

- Run a held-out evaluation on the original English GSM8K test set.
- Evaluate Tamil, Malayalam, Romanized Tamil, and Romanized Malayalam separately.
- Keep all evaluation examples out of training and validation.
- Report exact-match accuracy using robust numeric extraction and document
  treatment of commas, decimals, signs, malformed answers, and equivalent forms.
- Compare against the unmodified base model using the same prompts, decoding
  settings, and evaluation examples.
- Include per-language results, aggregate results, and error analysis.
- Verify that answers ending in `#### <number>` are extracted correctly; do not
  rely on loss alone as evidence of reasoning improvement.

### C. Complete the xCOT research method

The current script is multilingual SFT only. To claim a complete xCOT
implementation, design, implement, and ablate the following components according
to the intended paper/method specification:

- **xICL:** implement the cross-lingual in-context learning setup and define how
  demonstrations are selected and ordered.
- **Random Online CoT:** implement the intended online/randomized chain-of-thought
  training or sampling procedure, with reproducible random seeds.
- **mSampling:** implement the intended multilingual sampling strategy and
  document language proportions and sampling probabilities.
- **Token-level xDistill:** implement the intended token-level distillation
  objective and specify teacher outputs, alignment, masking, and loss weighting.

Do not label the current baseline as the full method until these components are
implemented and evaluated.

### D. Improve experimental rigor

- Add a fixed, documented evaluation protocol and multiple random seeds where
  compute allows.
- Compare LoRA settings and learning rates through controlled ablations.
- Check for translation quality and mathematical equivalence across language
  variants.
- Inspect long examples truncated at 1024 tokens and quantify truncation rates.
- Analyze language-specific failures, answer-format failures, and reasoning
  quality—not just final numeric accuracy.
- Consider saving a merged model only if required for deployment; keep the
  original adapter and base-model identifier for reproducibility.

## 8. Important limitations

- The target reasoning is English even for Tamil and Malayalam questions. This
  tests multilingual question understanding with English reasoning targets; it
  does not train Tamil/Malayalam reasoning generation.
- Romanized variants are separate input fields, but their presence alone does
  not constitute code-switching training.
- A source-level train/validation split helps prevent parallel-variant leakage,
  but independent test sets must still be kept separate.
- `AutoModelForMultimodalLM` and `Qwen3.5-2B` support depend on the installed
  Transformers version.
- Training completion and final quality cannot be inferred from the notebook
  source alone; confirm them from runtime logs and evaluation.

## 9. Suggested next milestone

**Milestone 1 — establish a reproducible baseline:** complete one training run,
reload the saved adapter, evaluate base model versus fine-tuned model on separate
English/Tamil/Malayalam/Romanized test sets, and publish a results table with
the exact software/hardware configuration.

Only after this baseline is reproducible should the additional xCOT components
be implemented one at a time and evaluated through ablation studies.
