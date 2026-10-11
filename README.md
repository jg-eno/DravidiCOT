# DravidiCOT

DravidiCOT explores multilingual chain-of-thought fine-tuning and evaluation for mathematical reasoning in Tamil and Malayalam. The repository includes a configurable xCOT training pipeline, a separate multilingual LoRA supervised fine-tuning baseline, and scripts and artifacts for evaluating Tamil and Malayalam models on GSM8K-style problems.

## Repository contents

- `training_scripts/xcot/` — xCOT training script, configuration, dependencies, run script, and training log. The pipeline brings together cross-lingual demonstrations (xICL), multilingual sampling (mSampling), generated reasoning variants, and a token-level distillation component.
- `training_scripts/sft/` — separate multilingual LoRA SFT implementation and configuration, with its own walkthrough in `README_xcot_ta_ml_gsm8k.md`. Its Hugging Face model repository is [`jg-eno/Qwen-3.5-2B-sft-ta-ml`](https://huggingface.co/jg-eno/Qwen-3.5-sft-ta-ml).
- `evals/scripts/gpu/` — current evaluation script and launchers for Tamil and Malayalam, supporting both base-model and LoRA-adapter evaluation.

## Datasets

- **xCOT training:** [`jg-eno/xcot-ta_ml-gsm8k`](https://huggingface.co/datasets/jg-eno/xcot-ta_ml-gsm8k), `train` split. Provides English, Tamil, Malayalam, and Romanized Tamil/Malayalam question variants with English answers.
- **SFT training:** [`jg-eno/xcot-ta_ml-gsm8k`](https://huggingface.co/datasets/jg-eno/xcot-ta_ml-gsm8k), `train` split, as set in `training_scripts/sft/train_config.yaml`.
- **Malayalam evaluation:** [`jg-eno/ml-gsm8k-test`](https://huggingface.co/datasets/jg-eno/ml-gsm8k-test), `test` split.
- **Tamil evaluation:** [`Vishal0407/GSM8K_TAMIL`](https://huggingface.co/datasets/Vishal0407/GSM8K_TAMIL).
- **Local data:** `data/malayalam_train.jsonl` is included as a repository artifact; its use is not referenced by the current training configurations.
- `evals/scripts/old/` — earlier evaluation scripts and notebooks retained for reference.
- `evals/results/old/` — previously saved Tamil evaluation metrics and predictions.
- `data/` — local Malayalam JSONL data.

The training configurations use `Qwen/Qwen3.5-2B`. Training and evaluation require a compatible Python environment and CUDA-capable GPU; model and dataset downloads require network access. See the SFT walkthrough and YAML configurations for pipeline details.
