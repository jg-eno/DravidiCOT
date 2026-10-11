#!/bin/bash
set -e

# Load environment from .env
cd /home/precision/Glen_Files/xCOT/eval
source .venv/bin/activate

# Export HF_TOKEN and WANDB_API_KEY from xCOT/.env
set -a
source /home/precision/Glen_Files/xCOT/.env
set +a

# Use GPU 0 (free after training completed)
export CUDA_VISIBLE_DEVICES=0

echo "=== Environment ==="
echo "HF_TOKEN set: ${HF_TOKEN:+yes}"
echo "WANDB_API_KEY set: ${WANDB_API_KEY:+yes}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Running BASELINE eval (Malayalam) ==="
python baseline_eval.py \
    --lang ml \
    --model-id Qwen/Qwen3.5-2B \
    --dataset-id jg-eno/ml-gsm8k-test \
    --n-eval 500 \
    --batch-size 4 \
    --max-new-tokens 768 \
    --output-dir eval_results \
    --wandb

echo ""
echo "=== Running LoRA-ADAPTER eval (Malayalam) ==="
python baseline_eval.py \
    --lang ml \
    --model-id Qwen/Qwen3.5-2B \
    --dataset-id jg-eno/ml-gsm8k-test \
    --n-eval 500 \
    --batch-size 4 \
    --max-new-tokens 768 \
    --output-dir eval_results \
    --lora-adapter /home/precision/Glen_Files/xCOT/training/xcot_qwen35_lora/final_adapter \
    --wandb

echo ""
echo "=== Running BASELINE eval (Tamil) ==="
python baseline_eval.py \
    --lang ta \
    --model-id Qwen/Qwen3.5-2B \
    --dataset-id Vishal0407/GSM8K_TAMIL \
    --n-eval 500 \
    --batch-size 4 \
    --max-new-tokens 768 \
    --output-dir eval_results \
    --wandb

echo ""
echo "=== Running LoRA-ADAPTER eval (Tamil) ==="
python baseline_eval.py \
    --lang ta \
    --model-id Qwen/Qwen3.5-2B \
    --dataset-id Vishal0407/GSM8K_TAMIL \
    --n-eval 500 \
    --batch-size 4 \
    --max-new-tokens 768 \
    --output-dir eval_results \
    --lora-adapter /home/precision/Glen_Files/xCOT/training/xcot_qwen35_lora/final_adapter \
    --wandb

echo "=== All evals complete ==="
