#!/bin/bash
set -e

# Load environment from .env
cd /home/precision/Glen_Files/xCOT/training
VENVPY=/home/precision/Glen_Files/xCOT/eval/.venv/bin/python

# Export HF_TOKEN and WANDB_API_KEY from xCOT/.env
set -a
source /home/precision/Glen_Files/xCOT/.env
set +f

echo "=== Environment ==="
echo "HF_TOKEN set: ${HF_TOKEN:+yes}"
echo "WANDB_API_KEY set: ${WANDB_API_KEY:+yes}"
$VENVPY -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# Run the training script with YAML config
echo "=== Starting SFT training pipeline (config: train_config.yaml) ==="
$VENVPY xcot_ta_ml_gsm8k_train.py --config train_config.yaml

echo "=== Training pipeline complete ==="
