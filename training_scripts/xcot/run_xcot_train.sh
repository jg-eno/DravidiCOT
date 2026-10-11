#!/bin/bash
set -e

cd /home/precision/Glen_Files/xCOT/training
source /home/precision/Glen_Files/xCOT/eval/.venv/bin/activate

# Export HF_TOKEN and WANDB_API_KEY from xCOT/.env
set -a
source /home/precision/Glen_Files/xCOT/.env
set +a

echo "=== Environment ==="
echo "HF_TOKEN set: ${HF_TOKEN:+yes}"
echo "WANDB_API_KEY set: ${WANDB_API_KEY:+yes}"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== Starting xCOT training pipeline ==="
python -u xcot_train.py --config xcot_config.yaml

echo "=== xCOT training pipeline complete ==="
