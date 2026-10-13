---
title: IndicReason-Tamil-TAPO
emoji: 🧮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
python_version: "3.12"
suggested_hardware: zero-a10g
models:
  - GL3MON/Qwen3.5-2B-Tamil-TAPO
---

# Tamil TAPO Math Solver

ZeroGPU Gradio demo for the merged Tamil TAPO weights. Uses the model card’s
system prompt, disabled thinking, and greedy decoding. Outputs an English
translation, worked solution, and boxed answer.

## CLI deployment

Install deployment dependencies into your preferred virtual environment:

```bash
python -m pip install 'huggingface_hub>=1,<2' python-dotenv
python hf_space/deploy.py
```

The script reads only HF_TOKEN from the repository root .env file, derives the
account name, creates a private Gradio Space named IndicReason-Tamil-TAPO, requests ZeroGPU,
and uploads only app.py, requirements.txt, and README.md. It never uploads .env
or adds your write token to the hosted app. The model is public.

Use `python hf_space/deploy.py USER/SPACE --public` to create a public Space.
An existing Space’s visibility is unchanged. Use a fresh name or an existing
Gradio Space. Account eligibility and token write permissions are required.
The CLI requests only ZeroGPU; it does not fall back to paid hardware.

## Runtime

The model loads at startup in BF16 and is placed on CUDA using ZeroGPU’s
emulation. GPU inference is decorated with `spaces.GPU(duration=120)`.
Requests run one at a time. The output limit ranges from 128 to 2048 tokens;
long generations may hit the allocation timeout. Quotas and queues apply.
Optional optimized DeltaNet kernels are omitted; PyTorch fallback is used.

Space variables: MODEL_ID (defaults to GL3MON/Qwen3.5-2B-Tamil-TAPO),
MODEL_REVISION (defaults to main; use a commit SHA for reproducibility), and
MAX_INPUT_TOKENS (defaults to 4096). Set HF_TOKEN as a read-only Space secret
only if you switch to a gated/private model.

For local inference, install requirements.txt and run app.py on a BF16-capable
CUDA GPU. ZeroGPU’s decorator has no effect outside ZeroGPU.
The Dockerfile is retained as an optional local container configuration and
is not uploaded or used for this Gradio Space.

## Sources

- [Model card](https://huggingface.co/GL3MON/Qwen3.5-2B-Tamil-TAPO)
- [ZeroGPU setup](https://huggingface.co/docs/hub/spaces-zerogpu)
