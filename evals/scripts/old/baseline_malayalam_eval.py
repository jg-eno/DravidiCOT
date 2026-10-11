import argparse
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForMultimodalLM

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


MODEL_ID = "Qwen/Qwen3.5-2B"
DATASET_ID = "jg-eno/ml-gsm8k-test"
SEED = 42
N_EVAL = 500
BATCH_SIZE = 4
MAX_NEW_TOKENS = 768
OUTPUT_DIR = Path("malayalam_gsm8k_eval")

SYSTEM_PROMPT = (
    "നിങ്ങൾ ഒരു ഗണിത പ്രശ്നപരിഹാര സഹായിയാണ്. "
    "ചോദ്യത്തിന് ഘട്ടംഘട്ടമായി മലയാളത്തിൽ ഉത്തരം നൽകുക. "
    "അവസാന വരിയിൽ #### എന്നതിന് ശേഷം അന്തിമ സംഖ്യാ ഉത്തരം നൽകുക. "
    "ഉദാഹരണത്തിന്: #### 42"
)


def normalize_number(value):
    if value is None:
        return None

    value = str(value).strip().replace(",", "")

    try:
        number = Decimal(value)
        if not number.is_finite():
            return None
        return format(number.normalize(), "f")
    except (InvalidOperation, ValueError):
        return None


def extract_gold_answer(text):
    matches = re.findall(
        r"####\s*<?\s*(-?\d[\d,]*(?:\.\d+)?)\s*>?",
        str(text),
    )
    return normalize_number(matches[-1]) if matches else None


def extract_prediction(text):
    patterns = [
        r"####\s*<?\s*(-?\d[\d,]*(?:\.\d+)?)\s*>?",
        r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?:final answer|the answer is|answer)\s*[:：]?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1])

    return None


def make_prompt(processor, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3.5-2B on the Malayalam GSM8K test set."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging (WANDB_API_KEY must be in .env)")
    parser.add_argument("--lora-adapter", type=str, default=None,
                        help="Path to LoRA adapter to load for evaluation")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model ID for LoRA adapter (required if --lora-adapter is set)")
    args = parser.parse_args()

    if args.n_eval < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--n-eval, --batch-size, and --max-new-tokens must be positive.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model name for wandb
    wandb_run_name = f"baseline-{args.model_id}"
    if args.lora_adapter:
        wandb_run_name = f"lora-{Path(args.lora_adapter).name}"

    if args.wandb:
        if not HAS_WANDB:
            raise ImportError("wandb is not installed. Install with: uv pip install wandb")
        wandb.init(
            project="xcot-malayalam-gsm8k-eval",
            name=wandb_run_name,
            config={
                "model_id": args.model_id,
                "dataset_id": args.dataset_id,
                "n_eval": args.n_eval,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "seed": SEED,
                "lora_adapter": args.lora_adapter,
            },
        )

    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"

    dataset = load_dataset(args.dataset_id, split="test")
    valid_rows = []

    for idx, row in enumerate(dataset):
        question = str(row["question_ml"] or "").strip()
        gold = extract_gold_answer(row["answer_en"])

        if question and gold is not None:
            valid_rows.append({
                "dataset_index": idx,
                "id": row["id"],
                "question": question,
                "gold": gold,
            })

    rng = random.Random(SEED)
    rng.shuffle(valid_rows)
    eval_rows = valid_rows[:min(args.n_eval, len(valid_rows))]

    if not eval_rows:
        raise RuntimeError("No valid evaluation examples found.")

    print(f"Dataset size: {len(dataset)}")
    print(f"Valid examples: {len(valid_rows)}")
    print(f"Evaluating: {len(eval_rows)}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()

    # Optionally load LoRA adapter
    if args.lora_adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter from: {args.lora_adapter}")
        model = PeftModel.from_pretrained(model, args.lora_adapter, torch_dtype=dtype)
        model = model.merge_and_unload()
        model = model.eval()
        print("LoRA adapter merged into model.")

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.use_cache = True

    print("Model loaded:", args.model_id)
    print("Device:", model.device)
    print("Tokenizer vocabulary size:", len(tokenizer))

    results = []

    for start_idx in tqdm(
        range(0, len(eval_rows), args.batch_size),
        desc="Evaluating",
    ):
        batch = eval_rows[start_idx:start_idx + args.batch_size]
        prompts = [make_prompt(processor, row["question"]) for row in batch]
        inputs = processor(text=prompts, padding=True, return_tensors="pt")
        inputs = {
            key: value.to(model.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_length:]
        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        for row, output in zip(batch, decoded):
            predicted_answer = extract_prediction(output)
            results.append({
                "dataset_index": row["dataset_index"],
                "id": row["id"],
                "question": row["question"],
                "gold_answer": row["gold"],
                "prediction": output,
                "predicted_answer": predicted_answer,
                "correct": (
                    predicted_answer is not None
                    and predicted_answer == row["gold"]
                ),
            })

    total = len(results)
    correct = sum(result["correct"] for result in results)
    extracted = sum(result["predicted_answer"] is not None for result in results)

    metrics = {
        "Evaluated examples": total,
        "Correct answers": correct,
        "Numeric exact-match accuracy (%)": round(100 * correct / total, 2) if total else 0.0,
        "Answer extraction coverage (%)": round(100 * extracted / total, 2) if total else 0.0,
        "Accuracy among extracted answers (%)": round(100 * correct / extracted, 2) if extracted else 0.0,
        "Unparseable predictions": total - extracted,
        "Incorrect extracted answers": extracted - correct,
    }

    with predictions_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\\n")

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("\\n" + "=" * 55)
    print("MALAYALAM GSM8K EVALUATION RESULTS")
    print("=" * 55)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\\nSaved files:")
    print("Predictions:", predictions_path)
    print("Metrics:", metrics_path)

    print("Sample incorrect predictions:")
    errors = [result for result in results if not result["correct"]]
    for result in errors[:5]:
        print("\nID:", result["id"])
        print("Gold:", result["gold_answer"])
        print("Predicted answer:", result["predicted_answer"])
        print("Prediction:", result["prediction"])

    if args.wandb:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
