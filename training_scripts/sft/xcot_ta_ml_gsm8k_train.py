"""Train a multilingual LoRA SFT adapter for Qwen3.5-2B.

This script follows the workflow in xcot-ta-ml-gsm8k-fixed(1).ipynb.
Run on a CUDA-enabled environment with enough GPU memory.

Configuration is loaded from a YAML file. Defaults to train_config.yaml.

Usage:
    python xcot_ta_ml_gsm8k_train.py [--config PATH]
"""

import argparse
import inspect
import math
import os
import random
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train a multilingual LoRA SFT adapter for Qwen3.5-2B."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ---- Config sections ----
    model_cfg = config["model"]
    ds_cfg = config["dataset"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    wandb_cfg = config.get("wandb", {})
    out_cfg = config["output"]
    env_cfg = config.get("env", {})

    # ---- Environment ----
    cuda_devices = str(env_cfg.get("cuda_visible_devices", ""))
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    MODEL_ID = model_cfg["model_id"]
    MAX_LENGTH = model_cfg["max_length"]
    REPO_ID = ds_cfg["repo_id"]
    DATASET_SPLIT = ds_cfg["split"]
    SEED = ds_cfg["seed"]
    VALIDATION_FRACTION = ds_cfg["validation_fraction"]
    LANGUAGE_FIELDS = ds_cfg["language_fields"]

    EPOCHS = train_cfg["epochs"]
    TRAIN_BATCH_SIZE = train_cfg["train_batch_size"]
    GRAD_ACCUM_STEPS = train_cfg["grad_accum_steps"]
    LEARNING_RATE = train_cfg["learning_rate"]
    LR_SCHEDULER_TYPE = train_cfg["lr_scheduler_type"]
    WARMUP_RATIO = train_cfg["warmup_ratio"]
    WEIGHT_DECAY = train_cfg["weight_decay"]
    GRADIENT_CHECKPOINTING = train_cfg["gradient_checkpointing"]

    bf16_setting = train_cfg.get("bf16", "auto")
    fp16_setting = train_cfg.get("fp16", "auto")

    if bf16_setting == "auto" and fp16_setting == "auto":
        bf16_supported = torch.cuda.is_bf16_supported()
        use_bf16 = bool(bf16_supported)
        use_fp16 = not use_bf16
    else:
        use_bf16 = bf16_setting is True or bf16_setting == "true"
        use_fp16 = fp16_setting is True or fp16_setting == "true"
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    OUTPUT_DIR = out_cfg["output_dir"]
    FINAL_ADAPTER_PATH = out_cfg["final_adapter_path"]

    # ---- Load dataset and split by source problem ----
    print(f"Loading dataset: {REPO_ID} (split: {DATASET_SPLIT})")
    raw_ds = load_dataset(REPO_ID, split=DATASET_SPLIT)
    print("Source records:", len(raw_ds))
    print("Columns:", raw_ds.column_names)

    source_records = []

    for source_index, row in enumerate(raw_ds):
        answer = str(row.get("answer_en") or "").strip()
        if not answer:
            continue

        variants = []
        for language, field in LANGUAGE_FIELDS.items():
            question = str(row.get(field) or "").strip()
            if question:
                variants.append({
                    "question": question,
                    "language": language,
                    "answer": answer,
                    "source_id": source_index,
                })

        if variants:
            source_records.append({
                "source_id": source_index,
                "variants": variants,
            })

    rng = random.Random(SEED)
    source_ids = [record["source_id"] for record in source_records]
    rng.shuffle(source_ids)

    n_val_sources = max(1, round(len(source_ids) * VALIDATION_FRACTION))
    val_source_ids = set(source_ids[:n_val_sources])

    train_examples = []
    val_examples = []

    for record in source_records:
        target = val_examples if record["source_id"] in val_source_ids else train_examples
        target.extend(record["variants"])

    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)

    print("Usable source problems:", len(source_records))
    print("Train source problems:", len(source_ids) - n_val_sources)
    print("Validation source problems:", n_val_sources)
    print("Expanded training examples:", len(train_ds))
    print("Expanded validation examples:", len(val_ds))
    print("Training example:", train_ds[0])

    # ---- Tokenize safely ----
    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    SYSTEM_PROMPT = (
        "You are a mathematical reasoning assistant. "
        "Solve the problem carefully. Explain the reasoning in English, "
        "even when the question is in another language. "
        "End with #### followed by the final numeric answer."
    )

    def render_chat(messages, add_generation_prompt):
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered, str):
            raise TypeError(
                f"Expected chat template to return str, got {type(rendered).__name__}"
            )
        return rendered

    def tokenize_example(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question language: {example['language']}\n"
                    f"Problem: {example['question']}"
                ),
            },
        ]

        prompt_text = render_chat(messages, add_generation_prompt=True)
        full_text = render_chat(
            messages + [{"role": "assistant", "content": example["answer"]}],
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # Use the common token prefix so a boundary-tokenization difference
        # cannot accidentally mask answer tokens or create a length mismatch.
        common_prefix_len = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids):
            if prompt_token != full_token:
                break
            common_prefix_len += 1

        full_ids = full_ids[:MAX_LENGTH]

        if common_prefix_len >= MAX_LENGTH or not full_ids:
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        masked_len = min(common_prefix_len, len(full_ids))
        labels = [-100] * masked_len + full_ids[masked_len:]

        return {
            "input_ids": [int(token) for token in full_ids],
            "attention_mask": [1] * len(full_ids),
            "labels": [int(token) for token in labels],
        }

    def tokenize_dataset(dataset):
        result = dataset.map(
            tokenize_example,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        result = result.filter(
            lambda row: (
                len(row["input_ids"]) > 0
                and len(row["input_ids"]) == len(row["attention_mask"])
                and len(row["input_ids"]) == len(row["labels"])
            ),
            desc="Filtering invalid/empty sequences",
        )
        return result

    tokenized_train = tokenize_dataset(train_ds)
    tokenized_val = tokenize_dataset(val_ds)

    def validate_tokenized_dataset(dataset, name):
        assert len(dataset) > 0, f"{name} dataset is empty."
        for i in range(min(10, len(dataset))):
            row = dataset[i]
            for key in ("input_ids", "attention_mask", "labels"):
                value = row[key]
                assert isinstance(value, list), (name, i, key, type(value))
                assert all(isinstance(x, int) for x in value), (
                    name, i, key, type(value[0]) if value else None
                )
            assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])
        print(f"{name}: {len(dataset)} valid examples; flat token lists verified.")

    validate_tokenized_dataset(tokenized_train, "Train")
    validate_tokenized_dataset(tokenized_val, "Validation")
    print(tokenized_train[0])

    # ---- Load Qwen3.5-2B and attach LoRA ----
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training configuration.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute dtype:", compute_dtype)

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
    )

    model.config.use_cache = False

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Custom collator and Trainer ----
    class XCOTDataCollator:
        def __init__(self, tokenizer, pad_to_multiple_of=8):
            self.tokenizer = tokenizer
            self.pad_to_multiple_of = pad_to_multiple_of

        def __call__(self, features):
            if not features:
                raise ValueError("Received an empty batch.")

            for i, feature in enumerate(features):
                for key in ("input_ids", "attention_mask", "labels"):
                    value = feature[key]
                    if not isinstance(value, (list, tuple)):
                        raise TypeError(f"Feature {i} {key} is {type(value).__name__}, expected list.")
                    if not all(isinstance(x, int) for x in value):
                        raise TypeError(f"Feature {i} {key} contains non-integer/nested values.")

            max_len = max(len(feature["input_ids"]) for feature in features)
            m = self.pad_to_multiple_of
            if m:
                max_len = ((max_len + m - 1) // m) * m

            pad_id = self.tokenizer.pad_token_id
            batch_input_ids, batch_attention, batch_labels = [], [], []

            for feature in features:
                ids = list(feature["input_ids"])
                mask = list(feature["attention_mask"])
                labels = list(feature["labels"])

                if not (len(ids) == len(mask) == len(labels)):
                    raise ValueError("input_ids, attention_mask, and labels lengths differ.")

                pad_len = max_len - len(ids)
                batch_input_ids.append(ids + [pad_id] * pad_len)
                batch_attention.append(mask + [0] * pad_len)
                batch_labels.append(labels + [-100] * pad_len)

            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

    data_collator = XCOTDataCollator(tokenizer)

    train_steps_per_epoch = math.ceil(
        len(tokenized_train) / (TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS)
    )
    total_training_steps = train_steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_training_steps * WARMUP_RATIO))

    # ---- Initialize wandb ----
    if HAS_WANDB and wandb_cfg.get("enabled", True):
        wandb.init(
            project=wandb_cfg.get("project", "xcot-multilingual-sft"),
            name=wandb_cfg.get("name", f"sft-{MODEL_ID}-lora-r{lora_cfg['r']}-{EPOCHS}ep"),
            config={
                "model_id": MODEL_ID,
                "dataset_id": REPO_ID,
                "epochs": EPOCHS,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "learning_rate": LEARNING_RATE,
                "lr_scheduler_type": LR_SCHEDULER_TYPE,
                "warmup_steps": warmup_steps,
                "weight_decay": WEIGHT_DECAY,
                "lora_r": lora_cfg["r"],
                "lora_alpha": lora_cfg["lora_alpha"],
                "lora_dropout": lora_cfg["lora_dropout"],
                "max_length": MAX_LENGTH,
                "validation_fraction": VALIDATION_FRACTION,
                "seed": SEED,
                "compute_dtype": str(compute_dtype),
                "train_examples": len(tokenized_train),
                "val_examples": len(tokenized_val),
            },
        )

    # ---- TrainingArguments ----
    ta_params = inspect.signature(TrainingArguments.__init__).parameters

    training_cfg = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "warmup_steps": warmup_steps,
        "weight_decay": WEIGHT_DECAY,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "save_strategy": train_cfg["save_strategy"],
        "save_total_limit": train_cfg["save_total_limit"],
        "logging_steps": train_cfg["logging_steps"],
        "logging_first_step": train_cfg["logging_first_step"],
        "report_to": "none" if not (HAS_WANDB and wandb_cfg.get("enabled", True)) else "wandb",
        "remove_unused_columns": train_cfg["remove_unused_columns"],
        "dataloader_pin_memory": train_cfg["dataloader_pin_memory"],
    }

    if train_cfg.get("load_best_model_at_end", False):
        training_cfg["load_best_model_at_end"] = True
        training_cfg["metric_for_best_model"] = "eval_loss"
        training_cfg["greater_is_better"] = False
        training_cfg["save_strategy"] = "epoch"

    if "eval_strategy" in ta_params:
        training_cfg["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_params:
        training_cfg["evaluation_strategy"] = "epoch"

    if "gradient_checkpointing_kwargs" in ta_params:
        training_cfg["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if "warmup_ratio" in ta_params:
        training_cfg.pop("warmup_steps", None)
        training_cfg["warmup_ratio"] = WARMUP_RATIO

    training_args = TrainingArguments(**training_cfg)

    # ---- Trainer ----
    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "data_collator": data_collator,
    }

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # ---- Add callbacks (early stopping) ----
    callbacks = []
    if "early_stopping_patience" in train_cfg:
        patience = train_cfg["early_stopping_patience"]
        if patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    if callbacks:
        trainer.callback_handler.callbacks.extend(callbacks)

    # Verify collation before starting a full run.
    batch = next(iter(trainer.get_train_dataloader()))
    for key, value in batch.items():
        print(key, tuple(value.shape), value.dtype)

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape == batch["labels"].shape
    print("Batch collation successful.")

    # ---- Train and save the adapter ----
    train_result = trainer.train()

    trainer.save_model(FINAL_ADAPTER_PATH)
    processor.save_pretrained(FINAL_ADAPTER_PATH)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Training complete.")
    print("LoRA adapter saved to:", FINAL_ADAPTER_PATH)

    if HAS_WANDB and wandb_cfg.get("enabled", True):
        wandb.finish()


if __name__ == "__main__":
    main()
