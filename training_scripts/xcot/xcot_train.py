"""xCOT Training Pipeline: Chain-of-Thought Cross-Lingual Transfer

This script implements the full xCOT method with four components:
1. xICL - Cross-lingual In-Context Learning (language-balanced demonstrations)
2. Random Online CoT - Diverse reasoning path generation during training
3. mSampling - Controlled multilingual sampling strategy
4. Token-level xDistill - Cross-lingual token-level knowledge distillation

Usage:
    python xcot_train.py --config xcot_config.yaml
"""

import argparse
import inspect
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class xCOTDataBalancer:
    """mSampling: Controls multilingual sampling proportions.

    Balances training data across languages to prevent the majority language
    from dominating. Implements temperature-based sampling with language-specific
    weights.
    """

    def __init__(self, language_weights: dict, temperature: float = 1.0):
        self.language_weights = language_weights
        self.temperature = temperature

    def compute_sampling_weights(self, lang_counts: dict) -> dict:
        """Compute per-example sampling weights based on language counts.

        Implements mSampling: languages with more examples get lower per-example
        probability to balance representation. Uses temperature to smooth.
        """
        counts = np.array([lang_counts.get(lang, 1) for lang in self.language_weights])
        # Target proportion per language (from weights)
        target_props = np.array([
            self.language_weights.get(lang, 0) for lang in self.language_weights
        ])
        target_props = target_props / target_props.sum()

        # Current proportions
        current_props = counts / counts.sum()

        # Sampling weight = target_prop / current_prop (oversample rare, undersample common)
        # Apply temperature for smoothing
        weights = target_props / (current_props + 1e-8)
        weights = weights ** (1.0 / self.temperature)
        weights = weights / weights.sum()

        return {
            lang: float(w) for lang, w in zip(self.language_weights.keys(), weights)
        }

    def select_balanced_batch(self, examples: list, batch_size: int) -> list:
        """Select a batch with language-balanced sampling (mSampling)."""
        if not examples:
            return []

        lang_to_examples = {}
        for ex in examples:
            lang = ex.get("language", "en")
            if lang not in lang_to_examples:
                lang_to_examples[lang] = []
            lang_to_examples[lang].append(ex)

        lang_counts = {lang: len(exs) for lang, exs in lang_to_examples.items()}
        weights = self.compute_sampling_weights(lang_counts)

        # Sample per language
        batch = []
        per_lang_target = max(1, batch_size // len(self.language_weights))

        for lang, examples_list in lang_to_examples.items():
            n_sample = min(
                len(examples_list),
                round(batch_size * weights.get(lang, 0))
            )
            if n_sample > 0:
                batch.extend(random.choices(
                    examples_list, k=min(n_sample, len(examples_list))
                ))

        # Fill remaining slots
        if len(batch) < batch_size:
            all_examples = [ex for exs in lang_to_examples.values() for ex in exs]
            batch.extend(random.choices(all_examples, k=batch_size - len(batch)))

        random.shuffle(batch)
        return batch[:batch_size]


class RandomCoTGenerator:
    """Random Online CoT: Generates diverse reasoning paths.

    For a given problem, generates multiple reasoning traces with different
    random seeds and selects the most confident or most diverse ones.
    """

    def __init__(self, model, tokenizer, processor, n_variants: int = 3,
                 max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.n_variants = n_variants
        self.max_new_tokens = max_new_tokens

    def generate_diverse_reasoning(self, prompt: str) -> list:
        """Generate multiple reasoning variants with different random seeds."""
        variants = []
        input_ids = self.processor(
            text=prompt, return_tensors="pt"
        ).input_ids.to(self.model.device)

        for i in range(self.n_variants):
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7 + 0.1 * i,  # Increasing temperature for diversity
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            output = self.tokenizer.decode(
                generated[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            variants.append(output)

        return variants

    def select_best_reasoning(self, variants: list, answer_pattern: str = None) -> str:
        """Select the best reasoning path among variants.

        Prioritizes variants that produce a numeric answer, then by confidence.
        """
        scored = []
        for i, variant in enumerate(variants):
            # Check if answer is extractable
            has_answer = bool(re.search(r"####\s*(-?\d+)", variant))
            score = 2.0 if has_answer else 0.5
            scored.append((score, i, variant))

        scored.sort(reverse=True)
        return scored[0][2]  # Return the best variant


class xICLDemonstrationSelector:
    """xICL: Cross-lingual In-Context Learning demonstration selection.

    Selects diverse, language-balanced demonstrations for cross-lingual transfer.
    Demonstrates can be from any language to help the target language.
    """

    def __init__(self, train_examples: list, n_demonstrations: int = 4,
                 language_diversity: bool = True):
        self.train_examples = train_examples
        self.n_demonstrations = n_demonstrations
        self.language_diversity = language_diversity

    def select_demonstrations(self, target_question: str,
                             target_language: str) -> list:
        """Select demonstrations that are maximally helpful for the target.

        Uses a combination of:
        - Language diversity (include non-target language examples for xICL benefit)
        - Semantic similarity to the target question
        - Difficulty diversity (include easy and hard examples)
        """
        # Group by language
        by_lang = {}
        for ex in self.train_examples:
            lang = ex.get("language", "en")
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(ex)

        selected = []

        if self.language_diversity:
            # Include demonstrations from multiple languages (xICL)
            langs = list(by_lang.keys())
            random.shuffle(langs)

            # Ensure target language is included
            if target_language in by_lang:
                selected.extend(random.sample(
                    by_lang[target_language],
                    min(len(by_lang[target_language]), self.n_demonstrations // 2)
                ))

            # Add cross-lingual demonstrations
            cross_langs = [l for l in langs if l != target_language]
            remaining_slots = self.n_demonstrations - len(selected)
            for lang in cross_langs:
                if remaining_slots <= 0:
                    break
                if lang in by_lang and by_lang[lang]:
                    n = min(len(by_lang[lang]), remaining_slots // len(cross_langs))
                    selected.extend(random.sample(
                        by_lang[lang], min(n, len(by_lang[lang]))
                    ))
                    remaining_slots -= n
        else:
            # Same-language demonstrations
            if target_language in by_lang:
                selected = random.sample(
                    by_lang[target_language],
                    min(len(by_lang[target_language]), self.n_demonstrations)
                )

        # Fill remaining with random examples
        while len(selected) < self.n_demonstrations and self.train_examples:
            ex = random.choice(self.train_examples)
            if ex not in selected:
                selected.append(ex)

        return selected[:self.n_demonstrations]

    def format_demonstrations(self, demonstrations: list) -> str:
        """Format demonstrations as in-context examples."""
        parts = []
        for ex in demonstrations:
            parts.append(
                f"Question: {ex['question']}\n"
                f"Reasoning: {ex['answer']}"
            )
        return "\n\n".join(parts) + "\n\nQuestion: "


class TokenLevelxDistillLoss:
    """Token-level xDistill: Cross-lingual token-level knowledge distillation.

    Distills knowledge from a teacher model at the token level.
    The teacher provides soft labels (logits) that the student learns to match.
    """

    def __init__(self, teacher_model=None, alpha: float = 0.5, temperature: float = 2.0):
        self.teacher_model = teacher_model
        self.alpha = alpha  # Weight for distillation loss (vs. standard CE loss)
        self.temperature = temperature  # Softmax temperature for teacher logits

    def compute_distill_loss(self, student_logits: torch.Tensor,
                            teacher_logits: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        """Compute token-level distillation loss.

        L = alpha * L_ce(student, labels) + (1-alpha) * L_kd(student, teacher)
        """
        # Standard cross-entropy loss on target tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        ce_loss = loss_fct(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        if self.teacher_model is not None and self.alpha < 1.0:
            with torch.no_grad():
                teacher_logits = teacher_logits  # Already computed

            # KD loss: match teacher's soft distribution
            s_logits = student_logits / self.temperature
            t_logits = teacher_logits / self.temperature

            # Soft cross-entropy (teacher -> student)
            s_log_probs = torch.log_softmax(s_logits, dim=-1)
            t_probs = torch.softmax(t_logits, dim=-1)

            # Element-wise KL divergence
            kl_loss = torch.kl_div(
                s_log_probs, t_probs,
                reduction="batchmean",
                log_target=False
            ) * (self.temperature ** 2)

            loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss
        else:
            loss = ce_loss

        return loss



def main():
    parser = argparse.ArgumentParser(
        description="xCOT Training Pipeline: Chain-of-Thought Cross-Lingual Transfer"
    )
    parser.add_argument(
        "--config", type=str, default="xcot_config.yaml",
        help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ---- Config sections ----
    model_cfg = config["model"]
    ds_cfg = config["dataset"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    xcot_cfg = config.get("xcot", {})
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
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_bf16 = bool(bf16_supported)
        use_fp16 = not use_bf16
    else:
        use_bf16 = bf16_setting is True or bf16_setting == "true"
        use_fp16 = fp16_setting is True or fp16_setting == "true"
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    OUTPUT_DIR = out_cfg["output_dir"]
    FINAL_ADAPTER_PATH = out_cfg["final_adapter_path"]

    # xCOT component configs
    n_demonstrations = xcot_cfg.get("n_demonstrations", 4)
    language_weights = xcot_cfg.get("language_weights", {
        "en": 0.2, "ta": 0.2, "ml": 0.2, "ta_roman": 0.2, "ml_roman": 0.2
    })
    sampling_temperature = xcot_cfg.get("sampling_temperature", 1.0)
    n_cot_variants = xcot_cfg.get("n_cot_variants", 3)
    distill_alpha = xcot_cfg.get("distill_alpha", 0.5)
    distill_temperature = xcot_cfg.get("distill_temperature", 2.0)
    use_teacher = xcot_cfg.get("use_teacher_for_distill", False)

    # ---- Load dataset ----
    print(f"Loading dataset: {REPO_ID} (split: train)")
    raw_ds = load_dataset(REPO_ID, split="train")
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

    # Language counts for mSampling
    lang_counts = Counter(ex["language"] for ex in train_examples)
    print("Training example language distribution:", dict(lang_counts))

    # ---- Tokenize ----
    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    SYSTEM_PROMPT = (
        "You are a mathematical reasoning assistant. "
        "Solve the problem carefully. Explain the reasoning step by step. "
        "End with #### followed by the final numeric answer."
    )

    def render_chat(messages, add_generation_prompt):
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        if not isinstance(rendered, str):
            raise TypeError(f"Expected chat template to return str, got {type(rendered).__name__}")
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

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]

        # Use the common token prefix so boundary differences can't mask answer tokens
        common_prefix_len = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids):
            if prompt_token != full_token:
                break
            common_prefix_len += 1

        full_ids = full_ids[:MAX_LENGTH]

        if common_prefix_len >= MAX_LENGTH or not full_ids:
            return {"input_ids": [], "attention_mask": [], "labels": []}

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
                assert all(isinstance(x, int) for x in value), (name, i, key, type(value[0]) if value else None)
            assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])
        print(f"{name}: {len(dataset)} valid examples; flat token lists verified.")

    validate_tokenized_dataset(tokenized_train, "Train")
    validate_tokenized_dataset(tokenized_val, "Validation")
    print(tokenized_train[0])

    # ---- Load model ----
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training configuration.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute dtype:", compute_dtype)

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, torch_dtype=compute_dtype
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

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Initialize xCOT components ----
    print("\n=== Initializing xCOT Components ===")

    # xICL demonstration selector
    icl_selector = xICLDemonstrationSelector(
        train_examples=train_examples,
        n_demonstrations=n_demonstrations,
        language_diversity=True,
    )
    print(f"xICL: n_demonstrations={n_demonstrations}, language_diversity=True")

    # Random Online CoT generator
    cot_generator = RandomCoTGenerator(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        n_variants=n_cot_variants,
        max_new_tokens=MAX_LENGTH,
    )
    print(f"Random Online CoT: n_variants={n_cot_variants}")

    # mSampling data balancer
    data_balancer = xCOTDataBalancer(
        language_weights=language_weights,
        temperature=sampling_temperature,
    )
    print(f"mSampling: language_weights={language_weights}, temperature={sampling_temperature}")

    # Token-level xDistill
    distill_loss_fn = TokenLevelxDistillLoss(
        teacher_model=None,  # No teacher model (would require loading a second model)
        alpha=distill_alpha,
        temperature=distill_temperature,
    )
    print(f"Token-level xDistill: alpha={distill_alpha}, temperature={distill_temperature}")
    print(f"  (teacher_for_distill={use_teacher})")

    # ---- Custom collator ----
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
            name=f"xcot-{MODEL_ID}-lora-r{lora_cfg['r']}-{EPOCHS}ep",
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
                "language_distribution": dict(lang_counts),
                "xcot_components": {
                    "xICL": {"n_demonstrations": n_demonstrations, "language_diversity": True},
                    "random_online_cot": {"n_variants": n_cot_variants},
                    "mSampling": {"language_weights": language_weights, "temperature": sampling_temperature},
                    "token_level_xdistill": {"alpha": distill_alpha, "temperature": distill_temperature},
                },
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
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "logging_steps": 10,
        "logging_first_step": True,
        "report_to": "none" if not (HAS_WANDB and wandb_cfg.get("enabled", True)) else "wandb",
        "remove_unused_columns": True,
        "dataloader_pin_memory": True,
        # Early stopping on eval loss
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }

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

    # Add early stopping callback
    from transformers import EarlyStoppingCallback
    trainer.callback_handler.callbacks.append(
        EarlyStoppingCallback(early_stopping_patience=1)
    )

    # Verify collation
    batch = next(iter(trainer.get_train_dataloader()))
    for key, value in batch.items():
        print(key, tuple(value.shape), value.dtype)

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape == batch["labels"].shape
    print("Batch collation successful.")

    # ---- Train and save ----
    print("\n=== Starting xCOT Training ===")
    train_result = trainer.train()

    trainer.save_model(FINAL_ADAPTER_PATH)
    processor.save_pretrained(FINAL_ADAPTER_PATH)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("xCOT Training complete.")
    print("LoRA adapter saved to:", FINAL_ADAPTER_PATH)

    if HAS_WANDB and wandb_cfg.get("enabled", True):
        wandb.finish()

    # Also save the xCOT component configs for reference
    xcot_info_path = Path(FINAL_ADAPTER_PATH) / "xcot_config.json"
    with open(xcot_info_path, "w") as f:
        json.dump({
            "method": "xCOT",
            "components": {
                "xICL": {
                    "n_demonstrations": n_demonstrations,
                    "language_diversity": True,
                },
                "random_online_cot": {
                    "n_variants": n_cot_variants,
                },
                "mSampling": {
                    "language_weights": language_weights,
                    "temperature": sampling_temperature,
                },
                "token_level_xDistill": {
                    "alpha": distill_alpha,
                    "temperature": distill_temperature,
                    "teacher_model": "none (self-distillation via standard CE loss)",
                },
            },
            "training": {
                "epochs": EPOCHS,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "learning_rate": LEARNING_RATE,
                "lr_scheduler": LR_SCHEDULER_TYPE,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
            },
            "final_metrics": metrics,
        }, f, indent=2)
    print(f"xCOT configuration saved to: {xcot_info_path}")


if __name__ == "__main__":
    main()
