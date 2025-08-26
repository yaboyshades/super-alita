"""LoRA fine-tuning scaffold using PEFT + Transformers.

Usage (example):
  python backend/fine_tuning/train_adapter.py --base-model meta-llama/Meta-Llama-3-8B-Instruct \
      --feedback-file feedback_dump.json --epochs 1

Expects feedback JSON list with objects containing at minimum:
  {"prompt": "...", "final_code": "..."}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("alita.finetune")

try:  # pragma: no cover - heavy deps optional
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    AVAILABLE = True
except Exception as e:  # noqa: BLE001
    log.warning("Fine-tune stack unavailable: %s", e)
    AVAILABLE = False


def load_feedback(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):  # type: ignore[unreachable]
        raise ValueError("Feedback file must contain a list of objects")
    return data  # type: ignore[return-value]


def format_samples(feedback: list[dict[str, Any]]) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for row in feedback:
        prompt = row.get("prompt") or "Refactor the following code"
        final_code = row.get("final_code") or row.get("code") or ""
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{final_code}".strip()
        samples.append({"text": text})
    return samples


def build_dataset(samples: list[dict[str, str]], tokenizer):  # type: ignore[no-untyped-def]
    ds = Dataset.from_list(samples)
    return ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=2048))


def train(args):  # type: ignore[no-untyped-def]
    if not AVAILABLE:
        raise SystemExit("Fine-tune dependencies not installed.")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    log.info("Loading base model %s", args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb,
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    feedback = load_feedback(args.feedback_file)
    samples = format_samples(feedback)
    ds = build_dataset(samples, tok)
    out_dir = args.output_dir or f"adapters/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        fp16=True,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds)
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf8") as fh:
        json.dump({
            "base_model": args.base_model,
            "samples": len(samples),
            "epochs": args.epochs,
        }, fh, indent=2)
    log.info("Adapter saved to %s", out_dir)


def parse_args():  # type: ignore[no-untyped-def]
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--feedback-file", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--output-dir")
    return p.parse_args()


if __name__ == "__main__":  # pragma: no cover
    a = parse_args()
    train(a)