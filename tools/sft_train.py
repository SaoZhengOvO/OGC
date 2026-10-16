#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OGC 赛题二 · 结构化多任务 LoRA SFT（AMD MI300X / ROCm）

在 MI300X 上对 Qwen2.5-3B 做 LoRA 微调，一次生成三个字段：
emotion_label / user_profile / response_text。

用法::

    python3 tools/sft_train.py --smoke            # 20 步冒烟，验证脚本可跑
    python3 tools/sft_train.py --epochs 2         # 正式训练，产物落在 runs/sft-lora-v1

依赖（服务器镜像自带）：torch 2.12+rocm / transformers / peft / datasets / accelerate。
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "GivenData" / "participant" / "model"
DATA_FILE = ROOT / "data" / "processed" / "train_samples.jsonl"
OUT_DIR = ROOT / "runs" / "sft-lora-v1"

SYSTEM = (
    "你是情感陪伴对话模型。读对话历史，只输出一个 JSON 对象，字段固定为 "
    "emotion_label（16 类之一）、user_profile（personality_traits/interests/style "
    "三个字符串数组，取值必须来自固定词表，没有就给空数组）、response_text"
    "（对用户最后一句的自然回复）。不要输出任何解释或多余文字。"
)


def read_rows(limit):
    rows = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def to_text(tok, row):
    hist = row.get("history") or []
    lines = []
    for h in hist:
        who = "用户" if h.get("role") == "user" else "助手"
        lines.append(who + "：" + (h.get("content") or ""))
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "对话历史：\n" + "\n".join(lines) + "\n\n请输出 JSON。",
        },
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tgt = row.get("target") or {}
    answer = json.dumps(
        {
            "emotion_label": tgt.get("emotion_label", "neutral"),
            "user_profile": tgt.get("user_profile")
            or {"personality_traits": [], "interests": [], "style": []},
            "response_text": tgt.get("response_text", ""),
        },
        ensure_ascii=False,
    )
    return prompt, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()

    print(
        "torch", torch.__version__,
        "hip", torch.version.hip,
        "cuda_avail", torch.cuda.is_available(),
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    limit = 200 if a.smoke else (a.limit or 0)
    rows = read_rows(limit)
    print("rows", len(rows), flush=True)

    prompts, answers = [], []
    for r in rows:
        p, ans = to_text(tok, r)
        prompts.append(p)
        answers.append(ans)

    PAD = tok.pad_token_id

    def enc(i):
        p = tok(prompts[i], add_special_tokens=False)["input_ids"]
        ans_ids = tok(answers[i] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + ans_ids)[: a.maxlen]
        labels = ([-100] * len(p) + ans_ids)[: a.maxlen]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
        }

    ds = Dataset.from_dict({"i": list(range(len(rows)))})
    ds = ds.map(
        lambda b: {
            "input_ids": [enc(i)["input_ids"] for i in b["i"]],
            "attention_mask": [enc(i)["attention_mask"] for i in b["i"]],
            "labels": [enc(i)["labels"] for i in b["i"]],
        },
        batched=True,
        batch_size=64,
        remove_columns=["i"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    def collate(feats):
        mx = max(len(f["input_ids"]) for f in feats)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in feats:
            pad = mx - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [PAD] * pad)
            out["attention_mask"].append(f["attention_mask"] + [0] * pad)
            out["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}

    args = TrainingArguments(
        output_dir=a.out,
        overwrite_output_dir=True,
        per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=a.epochs,
        max_steps=(20 if a.smoke else -1),
        logging_steps=5,
        save_steps=1000,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=42,
    )
    tr = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    t0 = time.time()
    tr.train()
    print("train_seconds", round(time.time() - t0, 1), flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("SAVED-ADAPTER", a.out, flush=True)


if __name__ == "__main__":
    main()
