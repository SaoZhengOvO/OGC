#!/usr/bin/env python3
"""用 LoRA 微调基座模型，让它直接吐出评测要的那一个 JSON object。

设计取舍，三条：

1. **不自己抄提示词。** 训练样本由 `tools/make_sft_data.py` 生成，那份脚本
   直接加载 `deploy/participant/run_inference.py` 的官方提示词。训练与推理
   必须是同一个任务，否则微调白做。
2. **只用 assistant 那一段算损失。** 走 chat 模板的 `return_assistant_tokens_mask`，
   把 system 与 user 的 token 全部置成 -100。否则模型会被训练去复读提示词。
3. **超长样本从头截断，不截尾。** 尾部是最近的对话与我们要的答案；截尾会把
   答案切掉，那种样本只会教坏模型。长度分布见 `docs/reports/training-plan.md`。

不依赖 `Trainer` / `trl` / `datasets`：transformers 5.x 的 Trainer 接口还在变，
而这里需要的只是一个标准训练循环，手写反而更短、更可审计。

用法：
    python tools/train_sft.py --max-steps 20                  # 冒烟
    python tools/train_sft.py --epochs 2 --batch-size 2       # 正式
    python tools/train_sft.py --model <另一个基座> --out runs/sft-7b
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "GivenData" / "participant" / "model"
DEFAULT_TRAIN = REPO_ROOT / "data" / "processed" / "sft_train.jsonl"
DEFAULT_DEV = REPO_ROOT / "data" / "processed" / "sft_dev.jsonl"
DEFAULT_OUT = REPO_ROOT / "runs" / "sft"

# Qwen2 全注意力 + MLP 的线性层名。换基座时用 --lora-targets 覆盖。
QWEN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def describe_device() -> str:
    """把这次训练实际会用到的卡报出来——方案要按卡定，第一步就得知道卡是什么。"""
    if not torch.cuda.is_available():
        return "CPU（会非常慢；建议加 --device cpu 只做冒烟）"
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return (
        f"{props.name} | 显存 {props.total_memory / 1024**3:.1f} GiB | "
        f"算力 sm_{capability[0]}{capability[1]} | 卡数 {torch.cuda.device_count()}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 JSONL。"""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def template_ids(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]], *, generation: bool) -> list[int]:
    """渲染 chat 模板并取出 input_ids。

    transformers 5.x 的 `apply_chat_template(tokenize=True)` 返回的是
    BatchEncoding 而不是裸列表，直接 `list()` 会得到键名。这里统一成 id 列表。
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)
    if isinstance(rendered, dict) or hasattr(rendered, "keys"):
        return [int(token) for token in rendered["input_ids"]]  # type: ignore[index]
    return [int(token) for token in rendered]


def encode_sample(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """把一条 chat 记录编码成 (input_ids, assistant_mask)。

    不用 `return_assistant_tokens_mask`：Qwen2.5 自带的 chat 模板里没有
    `{% generation %}` 标记，那条路只会打印一句警告然后给出空掩码，实测会把
    全部 18249 条样本静默跳过。改成自己定位边界——把最后一条 assistant 消息
    之前的对话用 `add_generation_prompt=True` 渲染一次，它的长度就是分界点。

    超长时**从头截断**：尾部是最近的对话和我们要它学的那个答案。
    """
    prefix_ids = template_ids(tokenizer, messages[:-1], generation=True)
    full_ids = template_ids(tokenizer, messages, generation=False)
    boundary = len(prefix_ids)
    if full_ids[:boundary] != prefix_ids:
        raise ValueError("chat 模板不是简单拼接，无法定位 assistant 起点；换 --model 前先确认这一点")
    answer_length = len(full_ids) - boundary

    if len(full_ids) > max_length:
        keep = max_length if answer_length >= max_length else max(max_length, answer_length)
        cut = len(full_ids) - keep
        full_ids = full_ids[cut:]
        boundary = max(0, boundary - cut)
    mask = [0] * boundary + [1] * (len(full_ids) - boundary)
    if not any(mask):
        raise ValueError("这条样本一个 assistant token 都没有")
    return full_ids, mask


def collate(
    batch: list[tuple[list[int], list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """右侧补齐一批。非 assistant 位置在 labels 里是 -100，不计损失。"""
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (ids, mask) in enumerate(batch):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(
            [token if keep else -100 for token, keep in zip(ids, mask)], dtype=torch.long
        )
        attention[row, :length] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def make_batches(
    encoded: list[tuple[list[int], list[int]]],
    batch_size: int,
    rng: random.Random,
    *,
    shuffle: bool,
) -> Iterable[list[tuple[list[int], list[int]]]]:
    """按长度接近分组再打乱组序：同一批的补齐浪费最小，显存也稳。"""
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index][0]))
    buckets = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(buckets)
        for bucket in buckets:
            rng.shuffle(bucket)
    for bucket in buckets:
        yield [encoded[index] for index in bucket if encoded[index][0]]


def build_lora(model: PreTrainedModel, args: argparse.Namespace) -> PreTrainedModel:
    """挂 LoRA。"""
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(args.lora_targets),
    )
    wrapped = get_peft_model(model, config)
    trainable = sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in wrapped.parameters())
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} 目标层={len(args.lora_targets)} 类")
    print(f"可训练参数 {trainable:,} / {total:,}（{trainable / total * 100:.2f}%）")
    return wrapped


def evaluate(
    model: PreTrainedModel,
    batches: list[list[tuple[list[int], list[int]]]],
    pad_token_id: int,
    device: torch.device,
) -> float:
    """在验证集上算平均损失。"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in batches:
            tensors = {key: value.to(device) for key, value in collate(batch, pad_token_id).items()}
            total_loss += float(model(**tensors).loss)
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。所有默认值都按「一张 12 GiB 的卡也能跑」来定。"""
    parser = argparse.ArgumentParser(description="LoRA 微调数字人陪伴模型")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="基座模型目录")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="适配器与日志的落盘目录")
    parser.add_argument("--max-length", type=int, default=2560, help="截断长度；只截头不截尾")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+", default=list(QWEN_TARGETS))
    parser.add_argument("--max-steps", type=int, default=0, help="只跑这么多优化步（冒烟用；0 表示按 epochs）")
    parser.add_argument("--dev-limit", type=int, default=64, help="验证最多用多少批")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="从某个适配器目录接着训")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"设备      {describe_device() if device.type == 'cuda' else 'CPU'}")
    print(f"基座      {args.model}")
    print(f"输出      {args.out}")

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"权重精度  {str(dtype).replace('torch.', '')}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载权重…")
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=dtype)
    print(f"        {time.time() - started:.1f} 秒")
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = build_lora(model, args)
    if args.resume is not None:
        from peft import PeftModel  # noqa: PLC0415 - 只有续训才需要

        model.load_adapter(str(args.resume), adapter_name="default", is_trainable=True)
        print(f"续训自  {args.resume}")
    model.to(device)
    model.train()

    print("编码数据…")
    started = time.time()
    train_rows = read_jsonl(args.train)
    encoded_train: list[tuple[list[int], list[int]]] = []
    skipped = 0
    for row in train_rows:
        try:
            encoded_train.append(encode_sample(tokenizer, row["messages"], args.max_length))
        except (ValueError, KeyError):
            skipped += 1
    print(f"        训练 {len(encoded_train)} 条（跳过 {skipped} 条），{time.time() - started:.1f} 秒")
    if not encoded_train:
        raise SystemExit("一条训练样本都没编码成功——先看上面跳过的原因，别在空数据上转圈")

    encoded_dev: list[tuple[list[int], list[int]]] = []
    if args.dev.is_file():
        for row in read_jsonl(args.dev):
            try:
                encoded_dev.append(encode_sample(tokenizer, row["messages"], args.max_length))
            except (ValueError, KeyError):
                continue
        print(f"        验证 {len(encoded_dev)} 条")

    steps_per_epoch = max(1, math.ceil(len(encoded_train) / args.batch_size / args.grad_accum))
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"计划      {total_steps} 步（每轮 {steps_per_epoch} 步 × {args.epochs} 轮），预热 {warmup_steps} 步")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda")

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    dev_batches = list(make_batches(encoded_dev, args.batch_size, rng, shuffle=False))[: args.dev_limit]
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train.log.jsonl"
    log_handle = log_path.open("a", encoding="utf-8", newline="\n")

    def log(payload: dict[str, Any]) -> None:
        payload["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        log_handle.flush()

    log({"event": "start", "device": describe_device(), "args": vars(args) | {
        "model": str(args.model), "train": str(args.train), "dev": str(args.dev), "out": str(args.out), "resume": str(args.resume),
    }, "train_rows": len(encoded_train), "dev_rows": len(encoded_dev), "total_steps": total_steps})

    step = 0
    micro = 0
    running_loss = 0.0
    running_tokens = 0
    started = time.time()
    stop = False
    while not stop:
        for batch in make_batches(encoded_train, args.batch_size, rng, shuffle=True):
            tensors = {key: value.to(device) for key, value in collate(batch, tokenizer.pad_token_id).items()}
            loss = model(**tensors).loss / args.grad_accum
            loss.backward()
            running_loss += float(loss.detach()) * args.grad_accum
            running_tokens += int(tensors["attention_mask"].sum())
            micro += 1
            if micro % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            lr = lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - started
                average = running_loss / args.log_every if step > 1 else running_loss
                print(
                    f"步 {step:6d}/{total_steps}  损失 {average:.4f}  lr {lr:.2e}  "
                    f"{running_tokens / max(elapsed, 1e-6):.0f} tok/s  已用 {elapsed / 60:.1f} 分"
                )
                log({"event": "step", "step": step, "loss": round(average, 4), "lr": lr,
                     "tok_per_s": round(running_tokens / max(elapsed, 1e-6)), "elapsed_s": round(elapsed, 1)})
                running_loss = 0.0

            if dev_batches and args.eval_every and step % args.eval_every == 0:
                dev_loss = evaluate(model, dev_batches, tokenizer.pad_token_id, device)
                print(f"           验证损失 {dev_loss:.4f}")
                log({"event": "eval", "step": step, "dev_loss": round(dev_loss, 4)})

            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(args.out / f"step-{step}")
                print(f"           已存 {args.out / f'step-{step}'}")
                log({"event": "save", "step": step, "path": str(args.out / f"step-{step}")})

            if step >= total_steps:
                stop = True
                break
        if args.max_steps:
            break

    if dev_batches:
        dev_loss = evaluate(model, dev_batches, tokenizer.pad_token_id, device)
        print(f"最终验证损失 {dev_loss:.4f}")
        log({"event": "eval", "step": step, "dev_loss": round(dev_loss, 4), "final": True})
    model.save_pretrained(args.out / "final")
    print(f"适配器已存到 {args.out / 'final'}")
    log({"event": "done", "step": step, "elapsed_s": round(time.time() - started, 1), "path": str(args.out / "final")})
    log_handle.close()
    print(f"训练日志 {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
