#!/usr/bin/env python3
"""给已经跑完的 submission.jsonl 补算 BERTScore，不需要重跑模型。

## 为什么要单独一个脚本

`src/evaluate.py` 的 BERTScore 需要一个 `f(list[str]) -> list[list[vector]]` 的
**token 级**编码器。不提供时它会诚实返回 `{"computed": false, "reason": ...}`
而不是编一个分数——这是对的，但也意味着**所有已跑完的臂都没有第三项指标**。

官方「回应质量 20 分」公布的是 BLEU-4 / ROUGE-L / BERTScore 三项综合。少一项，
裁决就只能靠两项，而两项之间的权重组合同样未知。所以要把这一项补上。

好消息是 BERTScore **只依赖预测文本和参考文本**，与模型推理无关，所以可以在
GPU 空下来之后对着已有的 submission.jsonl 补算，不用重跑 300 条推理。

## 编码器选型

本赛题是中文对话。`bert-base-chinese` 是中文 BERTScore 的常规选择；英文的
`roberta-large` 在中文上会显著低估语义相似度。**编码器名字必须写进报告**，
因为换编码器会改变 BERTScore 的绝对值——它只在同一编码器下可比。

## 用法

    # 单臂
    python3 tools/bertscore_rescore.py \
        --submission runs/eval-clean-A/submission.jsonl \
        --references runs/slice300/val_references.jsonl \
        --encoder /mnt/workspace/models/bert-base-chinese

    # 一次算多臂，输出可直接喂给 tools/adjudicate_arms.py --bertscore
    python3 tools/bertscore_rescore.py --encoder /mnt/workspace/models/bert-base-chinese \
        --submission runs/eval-clean-A/submission.jsonl runs/eval-clean-B/submission.jsonl \
        --references runs/slice300/val_references.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.evaluate import _bertscore, load_references  # noqa: E402


def load_submission(path: Path) -> Dict[str, Dict[str, Any]]:
    """按 id 去重，保留最后一次出现的记录（并发写会把同一 id 写多行）。"""
    rows: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("id") or "")
            if not sid:
                continue
            if sid in rows:
                duplicates += 1
            rows[sid] = row
    if duplicates:
        print(f"  ⚠️  {path.name}: 丢弃 {duplicates} 行重复 id（保留最后一次）", file=sys.stderr)
    return rows


def make_encoder(model_dir: str, device: str, batch_size: int, max_length: int) -> Callable[[List[str]], List[Any]]:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()

    @torch.no_grad()
    def encode(texts: List[str]) -> List[Any]:
        vectors: List[Any] = []
        for start in range(0, len(texts), batch_size):
            batch = [t if t.strip() else "。" for t in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].bool()
            for i in range(len(batch)):
                keep = mask[i].clone()
                # BERTScore 的惯例是去掉特殊符号 [CLS]/[SEP]，否则它们会稳定贡献虚高相似度
                idx = keep.nonzero(as_tuple=False).flatten()
                if len(idx) > 2:
                    keep[idx[0]] = False
                    keep[idx[-1]] = False
                vec = hidden[i][keep].cpu().numpy().astype(np.float64)
                vectors.append(vec)
        return vectors

    return encode


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", required=True, help="本地编码器目录（离线环境必须是本地路径）")
    parser.add_argument("--references", required=True, help="val_references.jsonl")
    parser.add_argument("--submission", required=True, nargs="+", help="一个或多个 submission.jsonl")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", default="", help="把结果写进这个 json（默认只打印）")
    args = parser.parse_args(argv[1:])

    references = load_references(args.references)
    print(f"参考集 {len(references)} 条；编码器 {args.encoder}", file=sys.stderr)
    encoder = make_encoder(args.encoder, args.device, args.batch_size, args.max_length)

    results: Dict[str, Any] = {}
    for sub_path in args.submission:
        path = Path(sub_path)
        arm = path.parent.name
        rows = load_submission(path)
        cands: List[str] = []
        refs: List[str] = []
        for sid, row in rows.items():
            ref = references.get(sid)
            if not ref:
                continue
            gold = str(ref.get("response") or ref.get("response_text") or "").strip()
            pred = str(row.get("response") or row.get("response_text") or "").strip()
            if not gold:
                continue
            cands.append(pred)
            refs.append(gold)

        score = _bertscore(cands, refs, encoder)
        results[arm] = {"n": len(cands), "encoder": Path(args.encoder).name, **score}
        print(
            f"{arm:<22} n={len(cands):>5}  "
            f"P={score.get('precision', 0):.5f}  R={score.get('recall', 0):.5f}  F1={score.get('f1', 0):.5f}",
            file=sys.stderr,
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.out}", file=sys.stderr)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
