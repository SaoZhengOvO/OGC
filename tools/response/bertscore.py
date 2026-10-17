#!/usr/bin/env python3
"""离线 BERTScore：用本地已有的 Qwen2.5-3B 当编码器，把「回应质量 20 分」里
唯一没测过的那一项补上。

为什么要这个脚本
----------------
`src/evaluate.py` 的 `evaluate_predictions(..., bertscore_encoder=...)` 已经留好了接口：
给一个 `f(list[str]) -> list[list[vector]]`（**逐 token** 向量），它就能算 BERTScore
（贪心匹配 + 余弦 + F1，见 `_bertscore`）。

但仓库里**没有任何预置的编码器权重**（`GivenData/`、`deploy/` 都没有），
标准 BERTScore 用的 roberta-large 也不在离线镜像里。所以这里用**已有的基座模型**
Qwen2.5-3B 的最后一层隐状态做 token 向量，得到的是一个**可复现的代理值**，
不是官方那个 roberta-large 的 BERTScore。报告里必须这样标注，不能冒充官方口径。

两个模式
--------
* `--verify`：在小样本（默认 20 条）上同时跑
  (a) `src/evaluate.py` 的 `_bertscore`（参考实现，慢，Python 双循环）
  (b) 本脚本的向量化实现
  打印两者差值，证明向量化实现与仓库接口等价（阈值默认 1e-6）。
* 默认：全量/切片跑向量化实现，输出 bertscore.precision/recall/f1。

用法::

    python tools/response/bertscore.py --tag slice300-v8 --limit 300
    python tools/response/bertscore.py --tag slice300-v8 --verify --limit 20
    python tools/response/bertscore.py --predictions runs/eval-mi300x-v8-full/submission.jsonl \
        --references data/processed/val_references.jsonl --out runs/bertscore-full.json

依赖：numpy + torch + transformers（都是项目已有依赖，不引入新依赖）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_run(args) -> tuple[Path, Path]:
    if args.predictions:
        pred = Path(args.predictions)
        default_ref = pred.parent / "val_references_subset.jsonl"
        ref = Path(args.references) if args.references else default_ref
    elif args.tag:
        run = REPO_ROOT / "runs" / f"eval-{args.tag}"
        pred = run / "submission.jsonl"
        ref = Path(args.references) if args.references else run / "val_references_subset.jsonl"
    else:
        raise SystemExit("必须给 --tag 或 --predictions")
    if not pred.is_file():
        raise SystemExit(f"找不到预测文件 {pred}")
    if not ref.is_file():
        fallback = REPO_ROOT / "data" / "processed" / "val_references.jsonl"
        if fallback.is_file():
            ref = fallback
        else:
            raise SystemExit(f"找不到真值 {ref}")
    return pred, ref


class QwenTokenEncoder:
    """把文本编码成逐 token 向量（最后一层隐状态，L2 归一化）。"""

    def __init__(self, model_dir: Path, batch_size: int = 16, max_length: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, device_map="auto", torch_dtype="auto"
        )
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = next(self.model.parameters()).device

    def __call__(self, texts: list[str]) -> list[list]:
        import numpy as np

        vectors: list[list] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            encoded = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded, output_hidden_states=True).hidden_states[-1]
            hidden = hidden.float().cpu().numpy()
            mask = encoded["attention_mask"].cpu().numpy()
            for index in range(len(chunk)):
                keep = mask[index].astype(bool)
                tokens = hidden[index][keep]
                norms = np.linalg.norm(tokens, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors.append(list(tokens / norms))
        return vectors


def bertscore_vectorized(cand_vectors: list[list], ref_vectors: list[list]) -> dict:
    """与 src/evaluate.py::_bertscore 等价的向量化实现（贪心匹配 F1）。"""
    import numpy as np

    precisions, recalls = [], []
    for cand_vec, ref_vec in zip(cand_vectors, ref_vectors):
        c = np.asarray(cand_vec, dtype="float64")
        r = np.asarray(ref_vec, dtype="float64")
        if c.size == 0 or r.size == 0:
            precisions.append(0.0); recalls.append(0.0); continue
        sim = c @ r.T  # (len(c), len(r))，向量已归一化，点积即余弦
        precisions.append(float(sim.max(axis=1).mean()))
        recalls.append(float(sim.max(axis=0).mean()))
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"computed": True, "precision": p, "recall": r, "f1": f, "n": len(precisions)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线 BERTScore（Qwen2.5-3B token 向量）")
    parser.add_argument("--tag")
    parser.add_argument("--predictions")
    parser.add_argument("--references")
    parser.add_argument("--model-path", default="GivenData/participant/model")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--verify", action="store_true", help="与 src/evaluate.py 的 _bertscore 对齐验证")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    pred_path, ref_path = resolve_run(args)
    predictions = {r["id"]: r for r in read_jsonl(pred_path)}
    references = {r["id"]: r.get("target", {}) for r in read_jsonl(ref_path)}
    ids = [i for i in predictions if i in references]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit("预测与真值没有对齐的 id")

    cands = [(predictions[i].get("response_text") or "") for i in ids]
    refs = [(references[i].get("response_text") or "") for i in ids]
    print(f"预测 {pred_path}\n真值 {ref_path}\n可比对 {len(ids)} 条", flush=True)

    encoder = QwenTokenEncoder(REPO_ROOT / args.model_path, args.batch_size, args.max_length)
    print("编码候选中…", flush=True)
    cand_vecs = encoder(cands)
    print("编码参考中…", flush=True)
    ref_vecs = encoder(refs)
    result = bertscore_vectorized(cand_vecs, ref_vecs)
    print(f"向量化实现  P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f} (n={result['n']})")

    if args.verify:
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from evaluate import _bertscore  # noqa: PLC0415  # src/ 内部用顶层 import

        reference = _bertscore(cands, refs, encoder)
        print(f"src.evaluate    P={reference['precision']:.4f} R={reference['recall']:.4f} F1={reference['f1']:.4f}")
        for key in ("precision", "recall", "f1"):
            delta = abs(reference[key] - result[key])
            flag = "OK" if delta <= 1e-6 else "差异!"
            print(f"  {key:<9} Δ={delta:.3e} {flag}")

    if args.out:
        out_path = REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "predictions": str(pred_path),
            "references": str(ref_path),
            "n": len(ids),
            "encoder": f"Qwen2.5-3B last-hidden token vectors (max_length={args.max_length})",
            "note": "代理指标：非官方 roberta-large BERTScore",
            "bertscore": result,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())