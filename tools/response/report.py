#!/usr/bin/env python3
"""③ 号线（共情回复）的诊断报告：把一行 BLEU/ROUGE 拆成能指导写作的线索。

`tools/eval_loop.py` 只给一个总分，但那个数**不告诉你下一步改什么**。
这个脚本按长度分档、按长度偏离度、按最差样本三个角度展开，回答三个问题：

1. 我在哪一档最亏？（按参考长度分档看 BLEU-4 / ROUGE-L）
2. 我的长度是不是偏离参考分布？（cand/ref 比值分布 + 简短惩罚）
3. 最差的那些条到底错在哪？（导出最差 N 条，含真值对照，供人工归类失败模式）

指标口径**严格对齐** `src/evaluate.py`（char 分词、corpus 级 BLEU-4 带简短惩罚、
ROUGE-L 逐条算 F 再平均），所以这里的数与台账可比。

用法
----
    python tools/response/report.py --predictions runs/eval-resp-v4/submission.jsonl
    python tools/response/report.py --tag resp-v4            # 等价，自动找 runs/eval-<tag>/
    python tools/response/report.py --tag resp-v4 --worst 30 --dump runs/badcase-v4.jsonl

只用标准库。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as stats
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "runs"

#: 与 src/evaluate.py:49 完全一致的 CJK 判定
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

#: 与 src/evaluate.py 的 report 一致的分档
LENGTH_BANDS = ((0, 50, "<50"), (50, 150, "50-150"), (150, 400, "150-400"), (400, 10**9, "400+"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: JSON 解析失败: {exc.msg}") from exc
    if not rows:
        raise SystemExit(f"文件为空: {path}")
    return rows


def char_tokens(text: str) -> list[str]:
    """字符级分词：中文按字、英文数字按词，标点与空白丢弃。

    **必须与 src/evaluate.py:55-78 的 char_tokenize 完全一致**——这是与台账对得上数的前提。
    最初这里偷懒直接切片，结果把标点和空白也算进 n-gram，BLEU-4 会比台账高约 40%
    （0.0148 vs 0.0107），报告就失去可比性。
    """
    tokens: list[str] = []
    buffer: list[str] = []
    for ch in text:
        if CJK.match(ch):
            if buffer:
                tokens.append("".join(buffer).lower())
                buffer.clear()
            tokens.append(ch)
        elif ch.isalnum():
            buffer.append(ch)
        else:
            if buffer:
                tokens.append("".join(buffer).lower())
                buffer.clear()
    if buffer:
        tokens.append("".join(buffer).lower())
    return tokens


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu4(candidates: list[list[str]], references: list[list[str]]) -> dict:
    """corpus 级 BLEU-4，口径同 src/evaluate.py:107-144。入参为已分词序列。"""
    clipped: Counter = Counter()
    total: Counter = Counter()
    cand_len = ref_len = 0
    for cand, ref in zip(candidates, references):
        for n in range(1, 5):
            got = ngrams(cand, n)
            want = ngrams(ref, n)
            for gram, count in got.items():
                clipped[n] += min(count, want.get(gram, 0))
            total[n] += max(0, len(cand) - n + 1)
        cand_len += len(cand)
        ref_len += len(ref)
    precisions = []
    for n in range(1, 5):
        precisions.append(clipped[n] / total[n] if total[n] and clipped[n] else 0.0)
    if cand_len == 0:
        bp = 0.0
    elif cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len)
    if any(p == 0.0 for p in precisions):
        score = 0.0
    else:
        score = math.exp(sum(math.log(p) for p in precisions) / 4) * bp
    return {"bleu4": score, "brevity_penalty": bp, "precisions": precisions,
            "candidate_length": cand_len, "reference_length": ref_len}


def lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            current[j] = previous[j - 1] + 1 if token_a == token_b else max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(candidates: list[list[str]], references: list[list[str]]) -> dict:
    """逐条算 ROUGE-L F 再平均，口径同 src/evaluate.py:169-196。入参为已分词序列。"""
    fs, ps, rs = [], [], []
    for cand, ref in zip(candidates, references):
        if not cand or not ref:
            fs.append(0.0); ps.append(0.0); rs.append(0.0)
            continue
        lcs = lcs_length(cand, ref)
        p = lcs / len(cand)
        r = lcs / len(ref)
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
        ps.append(p); rs.append(r)
    return {"f": stats.fmean(fs), "precision": stats.fmean(ps), "recall": stats.fmean(rs)}


def band_of(length: int) -> str:
    for low, high, name in LENGTH_BANDS:
        if low <= length < high:
            return name
    return LENGTH_BANDS[-1][2]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.predictions:
        pred = Path(args.predictions)
        run_dir = pred.parent
    elif args.tag:
        run_dir = RUNS / f"eval-{args.tag}"
        pred = run_dir / "submission.jsonl"
    else:
        raise SystemExit("必须给 --tag 或 --predictions")
    if not pred.is_file():
        raise SystemExit(f"找不到 {pred}")
    ref = Path(args.references) if args.references else run_dir / "val_references_subset.jsonl"
    if not ref.is_file():
        fallback = REPO_ROOT / "data" / "processed" / "val_references.jsonl"
        if fallback.is_file():
            ref = fallback
        else:
            raise SystemExit(f"找不到真值 {ref}，也没有 {fallback}")
    return pred, ref


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="③ 号线诊断报告")
    parser.add_argument("--tag", help="实验名，读 runs/eval-<tag>/")
    parser.add_argument("--predictions", help="直接给 submission.jsonl")
    parser.add_argument("--references", help="真值文件（默认同目录 val_references_subset.jsonl）")
    parser.add_argument("--worst", type=int, default=20, help="输出最差 N 条")
    parser.add_argument("--dump", help="把最差 N 条写到这个 jsonl")
    args = parser.parse_args(argv)

    pred_path, ref_path = resolve_paths(args)
    predictions = {row["id"]: row for row in read_jsonl(pred_path) if isinstance(row.get("id"), str)}
    references = {row["id"]: row.get("target", {}) for row in read_jsonl(ref_path) if isinstance(row.get("id"), str)}
    ids = [i for i in predictions if i in references]
    if not ids:
        raise SystemExit(f"{pred_path} 与 {ref_path} 没有一条 id 对得上")

    # 口径必须与 tools/eval_loop.py / src/evaluate.py 一致：char 分词（中文按字）。
    # 若这里用默认的整串比较，数会与台账对不上，报告就失去可比性。
    raw_cands = [(predictions[i].get("response_text") or "") for i in ids]
    raw_refs = [(references[i].get("response_text") or "") for i in ids]
    cands = [char_tokens(t) for t in raw_cands]
    refs = [char_tokens(t) for t in raw_refs]

    print(f"预测   : {pred_path}")
    print(f"真值   : {ref_path}")
    print(f"可比对 : {len(ids)} 条")
    print()

    overall = bleu4(cands, refs)
    rouge = rouge_l(cands, refs)
    print("== 总分（与台账同口径）==")
    print(f"  BLEU-4        {overall['bleu4']:.4f}")
    print(f"  简短惩罚      {overall['brevity_penalty']:.4f}   (候选 {overall['candidate_length']} 字 / 参考 {overall['reference_length']} 字)")
    print(f"  各级精确率    {[round(p, 4) for p in overall['precisions']]}")
    print(f"  ROUGE-L F     {rouge['f']:.4f}   (P {rouge['precision']:.4f} / R {rouge['recall']:.4f})")
    empty = sum(1 for c in raw_cands if not c.strip())
    print(f"  空回复        {empty}/{len(ids)}")
    print("  注：本脚本的数与 runs/eval-<tag>/score.json 同口径，可直接对照；"
          "若两者不一致，先确认 --references 与 eval_loop 用的是同一份真值。")
    print()

    print("== 按参考长度分档 ==")
    print(f"  {'档位':<10} {'n':>4}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'回复中位':>8}")
    for _, _, name in LENGTH_BANDS:
        idx = [k for k, i in enumerate(ids) if band_of(len(raw_refs[k])) == name]
        if not idx:
            continue
        sub_c = [cands[k] for k in idx]
        sub_r = [refs[k] for k in idx]
        b = bleu4(sub_c, sub_r)["bleu4"]
        f = rouge_l(sub_c, sub_r)["f"]
        med = stats.median([len(raw_cands[k]) for k in idx])
        print(f"  {name:<10} {len(idx):>4}  {b:>8.4f}  {f:>8.4f}  {med:>8.0f}")
    print()

    print("== 长度偏离（候选/参考）==")
    ratios = [len(c) / len(r) for c, r in zip(raw_cands, raw_refs) if r]
    if ratios:
        srt = sorted(ratios)
        print(f"  中位 {stats.median(srt):.2f}   P10 {srt[int(len(srt)*0.1)]:.2f}   P90 {srt[int(len(srt)*0.9)]:.2f}")
        buckets = {"<0.6 太短": 0, "0.6-1.0 偏短": 0, "1.0-1.6 合适": 0, ">1.6 太长": 0}
        for value in ratios:
            key = "<0.6 太短" if value < 0.6 else ("0.6-1.0 偏短" if value < 1.0 else ("1.0-1.6 合适" if value < 1.6 else ">1.6 太长"))
            buckets[key] += 1
        for key, count in buckets.items():
            print(f"  {key:<12} {count:>4} 条  ({count/len(ratios):.0%})")
    print()

    scored = []
    for k, i in enumerate(ids):
        f = rouge_l([cands[k]], [refs[k]])["f"]
        b = bleu4([cands[k]], [refs[k]])["bleu4"]
        scored.append((f, b, i, len(raw_cands[k]), len(raw_refs[k]), raw_cands[k]))
    scored.sort(key=lambda row: (row[0], row[1]))
    worst = scored[: args.worst]
    print(f"== 最差 {len(worst)} 条（按 ROUGE-L 升序）==")
    for f, b, i, lc, lr, c in worst[:10]:
        print(f"  {i}  ROUGE-L {f:.3f}  BLEU {b:.3f}  长度 {lc}/{lr}")
        print(f"      {c[:70].replace(chr(10), ' ')}")
    if args.dump:
        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as handle:
            for f, b, i, lc, lr, c in worst:
                handle.write(json.dumps({
                    "id": i, "rouge_l_f": round(f, 4), "bleu4": round(b, 4),
                    "candidate_len": lc, "reference_len": lr,
                    "response_text": c,
                    "reference_text": references[i].get("response_text", ""),
                }, ensure_ascii=False) + "\n")
        print(f"\n最差 {len(worst)} 条已写 -> {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())