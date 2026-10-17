#!/usr/bin/env python3
"""诊断空回复：复现指定样本的原始生成，判断"空 response_text"到底来自哪一步。

背景：全量 v8（n=2354）有 26 条 response_text 为空（1.10%）。
已知它们**不在** parse_failure_ids 里——即 JSON 解析是成功的，
所以不是"解析失败兜底把字段清空"。需要看原始输出才能定位。

用法::

    python tools/response/diagnose_empty.py \
        --predictions runs/eval-mi300x-v8-full/submission.jsonl \
        --out runs/empty-diag.jsonl

    # 只跑指定 id（逗号分隔）
    python tools/response/diagnose_empty.py --ids val_000003_t5,val_000085_t7

输出：每条一行 JSON，含 id / prompt_tokens / gen_tokens / raw_len / parse_ok /
response_text_len / raw_head / raw_tail / system_len / user_len。

只用标准库 + torch/transformers（项目既有依赖）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="复现空回复样本的原始生成")
    parser.add_argument("--predictions", default="runs/eval-mi300x-v8-full/submission.jsonl")
    parser.add_argument("--samples", default="data/processed/val_samples.jsonl")
    parser.add_argument("--model-path", default="GivenData/participant/model")
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--ids", default=None, help="逗号分隔；不给则自动取空回复的 id")
    parser.add_argument("--out", default="runs/empty-diag.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    import os
    os.environ.setdefault("PARTICIPANT_PROMPT_DIR", str(REPO_ROOT / args.prompt_dir))

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    sys.path.insert(0, str(REPO_ROOT / "deploy" / "participant"))
    ri = load_module("ri", REPO_ROOT / "GivenData" / "participant" / "run_inference.py")
    wiring = load_module("wiring", REPO_ROOT / "deploy" / "participant" / "wiring.py")

    predictions = {r["id"]: r for r in read_jsonl(REPO_ROOT / args.predictions)}
    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:
        ids = [i for i, r in predictions.items() if not (r.get("response_text") or "").strip()]
    if args.limit:
        ids = ids[: args.limit]
    samples = {r["id"]: r for r in read_jsonl(REPO_ROOT / args.samples)}
    ids = [i for i in ids if i in samples]
    print(f"待诊断 {len(ids)} 条", flush=True)

    model_dir = REPO_ROOT / args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, device_map="auto", torch_dtype="auto"
    )
    model.eval()

    config = ri.load_config()
    generation = dict(config["generation"])
    official_prompt = ri.SYSTEM_PROMPT.format(
        emotions=", ".join(ri.EMOTIONS),
        personality=", ".join(sorted(ri.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(ri.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(ri.PROFILE_LABELS["style"])),
    )
    wiring.log_wiring()
    system_prompt = wiring.compose_system_prompt(official_prompt)
    print(f"system prompt 长度 = {len(system_prompt)} 字符", flush=True)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for index, sample_id in enumerate(ids, 1):
            sample = samples[sample_id]
            messages = wiring.build_messages(sample, system_prompt, ri.conversation_text)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_len = int(inputs["input_ids"].shape[1])
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation)
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            try:
                obj = ri.extract_object(raw)
                parse_ok = True
            except Exception:
                obj = {}
                parse_ok = False
            response = obj.get("response_text") if isinstance(obj, dict) else None
            record = {
                "id": sample_id,
                "prompt_tokens": input_len,
                "gen_tokens": int(len(generated)),
                "raw_len": len(raw),
                "raw_head": raw[:160],
                "raw_tail": raw[-80:],
                "parse_ok": parse_ok,
                "keys": sorted(obj.keys()) if isinstance(obj, dict) else [],
                "response_text_len": len(response) if isinstance(response, str) else None,
                "original_response_text_len": len(predictions.get(sample_id, {}).get("response_text") or ""),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{index}/{len(ids)}] {sample_id} prompt={input_len} gen={record['gen_tokens']} "
                  f"raw={record['raw_len']} parse_ok={parse_ok} resp_len={record['response_text_len']}", flush=True)
    print(f"写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
