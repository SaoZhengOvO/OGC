"""本地交互式对话：用**交付配置**直接跟数字人聊天。

为什么要这个工具：提交上去的东西是 `run_inference.py` 读 jsonl 批量跑，没有交互入口。
要判断「它像不像个陪伴型数字人」，只能真聊几句 —— 看回复长度、口吻、情绪判断、画像
画像抽得对不对。如果这里为了好玩另写一套提示词或解码参数，看到的就不是要交的东西，
所以本工具**原样复用**交付链路上的三个组件：

- `deploy/participant/run_inference.py` 的 `SYSTEM_PROMPT` / `conversation_text` /
  `parse_prediction` / `needs_schema_retry`
- `deploy/participant/wiring.py` 的 `compose_system_prompt` / `build_messages`（含 RAG 检索）
- `deploy/participant/configs/inference_config.json` 的 `generation` 参数

用法：

    .venv\\Scripts\\python.exe tools/chat.py
    .venv\\Scripts\\python.exe tools/chat.py --model GivenData/participant/model
    .venv\\Scripts\\python.exe tools/chat.py --no-rag          # 关掉检索，对比用
    .venv\\Scripts\\python.exe tools/chat.py --max-history 6   # 只带最近 6 轮

对话里可用的命令：`/reset` 重开一段、`/config` 看当前生效的配置、`/quit` 退出。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTICIPANT_DIR = REPO_ROOT / "deploy" / "participant"

# 与 deploy/participant/run_inference.py:562-566 逐字一致。
# 之所以要复制：那两个常量定义在 run_inference.main() **函数内部**，模块级拿不到。
# 会不会漂？tests/test_chat_prompts.py 会比对两边的字面量，改了这边不改那边会红。
SCHEMA_REMINDER = (
    "你上一条输出不符合要求：缺少 response_text 或 emotion_label。"
    "请只输出一个 JSON 对象，必须同时包含 response_text 与 emotion_label 两个字段，"
    "不要输出 json 之外的任何文字，不要输出 ``` 代码围栏。"
)


def _load(name: str, path: Path) -> types.ModuleType:
    """按路径加载模块（`run_inference` 这名字太通用，不当顶层 import 用）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - 路径写死，几乎不可能
        raise RuntimeError(f"加载不了 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_generation_config(model, generation: dict, GenerationConfig):
    """把配置里的 generation 字段落到 GenerationConfig 上。

    不能直接 `model.generate(**generation)`：transformers 5.x 把 `min_new_tokens`
    从 generate() 签名里删了，未知 kwargs 会被**静默丢弃**。走 GenerationConfig 才传得进去。
    """
    gen_cfg = GenerationConfig.from_model_config(model.config)
    valid = set(gen_cfg.to_dict().keys())
    unknown = sorted(set(generation) - valid)
    for key, value in generation.items():
        if key in valid:
            setattr(gen_cfg, key, value)
    if unknown:
        print(f"[warn] generation 里这些键不被 GenerationConfig 认识，已忽略：{unknown}")
    # pad_token 与 eos 对齐，否则 batch>1 时 transformers 会告警并随机选一个
    if gen_cfg.pad_token_id is None:
        gen_cfg.pad_token_id = gen_cfg.eos_token_id
    return gen_cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用交付配置跟数字人对话")
    parser.add_argument("--model", default=str(REPO_ROOT / "GivenData" / "participant" / "model"),
                        help="模型目录（默认官方给的那份）")
    parser.add_argument("--max-history", type=int, default=0, help="只带最近 N 段对话，0 表示全带")
    parser.add_argument("--no-rag", action="store_true", help="不走向量/BM25 检索，只用官方组包")
    parser.add_argument("--profile-mode", default=None,
                        help="覆盖 PARTICIPANT_PROFILE_MODE（off/floor/force），看画像策略的实际效果")
    args = parser.parse_args(argv)

    if args.profile_mode:
        import os
        os.environ["PARTICIPANT_PROFILE_MODE"] = args.profile_mode

    run_inference = _load("ogc_run_inference", PARTICIPANT_DIR / "run_inference.py")
    sys.path.insert(0, str(PARTICIPANT_DIR))
    wiring = _load("ogc_wiring", PARTICIPANT_DIR / "wiring.py")

    config = run_inference.load_config()
    generation = dict(config.get("generation") or {})
    model_dir = Path(args.model).resolve()
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"{model_dir} 下没有 config.json，不是模型目录")

    print(f"模型      {model_dir}")
    print(f"生成参数  {json.dumps(generation, ensure_ascii=False)}")
    print(f"后端      {config.get('backend')}   设备 device_map={config.get('device_map','auto')}")
    print(f"提示词    {'关掉检索（只走官方组包）' if args.no_rag else 'wiring.build_messages（含 RAG）'}")
    print(f"画像模式  {run_inference.profile_mode()}")
    print("正在加载模型……", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        device_map=config.get("device_map", "auto"),
        torch_dtype=config.get("torch_dtype", "auto"),
    )
    model.eval()
    gen_cfg = _build_generation_config(model, generation, GenerationConfig)
    device = next(model.parameters()).device
    print(f"加载完成 {time.perf_counter() - started:.1f}s，权重在 {device}\n")

    official_prompt = run_inference.SYSTEM_PROMPT
    system_prompt = official_prompt if args.no_rag else wiring.compose_system_prompt(official_prompt)

    history: list[dict] = []
    print("开始聊。命令：/reset 重开、/config 看配置、/quit 退出（Ctrl+C 也行）\n")

    while True:
        try:
            text = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in {"/quit", "/exit", "/q"}:
            break
        if text == "/reset":
            history.clear()
            print("[已重开一段对话]")
            continue
        if text == "/config":
            print(f"  模型        {model_dir.name}")
            print(f"  生成参数    {json.dumps(generation, ensure_ascii=False)}")
            print(f"  画像模式    {run_inference.profile_mode()}")
            print(f"  检索        {'off' if args.no_rag else 'on'}")
            print(f"  已聊轮数    {sum(1 for turn in history if turn['role'] == 'user')}")
            print(f"  系统提示词  {len(system_prompt)} 字")
            continue

        history.append({"role": "user", "content": text})
        sample = {"id": "local-chat", "history": history}
        if args.max_history > 0 and len(history) > args.max_history:
            sample = {"id": "local-chat", "history": history[-args.max_history:]}

        if args.no_rag:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": run_inference.conversation_text(sample)},
            ]
        else:
            messages = wiring.build_messages(sample, system_prompt, run_inference.conversation_text)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        began = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, generation_config=gen_cfg)
        elapsed = time.perf_counter() - began
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

        prediction, parse_failed = run_inference.parse_prediction("local-chat", decoded)
        retried = False
        if run_inference.needs_schema_retry(prediction, parse_failed):
            # 与交付链路一致：输出塌陷时追加一条硬约束重试一次
            retried = True
            retry_messages = messages + [{"role": "user", "content": SCHEMA_REMINDER}]
            prompt = tokenizer.apply_chat_template(retry_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(**inputs, generation_config=gen_cfg)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            prediction, parse_failed = run_inference.parse_prediction("local-chat", decoded)

        reply = prediction.get("response_text") or "（模型没给出回复文本）"
        history.append({"role": "assistant", "content": reply})

        print(f"\n数字人 > {reply}\n")
        profile = prediction.get("user_profile") or {}
        print(f"  [情绪] {prediction.get('emotion_label') or '（空）'}")
        print(f"  [画像] 人格 {profile.get('personality_traits') or []} / "
              f"兴趣 {profile.get('interests') or []} / 风格 {profile.get('style') or []}")
        print(f"  [耗时] {elapsed * 1000:.0f} ms，输出 {int(generated_ids.shape[-1])} token"
              f"{'，解析失败后重试过' if retried else ''}\n")

    print("再见。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
