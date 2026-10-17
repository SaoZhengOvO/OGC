"""构建检索语料：把 train 的 (用户轮, 助手回复) 对压成一行一条的 JSONL。

用法（仓库根目录）：

    python -m rag.corpus.build_corpus
    python -m rag.corpus.build_corpus --limit 2000      # 冒烟用

产出：``rag/corpus/corpus.jsonl``（被 .gitignore 挡在版本库外，可由本脚本随时重建）。

为什么要自己 build 而不是直接读 ``data/processed/train_samples.jsonl``：

* ``data/processed/`` 是数据适配器的派生产物，也在 .gitignore 里；
* 唯一权威来源是 ``GivenData/训练-验证-数据集/train/train_public.jsonl``。

**语料只取 train，绝不碰 val**。val 泄漏会让整套验证作废，这是硬约束。

规模（2026-10-17 实测）：

    样本 18,651 条 / 对话 8,328 段
    query 中位 78 字（总 1,597,864）
    reply 中位 129 字（总 4,073,746）
    正文合计 5,671,610 字符 ≈ 16 MB，落盘 JSONL ≈ 21 MB

不索引整段对话：单段对话最长 40,262 字符，18,651 段全塞进去会撑爆内存，
而且长文本的 BM25 分数会被长度归一化冲淡。只索引 query（用户这一句），
reply 只作为返回内容，不参与倒排。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_TRAIN = os.path.join(
    REPO_ROOT, "GivenData", "训练-验证-数据集", "train", "train_public.jsonl"
)
OUT_PATH = os.path.join(REPO_ROOT, "rag", "corpus", "corpus.jsonl")


def _iter_target_pairs(limit: int | None = None):
    """从原始 train 里逐条吐出 (sample_id, conversation_id, query, reply, emotion)。"""
    if not os.path.exists(RAW_TRAIN):
        raise SystemExit(f"找不到原始训练数据：{RAW_TRAIN}")

    emitted = 0
    with open(RAW_TRAIN, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conversation_id = str(record.get("conversation_id") or "")
            if not conversation_id.startswith("train_"):
                # 防呆：一旦有人把 val 混进来，这里直接炸，不允许静默通过。
                raise SystemExit(
                    f"语料只允许 train_ 开头的对话，遇到 {conversation_id!r}"
                )
            turns = record.get("turns") or []
            by_turn = {
                turn.get("turn_id"): turn
                for turn in turns
                if isinstance(turn, dict)
            }
            for point in record.get("prediction_points") or []:
                target_turn_id = point.get("target_user_turn_id")
                target = point.get("target") or {}
                query = ""
                turn = by_turn.get(target_turn_id)
                if isinstance(turn, dict) and turn.get("role") == "user":
                    query = str(turn.get("content") or "")
                reply = str(target.get("response_text") or "")
                if not query.strip() or not reply.strip():
                    # 空 query 或空 reply 会被 fusion 静默丢掉，这里先剔除。
                    continue
                emotion = str(target.get("emotion_label") or "")
                # 与 data/processed 的样本 id 规则一致：train_conv_000001_t9
                sample_id = f"{conversation_id}_t{int(target_turn_id or 0)}"
                yield sample_id, conversation_id, query, reply, emotion
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 RAG 检索语料")
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条（冒烟用）")
    parser.add_argument("--out", default=OUT_PATH, help="输出路径")
    args = parser.parse_args(argv)

    started = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = 0
    query_chars = 0
    reply_chars = 0
    conversations: set[str] = set()

    temp_path = args.out + ".part"
    with open(temp_path, "w", encoding="utf-8") as handle:
        for sample_id, conversation_id, query, reply, emotion in _iter_target_pairs(
            limit=args.limit
        ):
            handle.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "conv": conversation_id,
                        "q": query,
                        "r": reply,
                        "emo": emotion,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            query_chars += len(query)
            reply_chars += len(reply)
            conversations.add(conversation_id)

    os.replace(temp_path, args.out)

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    elapsed = time.time() - started
    print(f"语料已写入：{args.out}")
    print(f"  样本 {written:,} 条 / 对话 {len(conversations):,} 段")
    print(f"  query {query_chars:,} 字符 / reply {reply_chars:,} 字符")
    print(f"  正文合计 {query_chars + reply_chars:,} 字符，落盘 {size_mb:.1f} MB")
    print(f"  耗时 {elapsed:.1f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
