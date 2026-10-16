#!/usr/bin/env python3
"""用真值探测「纯关键词规则」在情绪分类上能拿多少分（纯 CPU，零 GPU）。

动机
----
情绪识别是 25 分、是两条线里权重大的一块。判断提示词规则有没有指望，先要知道
**光靠触发词能到哪**。这个脚本把 `prompts` 里那套触发词直接当成分类器套在
全量 2354 条真值上，量出：

1. 纯关键词分类器的准确率（对比「一律答 anxiety」的零成本下限 0.3190）；
2. 每个情绪类里，有多少条的用户原话**含有该类自己的触发词**（= 触发词的召回）；
3. 哪些类一条都抓不到 —— 那些必须靠语义判断，写再多触发词也没用。

用法::

    python tools/emotion_profile/emotion_rule_probe.py
    python tools/emotion_profile/emotion_rule_probe.py --fallback neutral
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

#: 第一步的触发词表 —— **必须与 prompts/emotion_profile.md 逐字一致**，
#: 否则审计结论就不可信（这一点是踩过坑的：曾经凭印象写成"踏实"，
#: 而提示词里其实是"踏实多了"，两者精确率差很多）。
#:
#: 顺序即优先级，先命中先算。mixed / neutral 没有词表（它们是定义式的），
#: 所以不在这里 —— 这也正是 §15.2 的发现。
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("anger", ("烦", "烦躁", "气", "凭什么", "受不了", "想摔键盘")),
    ("fear", ("怕", "害怕", "恐惧", "不敢")),
    ("helplessness", ("无力", "没办法", "改变不了", "只能忍着", "看不到头")),
    ("sadness", ("难过", "伤心", "委屈")),
    ("loneliness", ("孤单", "没人懂", "没人说话", "没人可以说")),
    ("anxiety", ("担心", "慌", "静不下来", "怕来不及", "悬着")),
    ("relaxed", ("轻松", "放下了", "踏实多了", "平静")),
    ("gratitude", ("谢谢", "感谢")),
    ("joy", ("开心", "太好了", "期待")),
    ("surprise", ("没想到", "居然", "怎么会")),
    ("shame", ("丢脸", "都怪我", "我不配")),
    ("disgust", ("恶心", "反感", "受不了这种东西")),
    ("care", ("我为你担心", "你还好吗")),
    ("pride", ("我做到了", "还不错吧")),
]

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
    "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
]


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_user_turn(sample: dict) -> str:
    """取最后一条用户消息。"""
    turns = [t for t in sample.get("history", []) if t.get("role") == "user"]
    if not turns:
        return ""
    turns.sort(key=lambda t: t.get("turn_id", 0))
    return turns[-1].get("content", "") or ""


def classify(text: str, fallback: str) -> str:
    """按 RULES 顺序匹配；都不中就用 fallback。"""
    for emotion, patterns in RULES:
        for pattern in patterns:
            if pattern in text:
                return emotion
    return fallback


def load_rules(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """从提示词模板里解析第一步的触发词表。

    只认「以引号开头、含 `→ 某情绪`」的行 —— 那正是第一步的词表格式，例如：

        "烦""烦躁""气" → anger

    第二步的规则行（如 `对还没发生的事反复担心 → anxiety`）不以引号开头，会被跳过。
    `#` 开头的注释行也跳过。

    有了它，就能**直接拿任意候选模板估它的关键词分类器准确率**，
    而不必手抄词表（手抄正是 18.3 那次自我更正的起因）。
    """
    rules: list[tuple[str, tuple[str, ...]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "→" not in line:
            continue
        if not line.startswith('"'):
            continue
        head, _, tail = line.partition("→")
        emotion = tail.strip()
        if emotion not in EMOTIONS:
            continue
        words = re.findall(r'"([^"]+)"', head)
        if words:
            rules.append((emotion, tuple(words)))
    return rules


def main(argv: list[str] | None = None) -> int:
    global RULES

    parser = argparse.ArgumentParser(description="纯关键词情绪分类器探针")
    parser.add_argument("--fallback", default="anxiety", choices=EMOTIONS, help="规则都不中时的兜底类")
    parser.add_argument(
        "--explain",
        default=None,
        choices=[emotion for emotion, _ in RULES],
        help="只看某一个类的触发词：它会捞到哪些金标类的样本（用来判断这条规则是帮忙还是抢样本）",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="审计全部触发词：按「错配程度」排序，找出方向反了的词（fear 的『怕』就是这么找到的）",
    )
    parser.add_argument(
        "--rules-file",
        default=None,
        help="从提示词模板里读词表（不手抄）。可给多个用逗号分隔，会逐个跑并对比准确率",
    )
    args = parser.parse_args(argv)

    # 用模板里的词表替换内置 RULES：避免手抄出错（见报告 §18.3）
    if args.rules_file:
        loaded = []
        for item in args.rules_file.split(","):
            path = Path(item.strip())
            if not path.is_absolute():
                path = REPO_ROOT / path
            rules = load_rules(path)
            if not rules:
                raise SystemExit(f"从 {path} 里没解析到任何词表行")
            loaded.append((path.name, rules))
        if len(loaded) == 1:
            RULES = loaded[0][1]
        else:
            # 多份模板：逐个跑，只对比整体准确率
            references = read_jsonl(PROCESSED / "val_references.jsonl")
            samples = {row["id"]: row for row in read_jsonl(PROCESSED / "val_samples.jsonl")}
            rows = []
            for reference in references:
                sample = samples.get(reference["id"])
                if sample is None:
                    continue
                rows.append((last_user_turn(sample), (reference.get("target") or {}).get("emotion_label")))
            print(f"对比 {len(loaded)} 份模板的词表（全量 {len(rows)} 条，兜底 {args.fallback}）\n")
            print(f"{'模板':<28}{'词表条数':>9}{'触发条数':>9}{'准确率':>9}{'相对兜底':>10}")
            print("-" * 66)
            for name, rules in loaded:
                hits = fired = 0
                for text, gold in rows:
                    # 未命中任何规则时用兜底类 —— 漏掉这一步会把兜底的分算成 0，
                    # 得出「规则越少越差」的错误结论（这个坑踩过一次）。
                    predicted = args.fallback
                    for emotion, patterns in rules:
                        if any(pattern in text for pattern in patterns):
                            predicted = emotion
                            fired += 1
                            break
                    hits += 1 if predicted == gold else 0
                accuracy = hits / len(rows)
                print(f"{name:<28}{len(rules):>9}{fired:>9}{accuracy:>9.4f}{accuracy - 0.3190:>+10.4f}")
            print()
            print("（兜底准确率 = 一律答 anxiety = 0.3190；这是**关键词分类器**上的估计，")
            print("  真实模型未必那么字面，最终以切片实测为准。）")
            return 0

    references = read_jsonl(PROCESSED / "val_references.jsonl")
    samples = {row["id"]: row for row in read_jsonl(PROCESSED / "val_samples.jsonl")}

    texts: dict[str, str] = {}
    gold: dict[str, str] = {}
    for reference in references:
        sample = samples.get(reference["id"])
        if sample is None:
            continue
        texts[reference["id"]] = last_user_turn(sample)
        gold[reference["id"]] = (reference.get("target") or {}).get("emotion_label")

    total = len(texts)
    gold_counter = Counter(gold.values())
    majority, majority_count = gold_counter.most_common(1)[0]

    # --- 0b) 全量审计：找出方向反了的触发词 ---
    if args.audit:
        audit = []
        for emotion, patterns in RULES:
            for pattern in patterns:
                subset = [key for key, text in texts.items() if pattern in text]
                if not subset:
                    continue
                subset_gold = Counter(gold[key] for key in subset)
                own = subset_gold.get(emotion, 0)
                majority, majority_count = subset_gold.most_common(1)[0]
                audit.append({
                    "emotion": emotion,
                    "pattern": pattern,
                    "fired": len(subset),
                    "own": own,
                    "precision": own / len(subset),
                    "majority": majority,
                    "majority_count": majority_count,
                    "swing": majority_count - own,   # 把该词让给它的多数类，最多能净赚多少条
                })

        # 只关心触发量够大的词：小样本的精确率不可信
        MIN_FIRED = 25
        audit.sort(key=lambda item: -item["swing"])
        print(f"全量 {total} 条；只列出触发 ≥ {MIN_FIRED} 次的词（小样本精确率不可信）\n")
        print(f"{'词':<8}{'当前判给':<14}{'触发':>6}{'真本类':>7}{'精确率':>8}"
              f"{'最多金标类':>14}{'净赚上界':>9}  判")
        print("-" * 88)
        for item in audit:
            if item["fired"] < MIN_FIRED:
                continue
            misaligned = item["majority"] != item["emotion"] and item["precision"] < 0.35
            verdict = "❌ 方向反了" if misaligned else ("➖ 偏弱" if item["precision"] < 0.35 else "✅")
            print(f"{item['pattern']:<8}{item['emotion']:<14}{item['fired']:>6}{item['own']:>7}"
                  f"{item['precision']:>8.4f}{item['majority'] + ' ' + str(item['majority_count']):>14}"
                  f"{item['swing']:>+9}  {verdict}")

        print()
        bad = [item for item in audit if item["fired"] >= MIN_FIRED
               and item["majority"] != item["emotion"] and item["precision"] < 0.35]
        print(f"方向反了的词：{len(bad)} 个")
        for item in bad:
            print(f"  · 「{item['pattern']}」判给 {item['emotion']}，但 {item['fired']} 次触发里"
                  f"最多的是 {item['majority']}（{item['majority_count']} 条），本类只有 {item['own']} 条"
                  f" → 让出去最多净赚 {item['swing']} 条")
        print()
        print("读法：「净赚上界」= 把该词让给它的多数类时，最多能纠正的条数。")
        print("      真实收益要看让出去之后它们落到哪条规则，所以这是上界；")
        print("      但上界为负或很大的正数，都说明这个词该搬走。")
        return 0

    # --- 0) 单类解释模式：这条触发词到底捞到了谁的样本 ---
    if args.explain:
        patterns = dict(RULES)[args.explain]
        fired = {key: text for key, text in texts.items()
                 if any(pattern in text for pattern in patterns)}
        gold_of_fired = Counter(gold[key] for key in fired)
        print(f"规则 `{args.explain}` 的触发词：{patterns}")
        print(f"全量 {total} 条里触发了 {len(fired)} 条。这些条目的金标分布：\n")
        print(f"{'金标情绪':<16}{'条数':>7}{'占触发比例':>12}   （该类全量条数）")
        print("-" * 62)
        for emotion, count in gold_of_fired.most_common():
            print(f"{emotion:<16}{count:>7}{count / len(fired):>12.4f}   ({gold_counter.get(emotion, 0)})")
        hit = gold_of_fired.get(args.explain, 0)
        print()
        print(f"触发里真正是 {args.explain} 的：{hit} / {len(fired)} = {hit / len(fired):.4f}（精确率）")
        stolen = len(fired) - hit
        print(f"被这条规则「抢走」的其它类样本：{stolen} 条")
        if stolen:
            top = [(emotion, count) for emotion, count in gold_of_fired.most_common() if emotion != args.explain][:5]
            print(f"  抢得最多的：{top}")
            print("  → 如果被抢的主要是更大的类（如 anxiety），把这条规则收窄通常对总体 Acc 有利；")
            print("    如果被抢的是同样小的类，收窄的收益有限。")

        # 逐词拆解：到底是哪一个词把方向带偏的
        print()
        print("  逐词拆解（同一类里，各触发词单独看）：")
        print(f"  {'触发词':<10}{'触发':>6}{'真' + args.explain:>10}{'精确率':>9}  触发里最多的金标类")
        print("  " + "-" * 66)
        for pattern in patterns:
            subset = {key: text for key, text in texts.items() if pattern in text}
            if not subset:
                print(f"  {pattern:<10}{0:>6}{'—':>10}{'—':>9}")
                continue
            subset_gold = Counter(gold[key] for key in subset)
            subset_hit = subset_gold.get(args.explain, 0)
            top_other = [(e, c) for e, c in subset_gold.most_common(3)]
            print(f"  {pattern:<10}{len(subset):>6}{subset_hit:>10}{subset_hit / len(subset):>9.4f}  {top_other}")
        print()
        print("  读法：某个词如果触发量大、精确率低、且最多金标类不是本类，")
        print("        把这个词移走（或移到它真正的多数类）通常是纯赚。")
        return 0

    # --- 1) 纯关键词分类器的整体准确率 ---
    predicted = {key: classify(text, args.fallback) for key, text in texts.items()}
    correct = sum(1 for key in texts if predicted[key] == gold[key])
    print(f"全量 {total} 条；零成本下限（一律 {majority}）= {majority_count / total:.4f}")
    print(f"纯关键词分类器（兜底 {args.fallback}）准确率 = {correct / total:.4f}")
    print(f"  与下限之差 = {(correct - majority_count) / total:+.4f}\n")

    # --- 2) 逐类：触发词的召回与精确率 ---
    pattern_of = {emotion: patterns for emotion, patterns in RULES}

    def own_trigger(text: str, emotion: str) -> bool:
        """该类自己的触发词是否出现（不看优先级顺序）。"""
        return any(pattern in text for pattern in pattern_of.get(emotion, ()))

    print(f"{'情绪':<16}{'金标数':>7}{'自身触发':>9}{'判对':>7}{'预测数':>8}{'精确率':>9}  说明")
    print("-" * 80)
    rows = []
    for emotion in EMOTIONS:
        support = gold_counter.get(emotion, 0)
        fired = sum(1 for key in texts if gold[key] == emotion and own_trigger(texts[key], emotion))
        true_positive = sum(1 for key in texts if gold[key] == emotion and predicted[key] == emotion)
        predicted_count = sum(1 for value in predicted.values() if value == emotion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        trigger_rate = fired / support if support else 0.0
        recall = true_positive / support if support else 0.0
        note = ""
        if support and fired == 0:
            note = "自身触发词一条都没出现 → 只能靠语义判断"
        elif support and trigger_rate < 0.10:
            note = "自身触发词几乎抓不到"
        elif support and recall == 0:
            note = "有触发词但全被更高优先级的类抢走"
        rows.append((emotion, support, trigger_rate, recall, predicted_count, precision, note))

    for emotion, support, trigger_rate, recall, predicted_count, precision, note in sorted(
        rows, key=lambda item: -item[1]
    ):
        print(f"{emotion:<16}{support:>7}{trigger_rate:>9.4f}{recall:>7.4f}"
              f"{predicted_count:>8}{precision:>9.4f}  {note}")

    # --- 3) 兜底类吃了多少（anxiety 陷阱的量化） ---
    print()
    fallback_share = sum(1 for value in predicted.values() if value == args.fallback) / total
    print(f"兜底类 {args.fallback} 占了预测的 {fallback_share:.4f}；"
          f"金标里 {args.fallback} 只占 {gold_counter.get(args.fallback, 0) / total:.4f}")
    print()
    print("读法：")
    print("  · 「触发词召回」= 该类金标里，有多少条的用户原话含该类自己的触发词。")
    print("    这个数很低，说明关键词规则对此类无效，提示词再堆触发词也没用。")
    print("  · 纯关键词分类器的准确率若低于下限，说明**规则本身**不是答案，")
    print("    价值在于给模型「可判定的抓手」，而不是替代模型判断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
