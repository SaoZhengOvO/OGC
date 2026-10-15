"""代码规范巡检：给 `docs/reports/code-review.md` 提供可复现的证据。

跑法::

    python docs/reports/audit_code.py

产物：`docs/reports/code-review.audit.json`（原文，供专家复算）+ 屏幕摘要。

只看不写 —— 本脚本不会修改任何被检查的文件。检查范围是 ⑤B 任务 B 指定的四个目录：
`src/`、`rag/`、`deploy/`、`tools/`（外加 `tests/`、`docs/reports/` 自身的公开脚本）。

标准来自 `src/*.py`：模块 docstring、公开函数有文档字符串、有类型标注。
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_JSON = Path(__file__).resolve().parent / "code-review.audit.json"

#: 检查哪些目录（相对仓库根）。`docs/reports` 也在列 —— 评审脚本自己也要接受同一把尺子。
TARGET_DIRS = ("src", "rag", "deploy", "tools", "tests", "docs/reports")

#: 跳过这些路径片段（虚拟环境、缓存、第三方）。
SKIP_PARTS = (".venv", "__pycache__", "site-packages", "dist", "context")

#: Windows 绝对路径（`C:\...` / `D:/...`）—— 写死的话换台机器就跑不起来。
#: 前面的负向断言是必须的：没有它会连 `https://` 里的 `s:/` 一起匹配上（踩过这个假阳性）。
WINDOWS_ABS_PATH = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]):[\\/]")

#: 容器内的绝对路径。不是错，但要人确认是有意的。
CONTAINER_ABS_PATH = re.compile(r"[\"'](/root|/opt|/workspace)/")

#: 常用标记。
TODO_MARKER = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")

#: 一行超过多少字符提示（PEP 8 是 79，工程上一般放到 120）。
LONG_LINE = 120

#: 重复代码块的最短长度（规范化后的行数）。
DUP_WINDOW = 6

#: 巡检工具自己的源码天然包含它要搜的那些字面量（`TODO|FIXME` 的正则、
#: 路径示例的注释），逐行文本检查会对自身产生假阳性。这些文件只做 AST 检查
#: （docstring / 类型标注 / 中文标识符），不做逐行文本检查。
TEXT_CHECK_EXEMPT = {"docs/reports/audit_code.py"}

#: 标识符里的中日韩字符 —— 变量名/函数名应当是 ASCII，中文只该出现在
#: 文档字符串和注释里（本仓库的既定风格）。
CJK_IN_IDENTIFIER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def cjk_identifiers(tree: ast.AST) -> list[dict]:
    """找出名字里带中文字符的标识符。"""
    hits: list[dict] = []
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Name):
            candidates = [node.id]
        elif isinstance(node, ast.Attribute):
            candidates = [node.attr]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates = [node.name]
        elif isinstance(node, ast.arg):
            candidates = [node.arg]
        elif isinstance(node, ast.keyword) and node.arg:
            candidates = [node.arg]
        for name in candidates:
            if CJK_IN_IDENTIFIER.search(name):
                hits.append({"line": getattr(node, "lineno", 0), "name": name})
    return hits


def bare_excepts(tree: ast.AST) -> list[dict]:
    """裸 `except:`，以及只写了 `pass` 的吞异常分支。"""
    hits: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            hits.append({"line": node.lineno, "kind": "bare_except"})
        elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            hits.append({"line": node.lineno, "kind": "except_pass"})
    return hits


def iter_python_files() -> list[Path]:
    found: list[Path] = []
    for name in TARGET_DIRS:
        root = REPO_ROOT / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in path.parts for part in SKIP_PARTS):
                continue
            found.append(path)
    return found


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def is_test_file(name: str) -> bool:
    """测试文件里的 `Test*` / `test_*` 按惯例不要求文档字符串，需单独统计。"""
    return name.startswith("tests/") or Path(name).name.startswith("test_")


def is_conventional_test_symbol(entry: dict, file_name: str) -> bool:
    if not is_test_file(file_name):
        return False
    return entry["name"].startswith("test_") or entry["name"].startswith("Test")


def _missing_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    missing: list[str] = []
    for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            missing.append(arg.arg)
    if node.args.vararg and node.args.vararg.annotation is None:
        missing.append("*" + node.args.vararg.arg)
    if node.args.kwarg and node.args.kwarg.annotation is None:
        missing.append("**" + node.args.kwarg.arg)
    return missing


def scan_file(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    name = rel(path)
    skip_text_checks = name in TEXT_CHECK_EXEMPT
    record: dict = {
        "file": name,
        "lines": len(lines),
        "text_checks_skipped": skip_text_checks,
        "module_docstring": False,
        "parse_error": None,
        "public_functions": [],
        "private_functions": 0,
        "classes": [],
        "windows_abs_path": [],
        "container_abs_path": [],
        "markers": [],
        "long_lines": [],
        "cjk_identifiers": [],
        "bare_excepts": [],
    }

    for index, line in enumerate(lines, start=1):
        if skip_text_checks:
            break
        if WINDOWS_ABS_PATH.search(line):
            record["windows_abs_path"].append({"line": index, "text": line.strip()[:160]})
        if CONTAINER_ABS_PATH.search(line):
            record["container_abs_path"].append({"line": index, "text": line.strip()[:160]})
        if TODO_MARKER.search(line):
            record["markers"].append({"line": index, "text": line.strip()[:160]})
        if len(line) > LONG_LINE:
            record["long_lines"].append({"line": index, "length": len(line)})

    try:
        tree = ast.parse(source)
    except SyntaxError as error:  # pragma: no cover - 语法坏了本身就是发现
        record["parse_error"] = f"{error.msg} (line {error.lineno})"
        return record

    record["module_docstring"] = bool(ast.get_docstring(tree))
    record["has_main_guard"] = '__main__' in source
    record["cjk_identifiers"] = cjk_identifiers(tree)
    record["bare_excepts"] = bare_excepts(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entry = {
                "name": node.name,
                "line": node.lineno,
                "docstring": bool(ast.get_docstring(node)),
                "missing_annotations": _missing_annotations(node),
                "missing_return": node.returns is None,
            }
            if node.name.startswith("_"):
                record["private_functions"] += 1
            else:
                record["public_functions"].append(entry)
        elif isinstance(node, ast.ClassDef):
            record["classes"].append(
                {"name": node.name, "line": node.lineno, "docstring": bool(ast.get_docstring(node))}
            )

    return record


def normalize(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    return re.sub(r"\s+", " ", stripped)


def find_duplicate_blocks(files: list[Path]) -> list[dict]:
    """找跨文件/跨函数重复的连续代码块（规范化后逐行比对）。"""
    index: dict[str, list[tuple[str, int]]] = {}
    for path in files:
        normalized = [normalize(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for start in range(len(normalized) - DUP_WINDOW + 1):
            window = normalized[start : start + DUP_WINDOW]
            if any(not line for line in window):
                continue
            key = "\n".join(window)
            index.setdefault(key, []).append((rel(path), start + 1))

    findings: list[dict] = []
    seen_headers: set[str] = set()
    for key, places in index.items():
        if len(places) < 2:
            continue
        header = key.split("\n")[0][:80]
        if header in seen_headers:  # 同一段重复只报一次，避免刷屏
            continue
        seen_headers.add(header)
        findings.append(
            {"first_line": header, "occurrences": [f"{name}:{line}" for name, line in places]}
        )
    findings.sort(key=lambda item: -len(item["occurrences"]))
    return findings


def summarize(records: list[dict], duplicates: list[dict]) -> dict:
    def places(predicate) -> list[str]:
        return [
            f"{record['file']}:{fn['line']} {fn['name']}"
            for record in records
            for fn in record["public_functions"]
            if predicate(record, fn)
        ]

    everything_missing_docs = places(lambda _r, fn: not fn["docstring"])
    production_missing_docs = places(
        lambda r, fn: not fn["docstring"] and not is_conventional_test_symbol(fn, r["file"])
    )
    public = [fn for record in records for fn in record["public_functions"]]
    return {
        "files_scanned": len(records),
        "total_lines": sum(record["lines"] for record in records),
        "production_files": [r["file"] for r in records if not is_test_file(r["file"])],
        "modules_without_docstring": [r["file"] for r in records if not r["module_docstring"]],
        "parse_errors": [{"file": r["file"], "error": r["parse_error"]} for r in records if r["parse_error"]],
        "public_functions": len(public),
        "public_without_docstring": everything_missing_docs,
        "public_without_docstring_production": production_missing_docs,
        "public_without_docstring_tests": len(everything_missing_docs) - len(production_missing_docs),
        "public_without_return_annotation": [
            f"{record['file']}:{fn['line']} {fn['name']}"
            for record in records
            for fn in record["public_functions"]
            if fn["missing_return"] and not is_conventional_test_symbol(fn, record["file"])
        ],
        "public_without_param_annotations": [
            f"{record['file']}:{fn['line']} {fn['name']} -> {fn['missing_annotations']}"
            for record in records
            for fn in record["public_functions"]
            if fn["missing_annotations"] and not is_conventional_test_symbol(fn, record["file"])
        ],
        "classes_without_docstring": [
            f"{record['file']}:{cls['line']} {cls['name']}"
            for record in records
            for cls in record["classes"]
            if not cls["docstring"]
            and not (is_test_file(record["file"]) and cls["name"].startswith("Test"))
        ],
        "windows_abs_paths": [
            {"place": f"{record['file']}:{hit['line']}", "text": hit["text"]}
            for record in records
            for hit in record["windows_abs_path"]
        ],
        "container_abs_paths": [
            {"place": f"{record['file']}:{hit['line']}", "text": hit["text"]}
            for record in records
            for hit in record["container_abs_path"]
        ],
        "markers": [
            {"place": f"{record['file']}:{hit['line']}", "text": hit["text"]}
            for record in records
            for hit in record["markers"]
        ],
        "long_lines": [
            {"place": f"{record['file']}:{hit['line']}", "length": hit["length"]}
            for record in records
            for hit in record["long_lines"]
        ],
        "cjk_identifiers": [
            {"place": f"{record['file']}:{hit['line']}", "name": hit["name"]}
            for record in records
            for hit in record["cjk_identifiers"]
        ],
        "bare_excepts": [
            {"place": f"{record['file']}:{hit['line']}", "kind": hit["kind"]}
            for record in records
            for hit in record["bare_excepts"]
        ],
        "duplicate_blocks": duplicates,
    }


def print_report(summary: dict) -> None:
    print(
        f"扫描 {summary['files_scanned']} 个文件 / {summary['total_lines']} 行"
        f"（生产代码 {len(summary['production_files'])} 个）\n"
    )

    def show(title: str, items: list, limit: int = 14) -> None:
        print(f"== {title}：{len(items)}")
        for item in items[:limit]:
            print(f"   - {item}")
        if len(items) > limit:
            print(f"   ... 另有 {len(items) - limit} 条（见 JSON）")
        print()

    show("模块缺 docstring", summary["modules_without_docstring"])
    show("生产代码公开函数缺文档字符串", summary["public_without_docstring_production"])
    print(f"== （另有 {summary['public_without_docstring_tests']} 个测试方法缺文档字符串，按惯例豁免）\n")
    show("公开函数缺返回类型", summary["public_without_return_annotation"])
    show("公开函数缺参数类型", summary["public_without_param_annotations"])
    show("类缺 docstring", summary["classes_without_docstring"])
    show("Windows 绝对路径", [f"{h['place']}  {h['text']}" for h in summary["windows_abs_paths"]])
    show("容器绝对路径（确认是否有意）", [f"{h['place']}  {h['text']}" for h in summary["container_abs_paths"]])
    show("TODO/FIXME 标记", [f"{h['place']}  {h['text']}" for h in summary["markers"]])
    show(
        "重复代码块（连续 6 行）",
        [f"{d['occurrences']}  <- {d['first_line']}" for d in summary["duplicate_blocks"]],
    )
    show("超过 120 字符的行", [f"{h['place']}  ({h['length']} 字符)" for h in summary["long_lines"]])
    show("中文标识符", [f"{h['place']}  {h['name']}" for h in summary["cjk_identifiers"]])
    show(
        "except 分支",
        [f"{h['place']}  {h['kind']}" for h in summary["bare_excepts"]],
    )


def main() -> int:
    files = iter_python_files()
    if not files:
        print("没找到要扫描的 Python 文件", file=sys.stderr)
        return 1

    records = [scan_file(path) for path in files]
    duplicates = find_duplicate_blocks(files)
    summary = summarize(records, duplicates)

    AUDIT_JSON.write_text(
        json.dumps({"summary": summary, "per_file": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print_report(summary)
    print(f"\n完整结果 -> {AUDIT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
