#!/usr/bin/env bash
# 镜像内离线自检。组委会是在**离线环境**里跑我们的镜像的，所以「联网才能跑」
# 等于「跑不起来」= 零分。这个脚本就在镜像里跑，验证断网也能出结果。
#
# 在镜像内执行：
#   bash /opt/offline_check.sh
# 从外面带断网跑（最接近真实评测）：
#   docker run --rm --network none ogc-participant:latest bash /opt/offline_check.sh
# 只做静态检查、不真跑推理（省时间）：
#   SKIP_RUNTIME=1 bash /opt/offline_check.sh
#
# 退出码 0 = 全部通过；非 0 = 有失败项，看上面的 [FAIL] 行。

set -uo pipefail

PARTICIPANT="${PARTICIPANT_DIR:-/root/participant}"
CONTENT_ROOT="${PARTICIPANT_CONTENT_ROOT:-/root}"
PROMPT_DIR="${PARTICIPANT_PROMPT_DIR:-$PARTICIPANT/prompts}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/offline-check.XXXXXX")"
PASS=0
FAIL=0

ok()   { printf '  \033[32m[ OK ]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
info() { printf '  \033[36m[info]\033[0m %s\n' "$*"; }
head_() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

head_ "1. 参赛工具是否就位"
for f in start.sh run_inference.py configs/inference_config.json templates/submission_schema.json; do
  if [ -f "$PARTICIPANT/$f" ]; then ok "$PARTICIPANT/$f"; else bad "缺少 $PARTICIPANT/$f"; fi
done
[ -f "$PARTICIPANT/wiring.py" ] && ok "接线层 wiring.py 在（提示词/RAG 回退生效）" \
  || info "没有 wiring.py —— 按官方基线行为跑，仍然合规"

# 行尾检查：这是镜像内唯一能直接验到「真实文件」的地方，所以放在这里最靠前。
# CRLF 的 shell 脚本会让 bash 在第一行就报 bad interpreter，镜像直接判失败。
crlf_bad=0
for f in "$PARTICIPANT"/*.sh; do
  [ -f "$f" ] || continue
  if LC_ALL=C grep -q $'\r' "$f" 2>/dev/null; then
    bad "$(basename "$f") 带 CRLF 行尾 —— bash 会报 bad interpreter，必须转成 LF"
    crlf_bad=1
  fi
done
[ "$crlf_bad" -eq 0 ] && ok "所有 .sh 都是 LF 行尾（可被 bash 正常解析）"

head_ "2. 模型权重是否打进镜像"
# 镜像里权重必须在 $PARTICIPANT/model（官方按 model_path="./model" 相对解析）。
# 但 `build.sh --stage-only` 的暂存目录是「构建上下文」布局：model/ 与 participant/ 平级，
# Dockerfile 再把它 COPY 进 /root/participant/model。两个位置都认，本地才验得动。
MODEL_DIR=""
for cand in "$PARTICIPANT/model" "$CONTENT_ROOT/model"; do
  [ -f "$cand/config.json" ] && { MODEL_DIR="$cand"; break; }
done
if [ -n "$MODEL_DIR" ]; then
  ok "权重目录：$MODEL_DIR"
  [ "$MODEL_DIR" = "$PARTICIPANT/model" ] \
    || info "（构建上下文布局；Dockerfile 会把它 COPY 到 $PARTICIPANT/model）"
  weights="$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' | wc -l | tr -d ' ')"
  if [ "${weights:-0}" -gt 0 ]; then
    # du 不一定有（精简过的宿主/容器里常见），退化成 find+awk 求和，别让体积那一栏空着。
    if command -v du >/dev/null 2>&1; then
      total="$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)"
    fi
    if [ -z "${total:-}" ]; then
      total="$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -printf '%s\n' 2>/dev/null \
        | awk '{s+=$1} END {if (NR) printf "%.2f GB", s/1073741824}')"
    fi
    [ -n "${total:-}" ] || total="(体积算不出，缺 du 与 find -printf)"
    ok "$weights 个 safetensors 分片，共 $total"
  else
    bad "没有 .safetensors —— 权重没打进镜像"
  fi
else
  bad "找不到 model/config.json（试过 $PARTICIPANT/model 与 $CONTENT_ROOT/model）"
fi

head_ "3. Python 与 torch"
PY=""
for cand in python3 python; do
  command -v "$cand" >/dev/null 2>&1 && { PY="$cand"; break; }
done
if [ -n "$PY" ]; then
  ok "python: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
  if $PY -c 'import torch' 2>/dev/null; then
    ok "torch $($PY -c 'import torch; print(torch.__version__)')"
    $PY -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
      && info "CUDA 可用" || info "当前看不到 GPU（若是 --network none 的 CPU 试跑，属正常）"
  else
    bad "这个 python 里没有 torch —— 装依赖的基础镜像选错了"
  fi
  $PY -c 'import transformers' 2>/dev/null \
    && ok "transformers $($PY -c 'import transformers; print(transformers.__version__)')" \
    || bad "没有 transformers"
else
  bad "镜像里找不到 python"
fi

head_ "4. 静态扫描：代码里有没有外部网络依赖"
# 允许 127.0.0.1 / localhost（vLLM 本地服务）；其余外部主机一律报出来。
# 只扫「代码」(.py/.sh)。语料/数据里的 URL 是待检索的文本内容，不会发起任何请求，
# 上一版把它们当网络依赖报出来，是误报——评测时那只是一段字符串。
mapfile -t hits < <(
  find "$PARTICIPANT" "$CONTENT_ROOT/rag" -type f \( -name '*.py' -o -name '*.sh' \) 2>/dev/null \
    | xargs grep -InE 'https?://[^"'"'"' ]+' 2>/dev/null \
    | grep -vE '127\.0\.0\.1|localhost|0\.0\.0\.0|\$\{?[A-Z_]*HOST\}?' \
    | grep -vE ':[0-9]+:[[:space:]]*#' || true
)
if [ "${#hits[@]}" -eq 0 ]; then
  ok "代码里没有指向外部主机的 URL"
else
  bad "代码里有指向外部主机的 URL（必须确认，否则断网评测会失败）："
  printf '        %s\n' "${hits[@]:0:10}"
fi
# 数据文件里的 URL 单独报一行 info：不是网络依赖，但专家评审看到语料里带外链会提问，先点明。
data_urls="$(find "$CONTENT_ROOT/rag" -type f \( -name '*.jsonl' -o -name '*.json' \) 2>/dev/null \
  | xargs grep -lE 'https?://' 2>/dev/null | wc -l | tr -d ' ')"
[ "${data_urls:-0}" -gt 0 ] && info "检索语料里有 $data_urls 个数据文件含 URL 文本（只是待检索文本，不会联网）"
if grep -rInE 'pip install|huggingface_hub|snapshot_download' "$PARTICIPANT" 2>/dev/null \
     | grep -vE '\.md:|#' | head -5 | grep -q .; then
  bad "代码里出现 pip install / 在线下载调用"
else
  ok "没有 pip install / 在线下载调用"
fi

head_ "5. 提示词与检索模块的接线状态"
if [ -d "$PROMPT_DIR" ] && [ -n "$(find "$PROMPT_DIR" -maxdepth 1 -name '*.md' -o -maxdepth 1 -name '*.txt' 2>/dev/null)" ]; then
  ok "提示词目录已挂载：$(find "$PROMPT_DIR" -maxdepth 1 \( -name '*.md' -o -name '*.txt' \) -printf '%f ' 2>/dev/null)"
else
  info "提示词目录为空（$PROMPT_DIR）—— 会用官方内置提示词，合规但没有队友的调优"
fi
[ -f "$CONTENT_ROOT/rag/fusion.py" ] && ok "rag/fusion.py 在" \
  || info "没有 rag/fusion.py —— 组包走官方版"
[ -f "$CONTENT_ROOT/rag/retrieve.py" ] && ok "rag/retrieve.py 在" \
  || info "没有 rag/retrieve.py —— 不检索"

if [ "${SKIP_RUNTIME:-0}" = "1" ]; then
  head_ "6. 真跑推理（已按 SKIP_RUNTIME=1 跳过）"
else
  head_ "6. 断网真跑：3 条样例走完整入口"
  PROBE_IN="$WORK/probe_input.jsonl"
  cat > "$PROBE_IN" <<'JSONL'
{"id":"test1_000001_t5","conversation_id":"test1_conv_000001","target_user_turn_id":5,"history":[{"turn_id":1,"role":"user","content":"最近工作压力好大，晚上总是睡不着。"},{"turn_id":2,"role":"assistant","content":"听起来你这段时间挺难的，是有什么具体的事压着吗？"},{"turn_id":3,"role":"user","content":"项目要上线了，每天都加班到很晚。"},{"turn_id":4,"role":"assistant","content":"一直在绷着确实累。你自己有没有什么能松一松的时候？"},{"turn_id":5,"role":"user","content":"几乎没有，我现在连饭都是随便对付一口。"}]}
{"id":"test1_000001_t7","conversation_id":"test1_conv_000001","target_user_turn_id":7,"history":[{"turn_id":1,"role":"user","content":"最近工作压力好大，晚上总是睡不着。"},{"turn_id":2,"role":"assistant","content":"听起来你这段时间挺难的，是有什么具体的事压着吗？"},{"turn_id":3,"role":"user","content":"项目要上线了，每天都加班到很晚。"},{"turn_id":4,"role":"assistant","content":"一直在绷着确实累。你自己有没有什么能松一松的时候？"},{"turn_id":5,"role":"user","content":"几乎没有，我现在连饭都是随便对付一口。"},{"turn_id":6,"role":"assistant","content":"那身体真的会扛不住。要不先从吃饭这一件小事开始？"},{"turn_id":7,"role":"user","content":"说得对，明天我试着好好吃一顿。"}]}
{"id":"test1_000002_t5","conversation_id":"test1_conv_000002","target_user_turn_id":5,"history":[{"turn_id":1,"role":"user","content":"我和室友闹了点矛盾，不太想回宿舍。"},{"turn_id":2,"role":"assistant","content":"住在一起有摩擦挺常见的，发生什么了？"},{"turn_id":3,"role":"user","content":"她总是不打招呼就用我的东西。"},{"turn_id":4,"role":"assistant","content":"那确实会让人不舒服，你有跟她提过吗？"},{"turn_id":5,"role":"user","content":"没有，我怕说了会更尴尬。"}]}
JSONL

  OUT_DIR="$WORK/result"
  # 双保险：HF/transformers 离线开关 + 把代理指向死端口。真去联网就会立刻失败。
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9
  export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9
  export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

  info "执行：bash $PARTICIPANT/start.sh <3条样例> <结果目录>"
  if bash "$PARTICIPANT/start.sh" "$PROBE_IN" "$OUT_DIR" > "$WORK/run.log" 2>&1; then
    ok "入口脚本退出码 0"
  else
    bad "入口脚本失败，日志末尾："
    tail -n 25 "$WORK/run.log" | sed 's/^/        /'
  fi

  SUB="$OUT_DIR/submission.jsonl"
  if [ -f "$SUB" ]; then
    lines="$(wc -l < "$SUB")"
    [ "$lines" -eq 3 ] && ok "submission.jsonl 3 行，与输入条数一致" \
      || bad "submission.jsonl 是 $lines 行，应为 3 行"
    if $PY - "$SUB" <<'PYEOF'
import json, sys, pathlib
required = ("id", "response_text", "emotion_label", "user_profile", "memory_refs")
allowed = {"joy","gratitude","relaxed","care","pride","neutral","surprise","mixed",
           "sadness","loneliness","anxiety","anger","fear","disgust","shame",
           "helplessness",""}
rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
bad = []
for i, row in enumerate(rows, 1):
    missing = [k for k in required if k not in row]
    if missing:
        bad.append(f"第{i}行缺字段 {missing}")
        continue
    if row["emotion_label"] not in allowed:
        bad.append(f"第{i}行 emotion_label 非法: {row['emotion_label']!r}")
    profile = row["user_profile"]
    if not isinstance(profile, dict) or set(profile) != {"personality_traits","interests","style"}:
        bad.append(f"第{i}行 user_profile 维度不对: {profile!r}")
    if not isinstance(row["memory_refs"], list):
        bad.append(f"第{i}行 memory_refs 不是数组")
    if not isinstance(row["response_text"], str):
        bad.append(f"第{i}行 response_text 不是字符串")
if bad:
    print("\n".join("        " + b for b in bad)); raise SystemExit(1)
empty = sum(1 for r in rows if not r["response_text"].strip())
print(f"        {len(rows)} 行 schema 合规；空回复 {empty}/{len(rows)} 行")
raise SystemExit(0)
PYEOF
    then ok "submission.jsonl 每行都符合提交 schema"; else bad "submission.jsonl 有不合规行（见上）"; fi
    info "第 1 行预览：$(head -c 220 "$SUB")"
  else
    bad "没有生成 $SUB"
  fi

  RPT="$OUT_DIR/performance_report.json"
  if [ -f "$RPT" ]; then
    ok "performance_report.json 在"
    $PY - "$RPT" <<'PYEOF'
import json, sys, pathlib
d = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
need = ["backend","complete","required_rounds","rounds","average_latency_ms",
        "hardware_label","test_file_sha256","submission_sha256","samples"]
miss = [k for k in need if k not in d]
print("        缺失字段: " + str(miss) if miss else
      f"        backend={d.get('backend')} samples={d.get('samples')} "
      f"rounds={d.get('rounds')}/{d.get('required_rounds')} complete={d.get('complete')} "
      f"avg={d.get('average_latency_ms')}ms label={d.get('hardware_label')!r}")
PYEOF
    info "注意：rounds 只有够 100 条时才 complete=true。3 条样例必然 false，这是对的。"
  else
    bad "没有生成 $RPT"
  fi
  info "完整日志：$WORK/run.log"
fi

head_ "结果"
printf '  通过 %d 项，失败 %d 项\n' "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf '\n\033[31m离线自检未通过。这份镜像不要提交。\033[0m\n'
  exit 1
fi
printf '\n\033[32m离线自检通过。\033[0m\n'
exit 0
