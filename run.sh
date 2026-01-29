export GOOGLE_APPLICATION_CREDENTIALS=/home/huihuihuang/Hint/huihui-472807-b8069b88637c.json
export PATH=/home/huihuihuang/Hint/codeql:$PATH
CPP_QUERIES_DIR="/home/huihuihuang/Hint/codeql/qlpacks/codeql/cpp-queries"

# CASE selection:
# - Set RUN_ALL=true to run every case below.
# - Or set RUN_ALL=false and list desired case names in RUN_CASES.
#   Example: RUN_CASES=("redis" "zstd_issue4112")
RUN_ALL=false
RUN_CASES=("redis-8_6-rc1")
# MODE options:
#   single_with_hints  - use --single-source hints✅ enhanced query✅
#   full_with_hints    - analyze full project with hints✅ enhanced query✅
#   full_no_hints      - analyze full project without hints❌ enhanced query✅
#   full_with_hints_no_enhanced_queries - analyze full project with hints✅ enhanced query❌
#   full_no_hints_no_enhanced_queries - analyze full project without hints❌ enhanced query❌
MODE="full_with_hints"

case "$MODE" in
  single_with_hints)
    USE_SINGLE_SOURCE=true
    USE_HINTS=true
    USE_ENHANCED_QUERIES=true
    ;;
  full_with_hints)
    USE_SINGLE_SOURCE=false
    USE_HINTS=true
    USE_ENHANCED_QUERIES=true
    ;;
  full_no_hints)
    USE_SINGLE_SOURCE=false
    USE_HINTS=false
    USE_ENHANCED_QUERIES=true
    ;;
  full_with_hints_no_enhanced_queries)
    USE_SINGLE_SOURCE=false
    USE_HINTS=true
    USE_ENHANCED_QUERIES=false
    ;;
  full_no_hints_no_enhanced_queries)
    USE_SINGLE_SOURCE=false
    USE_HINTS=false
    USE_ENHANCED_QUERIES=false
    ;;

  *)
    echo "Unknown MODE: $MODE"
    exit 1
    ;;
esac

if [ "$USE_HINTS" = false ]; then
  SKIP_HINTS_FLAG="--skip-hints"
  SKIP_HINTS_NAME="_skip_hints"
else
  SKIP_HINTS_FLAG=""
  SKIP_HINTS_NAME=""
fi

if [ "$USE_ENHANCED_QUERIES" = false ]; then
  ENHANCED_QUERIES_FLAG="--no-enhanced-queries"
  ENHANCED_QUERIES_NAME="_no_enhanced_queries"
else
  ENHANCED_QUERIES_FLAG=""
  ENHANCED_QUERIES_NAME=""
fi

adjust_output_path() {
  local base="$1"
  local out="$base"

  if [ "$USE_SINGLE_SOURCE" = true ]; then
    out="${out}_single_source"
  else
    out="${out}_full"
  fi

  if [ "$USE_HINTS" = false ]; then
    out="${out}${SKIP_HINTS_NAME}"
  fi
  if [ "$USE_ENHANCED_QUERIES" = false ]; then
    out="${out}${ENHANCED_QUERIES_NAME}"
  fi

  echo "$out"
}

prepare_hints_from_single() {
  local base="$1"
  local output_path="$2"

  # Only reuse hints when running full project with hints but without enhanced queries
  if [ "$MODE" != "full_with_hints_no_enhanced_queries" ]; then
    return
  fi
  echo "Reusing hints from ${base}"
  local single_output="${base}_single_source"
  local single_hints="${single_output}/hints.json"
  local dest_hints="${output_path}/hints.json"

  if [ -f "$single_hints" ]; then
    mkdir -p "$output_path"
    cp "$single_hints" "$dest_hints"
    echo "Reusing hints from ${single_hints}"
  else
    echo "No hints.json found in ${single_output}; proceeding without cached hints"
  fi
}

should_run() {
  local name="$1"
  if [ "$RUN_ALL" = true ]; then
    return 0
  fi
  local c
  for c in "${RUN_CASES[@]}"; do
    if [ "$c" = "$name" ]; then
      return 0
    fi
  done
  return 1
}

run_case() {
    local project_path="$1"
    local output_path="$2"
    shift 2

    python main.py --no-reuse-db --project "$project_path" --output "$output_path" \
        --cpp-queries-dir "$CPP_QUERIES_DIR" \
        $SKIP_HINTS_FLAG \
        $ENHANCED_QUERIES_FLAG \
        --source-root "$project_path" \
        "$@"
}
##########
# Redis #
##########
if should_run "redis"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/redis
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/redis/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/redis/src/replication.c" "/home/huihuihuang/Hint/cloned_proj/redis/src/redis-benchmark.c" "/home/huihuihuang/Hint/cloned_proj/redis/src/server.h"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --source-root "/home/huihuihuang/Hint/cloned_proj/redis/src"
fi
fi

##########
# Htop #
##########
if should_run "htop"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/htop
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/htop/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/htop/Settings.c" "/home/huihuihuang/Hint/cloned_proj/htop/XUtils.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue3937 using tmux_pr3941_before#
#########
if should_run "tmux_issue3937"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3937_pr3941_before/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/arguments.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue3938 using tmux_pr3941_before#
#########
if should_run "tmux_issue3938"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3938_pr3941_before/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/cmd-command-prompt.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi


##########
# tmux issue3939 using tmux_pr3941_before#
#########
if should_run "tmux_issue3939"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3939_pr3941_before/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/cmd-confirm-before.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue3940 using tmux_pr3941_before#
#########
if should_run "tmux_issue3940"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3940_pr3941_before/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/layout-custom.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/layout.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue3949 using tmux_pr3972_before#
#########
if should_run "tmux_issue3949"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3949_pr3972_before/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before/arguments.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue3950 using tmux_20240420#
#########
if should_run "tmux_issue3950"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3950_20240420/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/cmd-confirm-before.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue3951 using tmux_20240420#
#########
if should_run "tmux_issue3951"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3951_20240420/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/status.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/compat/reallocarray.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue3952 using tmux_20240420#
#########
if should_run "tmux_issue3952"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3952_20240420/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/status.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue4298 using tmux_20241213#
#########
if should_run "tmux_issue4298"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20241213
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue4298_20241213/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20241213/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20241213/mode-tree.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# tmux issue4394 using tmux_20250303#
#########
if should_run "tmux_issue4394"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20250303
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/tmux_issue4394_20250303/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/window.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/utf8.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/compat/reallocarray.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# zstd issue4112 using zstd_pr4115_before#
#########
if should_run "zstd_issue4112"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/zstd_issue4112_pr4115_before/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/compress/zstd_compress.c" "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/common/allocations.h" "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/common/zstd_deps.h"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# vim issue14254 using vim_915f3_before#
#########
if should_run "vim_issue14254"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/vim_915f3_before
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/vim_issue14254_915f3_before/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/vim_915f3_before/src/vim9expr.c" "/home/huihuihuang/Hint/cloned_proj/vim_915f3_before/src/dict.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# vim issue14255 using vim_915f3_before#
#########
if should_run "vim_issue14255"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/vim_915f3_before
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj/vim_issue14255_915f3_before/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/vim_915f3_before/src/memline.c" "/home/huihuihuang/Hint/cloned_proj/vim_915f3_before/src/memline.c" "/home/huihuihuang/Hint/cloned_proj/vim_915f3_before/src/crypt.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# example2-1: Standard version (malloc/free)#
#########
if should_run "example2-1"; then
PROJECT_PATH=/home/huihuihuang/Hint/data/example2-1
BASE_OUTPUT="/home/huihuihuang/Hint/output/data/example2-1/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/data/example2-1/main_standard.cpp"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi

##########
# example2-2: Custom micro version (MICRO_malloc/MICRO_free)#
#########
if should_run "example2-2"; then
PROJECT_PATH=/home/huihuihuang/Hint/data/example2-2
BASE_OUTPUT="/home/huihuihuang/Hint/output/data/example2-2/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/data/example2-2/main_micro.cpp"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
fi


##########
# Redis-8.6-rc1 #
##########
if should_run "redis-8_6-rc1"; then
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj_test/redis-8_6-rc1
BASE_OUTPUT="/home/huihuihuang/Hint/output/cloned_proj_test/redis-8_6-rc1/output"
OUTPUT_PATH=$(adjust_output_path "$BASE_OUTPUT")
prepare_hints_from_single "$BASE_OUTPUT" "$OUTPUT_PATH"
if [ "$USE_SINGLE_SOURCE" = true ]; then
  echo "Not supported"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --source-root "/home/huihuihuang/Hint/cloned_proj_test/redis-8_6-rc1/src"
fi
fi