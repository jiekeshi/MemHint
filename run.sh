export GOOGLE_APPLICATION_CREDENTIALS=/home/huihuihuang/Hint/huihui-472807-b8069b88637c.json
export PATH=/home/huihuihuang/Hint/codeql:$PATH
CPP_QUERIES_DIR="/home/huihuihuang/Hint/codeql/qlpacks/codeql/cpp-queries"

# MODE options:
#   single_with_hints  - use --single-source and generate hints
#   full_with_hints    - analyze full project with hints
#   full_no_hints      - analyze full project without hints
MODE="single_with_hints"

case "$MODE" in
  single_with_hints)
    USE_SINGLE_SOURCE=true
    SKIP_HINTS=false
    ;;
  full_with_hints)
    USE_SINGLE_SOURCE=false
    SKIP_HINTS=false
    ;;
  full_no_hints)
    USE_SINGLE_SOURCE=false
    SKIP_HINTS=true
    ;;
  *)
    echo "Unknown MODE: $MODE"
    exit 1
    ;;
esac

if [ "$SKIP_HINTS" = true ]; then
  SKIP_HINTS_FLAG="--skip-hints"
  SKIP_HINTS_NAME="_skip_hints"
else
  SKIP_HINTS_FLAG=""
  SKIP_HINTS_NAME=""
fi

adjust_output_path() {
  local base="$1"
  local out="$base"

  if [ "$USE_SINGLE_SOURCE" = true ]; then
    out="${out}_single_source"
  else
    out="${out}_full"
  fi

  if [ "$SKIP_HINTS" = true ]; then
    out="${out}${SKIP_HINTS_NAME}"
  fi

  echo "$out"
}

run_case() {
    local project_path="$1"
    local output_path="$2"
    shift 2

    python main.py --debug --no-reuse-db --project "$project_path" --output "$output_path" \
        --cpp-queries-dir "$CPP_QUERIES_DIR" \
        $SKIP_HINTS_FLAG \
        "$@"
}
##########
# Redis #
##########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/redis
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/redis/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/redis/src/replication.c" "/home/huihuihuang/Hint/cloned_proj/redis/src/redis-benchmark.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# Htop #
##########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/htop
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/htop/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/htop/Settings.c" "/home/huihuihuang/Hint/cloned_proj/htop/XUtils.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue3937 using tmux_pr3941_before#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3937_pr3941_before/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/arguments.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue3938 using tmux_pr3941_before#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3938_pr3941_before/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/cmd-command-prompt.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi


##########
# tmux issue3939 using tmux_pr3941_before#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3939_pr3941_before/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/cmd-confirm-before.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue3940 using tmux_pr3941_before#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3940_pr3941_before/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/layout-custom.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/layout.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue3949 using tmux_pr3972_before#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3949_pr3972_before/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before/arguments.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue3950 using tmux_20240420#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3950_20240420/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/cmd-confirm-before.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue3951 using tmux_20240420#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3951_20240420/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/status.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/compat/reallocarray.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue3952 using tmux_20240420#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3952_20240420/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/status.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue4298 using tmux_20241213#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20241213
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue4298_20241213/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20241213/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20241213/mode-tree.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# tmux issue4394 using tmux_20250303#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20250303
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/tmux_issue4394_20250303/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/window.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/utf8.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/compat/reallocarray.c"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi

##########
# zstd issue4112 using zstd_pr4115_before#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before
OUTPUT_PATH=$(adjust_output_path "/home/huihuihuang/Hint/output/cloned_proj/zstd_issue4112_pr4115_before/output")
if [ "$USE_SINGLE_SOURCE" = true ]; then
  run_case "$PROJECT_PATH" "$OUTPUT_PATH" \
    --single-source "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/compress/zstd_compress.c" "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/common/allocations.h" "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/common/zstd_deps.h"
else
  run_case "$PROJECT_PATH" "$OUTPUT_PATH"
fi
