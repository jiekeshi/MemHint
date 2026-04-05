#!/usr/bin/env bash
# ============================================================================
# MemHint Replication Script
#
# Reproduces the evaluation from:
#   "Finding Memory Leaks in C/C++ Programs via Neuro-Symbolic Augmented
#    Static Analysis"
#
# Evaluated projects: Vim, tmux, OpenSSL, Redis, FreeRDP, curl, FFmpeg
#
# Usage:
#   1. Clone the 7 subject projects into subjects/ (see README.md)
#   2. Set GEMINI_API_KEY below
#   3. Run: bash run.sh
# ============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Gemini API key (required)
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

# LLM models (as reported in paper)
#   Stage 1, Phase 2 (summary generation): Gemini 3 Flash
#   Stage 3, Phase 6 (warning validation):  Gemini 3.1 Pro
LLM_MODEL="gemini-3-flash-preview"
LLM_VERIFY_MODEL="gemini-3.1-pro-preview"

# Batch size: 20 functions per LLM call (following IRIS batching strategy)
export HINT_BATCH_SIZE=20

# Analyzer: "codeql" or "infer" (run both to replicate full results)
ANALYZER="codeql"

# CodeQL queries directory (adjust to your CodeQL installation)
# Example: CPP_QUERIES_DIR="/path/to/codeql/qlpacks/codeql/cpp-queries"
CPP_QUERIES_DIR="${CPP_QUERIES_DIR:-}"

# Enable LLM-based warning validation (Stage 3, Phase 6)
USE_LLM_VERIFY=true

# Pipeline mode: "full" (all 3 stages)
PIPELINE_MODE="full"

# Hint resume: "fresh" to re-run all, "raw" to resume from partial results
HINT_RESUME_MODE="fresh"

# Directory containing cloned subject projects
SUBJECTS_DIR="${SUBJECTS_DIR:-subjects}"

# =============================================================================
# Which projects to run
# =============================================================================
# Set RUN_ALL=true to run all 7 projects, or list specific ones in RUN_CASES.

RUN_ALL=true
RUN_CASES=("vim" "tmux" "openssl" "redis" "freerdp" "curl" "ffmpeg")

# =============================================================================
# Project definitions (7 evaluated projects from the paper)
# =============================================================================

run_case_by_name() {
    local name="$1"
    case "$name" in

        vim)
            run_case \
                "${SUBJECTS_DIR}/vim_9_2_0015" \
                "output/vim_9_2_0015"
            ;;

        tmux)
            run_case \
                "${SUBJECTS_DIR}/tmux_3_6_a" \
                "output/tmux_3_6_a"
            ;;

        openssl)
            run_case \
                "${SUBJECTS_DIR}/openssl_3_6_1" \
                "output/openssl_3_6_1"
            ;;

        redis)
            run_case \
                "${SUBJECTS_DIR}/redis-8_6-rc1" \
                "output/redis-8_6-rc1" \
                --source-root "${SUBJECTS_DIR}/redis-8_6-rc1/src"
            ;;

        freerdp)
            run_case \
                "${SUBJECTS_DIR}/FreeRDP_3_23_0" \
                "output/FreeRDP_3_23_0"
            ;;

        curl)
            run_case \
                "${SUBJECTS_DIR}/curl_rc-8_19_0-3" \
                "output/curl_rc-8_19_0-3"
            ;;

        ffmpeg)
            run_case \
                "${SUBJECTS_DIR}/FFmpeg_n_8_1_dev" \
                "output/FFmpeg_n_8_1_dev"
            ;;

        *)
            echo "Error: Unknown project '$name'"
            echo "Available: vim, tmux, openssl, redis, freerdp, curl, ffmpeg"
            exit 1
            ;;
    esac
}

# =============================================================================
# Internal helpers
# =============================================================================

run_case() {
    local project_path="$1"
    local output_path="$2"
    shift 2

    if [ -z "$GEMINI_API_KEY" ]; then
        echo "Error: GEMINI_API_KEY is not set."
        echo "  export GEMINI_API_KEY=your_key"
        exit 1
    fi

    if [ ! -d "$project_path" ]; then
        echo "Warning: Project not found at $project_path — skipping."
        echo "  Clone it first (see README.md for instructions)."
        return 0
    fi

    local flags=()
    flags+=(--project "$project_path")
    flags+=(--output "$output_path")
    flags+=(--analyzer "$ANALYZER")
    flags+=(--model "$LLM_MODEL")
    flags+=(--pipeline-mode "$PIPELINE_MODE")
    flags+=(--no-reuse-db)

    if [ -n "$CPP_QUERIES_DIR" ]; then
        flags+=(--cpp-queries-dir "$CPP_QUERIES_DIR")
    fi

    if [ "$USE_LLM_VERIFY" = true ]; then
        flags+=(--use-llm-verify --llm-verify-model "$LLM_VERIFY_MODEL")
    fi

    case "$HINT_RESUME_MODE" in
        fresh) export HINT_FRESH_START=1 ;;
        *)     unset HINT_FRESH_START 2>/dev/null || true ;;
    esac

    echo ""
    echo "============================================================"
    echo "  MemHint: Analyzing $(basename "$project_path")"
    echo "  Analyzer: $ANALYZER | Model: $LLM_MODEL"
    echo "  Output:   $output_path"
    echo "============================================================"
    echo ""

    python main.py "${flags[@]}" "$@"
}

# =============================================================================
# Main
# =============================================================================

ALL_CASES=("vim" "tmux" "openssl" "redis" "freerdp" "curl" "ffmpeg")

if [ "$RUN_ALL" = true ]; then
    CASES_TO_RUN=("${ALL_CASES[@]}")
else
    CASES_TO_RUN=("${RUN_CASES[@]}")
fi

echo "MemHint Replication — Analyzer: $ANALYZER"
echo "Projects: ${CASES_TO_RUN[*]}"
echo ""

for case_name in "${CASES_TO_RUN[@]}"; do
    echo "=== [$case_name] ==="
    run_case_by_name "$case_name"
done

echo ""
echo "Replication complete. Results are in output/."
