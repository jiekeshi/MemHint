#!/usr/bin/env bash
# Runs the MemHint pipeline on the subjects. See ../README.md.
#
#   ./run.sh                      every project, both analyzers
#   ./run.sh tmux_3_6_a           one project
#   ./run.sh tmux_3_6_a codeql    one cell
#
# PHASES picks the range. 123456, the default, runs the whole pipeline: the LLM
# classifies the project's functions, Z3 validates the summaries, both analyzers
# run with them injected, and Z3 and a second LLM pass filter the warnings.
# 456 starts from summaries you supply (SUMMARIES=/path/to/hints.json, or
# SUMMARIES_DIR=/dir holding <project>.json); 45 is the same without the
# phase 6 LLM pass, so it needs no credential; 5 redoes the Z3 filter on
# warnings already on disk; 6 redoes the LLM validation. 3456 and 345 take the
# phase 2 summaries before validation instead (a run's hints_unvalidated.json)
# and redo the Z3 validation of phase 3 first.
#
# Requirements: CodeQL 2.25.1 and Infer
# 1.2.0 on PATH, the Python environment (uv, or pip -r requirements.txt), and
# the subject sources under $SUBJECTS at the commits projects.md names
# (fetch_subjects.sh checks them out).
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

if [ ! -f "$REPO/main.py" ]; then
    echo "!! No MemHint checkout at $REPO: mount the repository with"
    echo "     -v /path/to/MemHint:/memhint"
    exit 1
fi

# Subjects: a directory mounted at /subjects (e.g., in a container) wins;
# otherwise experiment/subjects inside the checkout.
if [ -z "${SUBJECTS:-}" ]; then
    if [ -d /subjects ] && [ -n "$(ls -A /subjects 2>/dev/null)" ]; then
        SUBJECTS=/subjects
    else
        SUBJECTS="$HERE/subjects"
    fi
fi
OUT_ROOT="${OUT_ROOT:-$HERE/output}"

# Phase 2 batches 20 functions per call, as the paper reports. main.py's default
# is the same; the variable only makes the choice visible here.
export HINT_BATCH_SIZE="${HINT_BATCH_SIZE:-20}"
PHASES="${PHASES:-123456}"
regenerate_summaries=0
case "$PHASES" in
    345)    start_from=3; extra="--no-llm-verify" ;;
    3456)   start_from=3; extra="" ;;
    45)     start_from=4; extra="--no-llm-verify" ;;
    5)      start_from=5; extra="--no-llm-verify" ;;
    6)      start_from=6; extra="" ;;
    456)    start_from=4; extra="" ;;
    123456) start_from=1; extra=""; regenerate_summaries=1 ;;
    *)      echo "!! PHASES must be 5, 45, 6, 456, 345, 3456 or 123456"; exit 1 ;;
esac
HINT_MODEL="${HINT_MODEL:-gemini-3-flash-preview}"
VERIFY_MODEL="${VERIFY_MODEL:-gemini-3.1-pro-preview}"

# Python: uv with the locked environment when available, else whatever python3
# has the requirements installed. RUNNER overrides both.
if [ -z "${RUNNER:-}" ]; then
    if command -v "${UV:-uv}" >/dev/null 2>&1; then
        RUNNER="${UV:-uv} run --frozen python"
    else
        RUNNER="python3"
    fi
fi

# Fail here rather than three hours in: the pipeline catches an analyzer that
# will not start, logs it, and finishes reporting zero bugs, which reads like a
# result.
missing=0
for tool in codeql infer; do
    if command -v "$tool" >/dev/null; then
        printf '%-8s %s\n' "$tool" "$(command -v "$tool")"
    else
        echo "!! $tool is not on PATH (the paper's runs used CodeQL 2.25.1 and Infer 1.2.0)"
        missing=1
    fi
done
if [ "$PHASES" != "45" ] && [ "$PHASES" != "5" ] && [ "$PHASES" != "345" ] && [ -z "${GEMINI_API_KEY:-}${OPENROUTER_API_KEY:-}" ]; then
    echo "!! no LLM credential: set GEMINI_API_KEY or OPENROUTER_API_KEY"
    echo "   (phases 2 and 6 need one; 345, 45 and 5 do not)"
    missing=1
fi
if [ "$regenerate_summaries" = 0 ] && [ "$PHASES" != "5" ] && [ "$PHASES" != "6" ] \
   && [ -z "${SUMMARIES:-}" ] && [ -z "${SUMMARIES_DIR:-}" ]; then
    echo "!! PHASES=$PHASES starts from existing summaries, but none were given."
    echo "   Set SUMMARIES=/path/to/hints.json (one project) or"
    echo "   SUMMARIES_DIR=/dir containing <project>.json, or run PHASES=123456."
    missing=1
fi

# The leak queries extend CodeQL's own cpp-queries pack, which ships beside the
# binary. Without this the run builds a database, finds the pack missing, and
# reports zero warnings.
if [ -z "${CPP_QUERIES_DIR:-}" ] && command -v codeql >/dev/null; then
    CPP_QUERIES_DIR="$(dirname "$(readlink -f "$(command -v codeql)")")/qlpacks/codeql/cpp-queries"
fi
if [ -d "${CPP_QUERIES_DIR:-}" ]; then
    printf '%-8s %s\n' "queries" "$CPP_QUERIES_DIR"
else
    echo "!! CodeQL's cpp-queries pack not found; set CPP_QUERIES_DIR=/path/to/qlpacks/codeql/cpp-queries"
    missing=1
fi
[ "$missing" -eq 0 ] || exit 1
printf '%-8s %s\n' "python" "$RUNNER"
printf '%-8s %s\n' "subjects" "$SUBJECTS"
echo

ALL_PROJECTS=(vim_9_2_0015 tmux_3_6_a openssl_3_6_1 redis-8_6-rc1
              FreeRDP_3_23_0 curl_rc-8_19_0-3 FFmpeg_n_8_1_dev linux_v7_0-rc1)

# linux/staging is analysed as a subsystem, not as the whole kernel.
source_root_for() {
    case "$1" in
        linux_v7_0-rc1) echo "drivers/staging" ;;
        *) echo "" ;;
    esac
}
summaries_for() {
    if [ -n "${SUMMARIES:-}" ]; then echo "$SUMMARIES"; else echo "${SUMMARIES_DIR:-}/$1.json"; fi
}

projects=("${@:1:1}")
[ $# -eq 0 ] && projects=("${ALL_PROJECTS[@]}")
analyzers=(codeql infer)
[ $# -ge 2 ] && analyzers=("$2")

for project in "${projects[@]}"; do
    src="$SUBJECTS/$project"
    if [ ! -d "$src" ]; then
        echo "!! $project: no source at $src; run experiment/fetch_subjects.sh $project"
        continue
    fi
    sub="$(source_root_for "$project")"

    for analyzer in "${analyzers[@]}"; do
        out="$OUT_ROOT/$project/pipeline_output/$analyzer"
        mkdir -p "$out"
        # The pipeline reads the summaries from the parent of --output, and both
        # analyzers use the same ones: that they are analyzer-agnostic is part of
        # what is being reproduced.
        if [ "$PHASES" != "6" ] && [ "$PHASES" != "5" ]; then
            # Start each cell from nothing, so a previous failure cannot be
            # mistaken for this run's result.
            rm -rf "$out"; mkdir -p "$out"
            if [ "$regenerate_summaries" = 0 ]; then
                summaries="$(summaries_for "$project")"
                if [ ! -f "$summaries" ]; then
                    echo "!! $project: no summaries at $summaries"; continue
                fi
                if [ "$start_from" = 3 ]; then
                    cp "$summaries" "$(dirname "$out")/hints_unvalidated.json"
                else
                    cp "$summaries" "$(dirname "$out")/hints.json"
                fi
            fi
            # Infer leaves infer-out behind, and a stale one makes the next run
            # read results it did not produce.
            rm -rf "$src/infer-out" "$src/.codeql-db"
        fi

        # Phase 6 resumes from llm_verify_bugs_partial.json, which is what a
        # crashed run needs and what a rerun must not have.
        [ "$PHASES" = "6" ] && rm -f "$out/llm_verify_bugs_partial.json"

        echo "=== $project / $analyzer"
        started=$(date +%s)
        ( cd "$REPO" && $RUNNER main.py \
            --project "$src" \
            --output "$out" \
            --analyzer "$analyzer" \
            --start-from "$start_from" $extra \
            --cpp-queries-dir "$CPP_QUERIES_DIR" \
            --model "$HINT_MODEL" \
            --llm-verify-model "$VERIFY_MODEL" \
            ${sub:+--source-root "$src/$sub"} \
        ) > "$OUT_ROOT/$project/$analyzer.log" 2>&1
        status=$?
        elapsed=$(( ($(date +%s) - started) / 60 ))

        # A caught analyzer failure still exits 0 and writes an empty warning
        # file, so check the file rather than the exit code.
        if [ "$PHASES" = "6" ]; then
            check="$out/llm_verify_bugs.json"; unit="verdicts"
            count=$(grep -o '"verdict"' "$check" 2>/dev/null | wc -l)
        else
            check="$out/${analyzer}_warnings.json"; unit="warnings"
            count=$(grep -o '"issue_type"' "$check" 2>/dev/null | wc -l)
        fi
        if [ "$status" -ne 0 ] || [ "$count" -eq 0 ]; then
            echo "    FAILED after ${elapsed} min: $count $unit, exit=$status"
            echo "    see $OUT_ROOT/$project/$analyzer.log"
        else
            echo "    $count $unit in ${elapsed} min"
        fi
    done

done

echo
echo "Results are under $OUT_ROOT/<project>/pipeline_output."
