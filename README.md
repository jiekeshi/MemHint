# MemHint

**Finding Memory Leaks in C/C++ Programs via Neuro-Symbolic Augmented Static Analysis**

MemHint is a neuro-symbolic pipeline that detects memory leaks in C/C++ projects by combining LLM-based function summary generation with Z3-based symbolic validation and industrial static analyzers (CodeQL, Infer).

## Overview

MemHint operates in three stages:

1. **Stage 1 — Summary Generation**: Parses the codebase with tree-sitter, uses Gemini 3 Flash to classify functions as allocators/deallocators (batch size 20), and validates each summary with Z3.
2. **Stage 2 — Summary-Augmented Analysis**: Injects validated summaries into CodeQL (`allocationFunctionModel`/`deallocationFunctionModel`) and Infer (`--pulse-model-alloc-pattern`/`--pulse-model-free-pattern`).
3. **Stage 3 — Warning Validation**: Filters infeasible warnings via Z3 path feasibility checking, then uses Gemini 3.1 Pro to confirm genuine bugs.

## Evaluated Projects

| Project | Version | URL |
|---------|---------|-----|
| Vim | 9.2.0015 | https://github.com/vim/vim |
| tmux | 3.6a | https://github.com/tmux/tmux |
| OpenSSL | 3.6.1 | https://github.com/openssl/openssl |
| Redis | 8.6-rc1 | https://github.com/redis/redis |
| FreeRDP | 3.23.0 | https://github.com/FreeRDP/FreeRDP |
| curl | 8.19.0-rc3 | https://github.com/curl/curl |
| FFmpeg | n8.1-dev | https://github.com/FFmpeg/FFmpeg |

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| CodeQL CLI | 2.23.9 | Static analyzer (Stage 2) |
| Infer | 1.2.0 | Static analyzer (Stage 2) |
| Z3 | 4.15.4 | Symbolic validation (Stages 1 & 3) |
| Gemini API key | — | LLM access (Stages 1 & 3) |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install static analyzers

**CodeQL** (v2.23.9):
```bash
# Download from https://github.com/github/codeql-cli-binaries/releases
# Add to PATH:
export PATH=/path/to/codeql:$PATH
```

**Infer** (v1.2.0):
```bash
# Download from https://github.com/facebook/infer/releases
# Add to PATH:
export PATH=/path/to/infer/bin:$PATH
```

### 3. Set Gemini API key

```bash
export GEMINI_API_KEY="your-api-key"
```

### 4. Clone subject projects

Clone each project at the specific version into a `subjects/` directory:

```bash
mkdir subjects && cd subjects

# Vim 9.2.0015
git clone https://github.com/vim/vim.git vim_9_2_0015
cd vim_9_2_0015 && git checkout v9.2.0015 && cd ..

# tmux 3.6a
git clone https://github.com/tmux/tmux.git tmux_3_6_a
cd tmux_3_6_a && git checkout 3.6a && cd ..

# OpenSSL 3.6.1
git clone https://github.com/openssl/openssl.git openssl_3_6_1
cd openssl_3_6_1 && git checkout openssl-3.6.1 && cd ..

# Redis 8.6-rc1
git clone https://github.com/redis/redis.git redis-8_6-rc1
cd redis-8_6-rc1 && git checkout 8.6-rc1 && cd ..

# FreeRDP 3.23.0
git clone https://github.com/FreeRDP/FreeRDP.git FreeRDP_3_23_0
cd FreeRDP_3_23_0 && git checkout 3.23.0 && cd ..

# curl 8.19.0-rc3
git clone https://github.com/curl/curl.git curl_rc-8_19_0-3
cd curl_rc-8_19_0-3 && git checkout curl-8_19_0-rc3 && cd ..

# FFmpeg n8.1-dev
git clone https://github.com/FFmpeg/FFmpeg.git FFmpeg_n_8_1_dev
cd FFmpeg_n_8_1_dev && git checkout n8.1-dev && cd ..
```

## Replication

### Run all 7 projects with CodeQL

```bash
bash run.sh
```

### Run all 7 projects with Infer

```bash
ANALYZER=infer bash run.sh
```

### Run a single project

Edit `run.sh` and set:
```bash
RUN_ALL=false
RUN_CASES=("vim")
```

Then run:
```bash
bash run.sh
```

Or run directly with `main.py`:
```bash
# With CodeQL
python main.py --project subjects/vim_9_2_0015 --output output/vim

# With Infer
python main.py --project subjects/vim_9_2_0015 --output output/vim --analyzer infer

# For Redis (source is under src/ subdirectory)
python main.py --project subjects/redis-8_6-rc1 --output output/redis --source-root subjects/redis-8_6-rc1/src
```

## Output

Each run produces the following in the output directory:

```
output/<project>/
├── hints.json                  # Validated function summaries
├── hints_all_raw.json          # Raw LLM output before Z3 validation
├── memory_safety_bugs.json     # Detected memory leaks by function
├── filtered_warnings.json      # Warnings removed by Z3 filtering
├── llm_verify_bugs.json        # LLM validation: TP vs FP per function
├── report.md                   # Human-readable report
└── memhint_YYYYMMDD_HHMMSS.log # Execution log
```

## Project Structure

```
MemHint/
├── main.py                          # CLI entry point
├── run.sh                           # Replication script for all 7 projects
├── requirements.txt                 # Python dependencies
├── proj_build_command.json          # CodeQL build commands per project
├── proj_build_command_infer.json    # Infer build commands per project
└── src/
    ├── core/
    │   ├── models.py                # Data structures (Hint, Warning, HintSet, etc.)
    │   └── pipeline.py              # Pipeline orchestrator (Stages 1-3)
    ├── analyzer/
    │   ├── adapters.py              # CodeQL & Infer integration
    │   └── queries/                 # Enhanced CodeQL .ql queries
    ├── symbolic/
    │   └── z3_solver.py             # Z3 hint validation & path feasibility
    ├── tree_sitter_parser.py        # C/C++ parsing via tree-sitter
    ├── llm_client.py                # Gemini API integration
    └── verify_bugs_llm.py           # LLM-based warning validation (Phase 6)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Gemini API key (required) |
| `HINT_BATCH_SIZE` | `20` | Functions per LLM call in Stage 1 |
| `SUBJECTS_DIR` | `subjects` | Directory containing cloned projects |
| `CPP_QUERIES_DIR` | — | Path to CodeQL cpp-queries directory |
| `ANALYZER` | `codeql` | Analyzer for `run.sh` (`codeql` or `infer`) |
