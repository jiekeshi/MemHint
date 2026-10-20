# MemHint

Code and data for **Finding Memory Leaks in C/C++ Programs via Neuro-Symbolic Augmented Static Analysis** (Huihui Huang, Jieke Shi, Bo Wang, Zhou Yang, David Lo; ACSAC 2026).

MemHint is a neuro-symbolic pipeline for finding memory leaks in C/C++. An LLM infers which project-specific functions allocate or free memory, Z3 checks that each inferred allocation or deallocation is reachable, the validated summaries are injected into CodeQL and Infer, and Z3 and a second LLM pass filter the analyzers' warnings. On eight open-source projects it found the 54 memory leaks listed in [BUGS.md](BUGS.md).

## What is in this repository

| Path | Purpose |
|---|---|
| `main.py` | Pipeline CLI (`--start-from` resumes at Phase 4, 5 or 6) |
| `src/` | The tool: parsing, LLM client, Z3 validation and filtering, analyzer integration |
| `proj_build_command_codeql.json`, `proj_build_command_infer.json` | How each subject is configured and built for each analyzer |
| `experiment/` | `fetch_subjects.sh` (checks out the subjects at the paper's commits), `run.sh` (runs the pipeline), `projects.md` (subject versions, commits, and what the builds compile) |
| `BUGS.md` | The 54 bugs of Table 1 with their upstream reports |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | Python environment, pinned to the versions the measurements were taken with |

The validated summaries the paper's runs produced are not shipped; the pipeline regenerates them (Phases 1 to 3), and any run's `hints.json` can be fed back in with `PHASES=456` (see [Running some phases only](#running-some-phases-only)).

## Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| Host | x86-64 Linux | the paper's measurements were taken on Ubuntu 22.04 |
| CPU cores | 4 | 8 |
| RAM | 16 GB | 32 GB |
| Free disk | 40 GB | 80 GB |

No GPU is required. The paper's measurements were taken with CodeQL 2.25.1 (query packs `codeql/cpp-queries` 1.5.8, `codeql/cpp-all` 6.1.4), Infer 1.2.0, Z3 4.15.4, clang-17 and the locked Python environment.

## Installation

```bash
git clone https://github.com/jiekeshi/MemHint.git
cd MemHint
./experiment/fetch_subjects.sh                           # the eight subjects, into experiment/subjects/
```

Then install, on `PATH`:

- CodeQL 2.25.1: the `codeql-bundle-linux64.tar.gz` of <https://github.com/github/codeql-action/releases/tag/codeql-bundle-v2.25.1>; set `CPP_QUERIES_DIR` to its `qlpacks/codeql/cpp-queries` directory.
- Infer 1.2.0: `infer-linux-x86_64-v1.2.0.tar.xz` of <https://github.com/facebook/infer/releases/tag/v1.2.0>.
- clang-17 (lld-17, llvm-17), from <https://apt.llvm.org>.
- The Python environment, on Python 3.11 or newer: `uv sync` (uses `uv.lock`), or `pip install -r requirements.txt`. `run.sh` uses uv when it is on `PATH` and `python3` otherwise.
- The subjects' build dependencies. On Ubuntu 22.04:

```bash
sudo apt-get install -y --no-install-recommends \
    build-essential make cmake pkg-config git ca-certificates curl xz-utils \
    autoconf automake libtool bison flex gettext python3 python3-distutils perl \
    libncurses-dev libevent-dev libssl-dev zlib1g-dev libpsl-dev libnghttp2-dev \
    libzstd-dev nasm libbz2-dev liblzma-dev libavcodec-dev libavformat-dev \
    libavutil-dev libswscale-dev libswresample-dev libkrb5-dev libasound2-dev \
    libwayland-dev libxkbcommon-dev libsystemd-dev libcjson-dev libusb-1.0-0-dev \
    uuid-dev libx11-dev libxext-dev libxinerama-dev libxcursor-dev libxi-dev \
    libxrandr-dev libxv-dev libxkbfile-dev libcairo2-dev libpcsclite-dev libicu-dev \
    libjpeg-dev libpng-dev libsdl2-dev libpam0g-dev libcups2-dev libsdl2-ttf-dev \
    libfuse3-dev clang lld llvm bc libelf-dev rsync kmod cpio
```

Check the tool versions:

```bash
codeql version --format=terse && infer --version | head -1
```

Expected: `2.25.1` and `Infer version v1.2.0-4c53e80cac`.

`fetch_subjects.sh` checks each project out at the commit the paper analysed (`experiment/projects.md`). The Linux kernel is the largest download; only `drivers/staging` is analysed.

## API key

Phases 2 and 6 call a model. Put one of these in a `.env` file in the repository root (or export it):

```dotenv
GEMINI_API_KEY=your-key
```

```dotenv
OPENROUTER_API_KEY=your-key
```

The paper's runs used Gemini 3 Flash for Phase 2 (20 functions per call) and Gemini 3.1 Pro for Phase 6 (one function per call); both are the defaults. Phases 1, 3, 4 and 5 need no key.

## Quick start

After [installation](#installation):

```bash
./experiment/fetch_subjects.sh tmux_3_6_a
echo 'GEMINI_API_KEY=your-key' > .env                    # or OPENROUTER_API_KEY
./experiment/run.sh tmux_3_6_a                           # all six phases on tmux, about 30 minutes
```

Results land in `experiment/output/tmux_3_6_a/pipeline_output/`, one file per phase (see [What a run reports](#what-a-run-reports)). `./experiment/run.sh` with no arguments runs all eight projects (about 28 hours); `PHASES=45` runs only the phases that need no API key.

## Running

Everything goes through `experiment/run.sh`; results go to `experiment/output/` (`OUT_ROOT` changes this).

```bash
./experiment/run.sh                        # all eight projects, both analyzers, all six phases
./experiment/run.sh tmux_3_6_a             # one project, both analyzers
./experiment/run.sh tmux_3_6_a infer       # one project, one analyzer
```

The whole benchmark takes about 28 hours and roughly $100 of LLM calls (Table 6 gives the per-project cost and time). tmux is the fastest subject, about half an hour, and carries ten of the paper's bugs.

### Running some phases only

`PHASES` selects what runs; each later phase reads what the earlier ones left in the output directory.

| `PHASES` | Runs | Needs |
|---|---|---|
| `123456` (default) | Everything: parse, LLM summaries, Z3 validation, analyzer, Z3 filter, LLM validation | API key |
| `3456` | Z3 validation, analyzer, Z3 filter, LLM validation, from phase 2 summaries you supply | API key, `SUMMARIES=` or `SUMMARIES_DIR=` (unvalidated) |
| `345` | Z3 validation, analyzer and Z3 filter only | `SUMMARIES=` or `SUMMARIES_DIR=` (unvalidated); no key |
| `456` | Analyzer, Z3 filter, LLM validation, from summaries you supply | API key, `SUMMARIES=` or `SUMMARIES_DIR=` |
| `45` | Analyzer and Z3 filter only | `SUMMARIES=` or `SUMMARIES_DIR=`; no key |
| `5` | Redo the Z3 filter on warnings already in the output directory | nothing |
| `6` | Redo the LLM validation on the Z3 survivors already there | API key |

```bash
# Phases 4-5 on tmux from the summaries an earlier run wrote; no API key needed
PHASES=45 SUMMARIES=experiment/output/tmux_3_6_a/pipeline_output/hints.json \
    ./experiment/run.sh tmux_3_6_a

# rerun only the Z3 filter after changing src/symbolic/z3_solver.py
PHASES=5 ./experiment/run.sh tmux_3_6_a
```

`SUMMARIES` names one project's `hints.json`; `SUMMARIES_DIR` a directory of `<project>.json` files when running several projects. Stage 1 output of any run (`pipeline_output/hints.json`) is in that format. For `345` and `3456` the summaries are the ones phase 2 wrote before validation, which a run keeps as `pipeline_output/hints_unvalidated.json`.

Other variables `run.sh` reads:

| Variable | Default | Meaning |
|---|---|---|
| `SUBJECTS` | `experiment/subjects` | Subject source directories |
| `OUT_ROOT` | `experiment/output` | Output root; one subdirectory per project. Use a distinct one to keep previous runs |
| `HINT_MODEL` | `gemini-3-flash-preview` | Phase 2 summary-generation model |
| `VERIFY_MODEL` | `gemini-3.1-pro-preview` | Phase 6 warning-validation model |
| `HINT_BATCH_SIZE` | `20` | Functions per Phase 2 call |
| `CPP_QUERIES_DIR` | the `qlpacks/codeql/cpp-queries` directory next to `codeql` | CodeQL C/C++ query packs |
| `RUNNER` | `uv run --frozen python` if uv is found, else `python3` | How Python is invoked |

Summaries are regenerated by every full run: Phase 2 asks a model to classify each candidate function and Phase 6 asks a model for each verdict, so the per-project counts move between runs. Phases 4 and 5 are deterministic given the same summaries. The paper's tables are the reference, not a value to match exactly.

## What a run reports

Each project produces `<out_root>/<project>/pipeline_output/`:

| File | Phase | Contents |
|---|---|---|
| `hints_unvalidated.json` | 2 | Summaries as the LLM wrote them |
| `hints.json` | 3 | Summaries Z3 kept; these are injected |
| `<analyzer>/validation_conflicts.txt` | 3 | Summaries Z3 rejected, with the reason |
| `<analyzer>/<analyzer>_warnings.json` | 4 | Raw analyzer warnings |
| `<analyzer>/memory_safety_bugs.json` | 5 | Warnings Z3 kept, grouped by function |
| `<analyzer>/filtered_warnings.json` | 5 | Warnings Z3 found no feasible leak path for, with the reason |
| `<analyzer>/unconfirmed_warnings.json` | 5 | Warnings that could not be encoded, dropped before Phase 6 |
| `<analyzer>/llm_verify_bugs.json` | 6 | LLM verdicts and reasons |
| `<analyzer>/pipeline_stats.json` | — | Per-phase counts, times, token counts and costs |
| `<analyzer>/report.md`, `<analyzer>/memhint_*.log` | — | The pipeline's report and log |

The paper's 54 bugs are listed in `BUGS.md`.

## Citation

```bibtex
@inproceedings{memhint2026,
  title     = {Finding Memory Leaks in C/C++ Programs via Neuro-Symbolic Augmented Static Analysis},
  author    = {Huang, Huihui and Shi, Jieke and Wang, Bo and Yang, Zhou and Lo, David},
  booktitle = {Annual Computer Security Applications Conference (ACSAC)},
  year      = {2026}
}
```
