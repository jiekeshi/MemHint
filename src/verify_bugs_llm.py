#!/usr/bin/env python3
"""
Pipeline-integrated script: use Gemini to verify whether reported bugs are real (true positive).

Reads memory_safety_bugs.json from the pipeline output directory, extracts function source
for each reported bug, and asks Gemini (with configurable model) to judge true vs false positive.
Writes llm_verify_bugs_partial.json incrementally during the run; final llm_verify_bugs.json on completion.

Invoked from run.sh when USE_LLM_VERIFY_BUGS=true.

Usage (from project root):
    python src/verify_bugs_llm.py --results-dir /path/to/output_subdir --project /path/to/project --model gemini-2.5-pro
    python -m src.verify_bugs_llm --results-dir ... --project ... --model ...
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
import time
import random
logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so "from src.llm_client import ..." works
# when this script is run from any cwd (e.g. run.sh, or python src/verify_bugs_llm.py).
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def extract_function_body_by_line(file_path: Path, line_1based: int) -> tuple[int, int, str] | None:
    """Find function containing the given line (1-based) using tree-sitter.
    Returns (start_line_1based, end_line_1based, source_text). Prefer this over name
    when the bug JSON has line info, to avoid wrong function with same-name statics/macros."""
    if not file_path.is_file() or line_1based is None:
        return None
    try:
        from src.tree_sitter_parser import CodeParser
        parser = CodeParser()
        info = parser.get_function_containing_line(file_path, line_1based)
        if info is None:
            return None
        return (info.start_line, info.end_line, info.code)
    except Exception:
        return None


def extract_function_body(file_path: Path, function_name: str) -> tuple[int, int, str] | None:
    """Find function by name using tree-sitter, return (start_line_1based, end_line_1based, source_text)."""
    if not file_path.is_file():
        return None
    try:
        from src.tree_sitter_parser import CodeParser
        parser = CodeParser()
        functions = parser.parse_file(file_path)
        info = functions.get(function_name)
        if info is None:
            return None
        return (info.start_line, info.end_line, info.code)
    except Exception:
        return None


def load_bugs(data: dict) -> list[tuple[str, dict]]:
    """Yield (category, entry) for each function entry in memory_safety_bugs.json.

    Expected JSON shape:
      { "total_functions": N, "MEMORY_LEAK": [...], ... }
      Each list item: { "file", "function", "bug_count", "bugs": [ { "line", "warning_type", "type",
        "message", "allocation_site", "trace", "suggested_fix" }, ... ] }
    """
    for key, value in data.items():
        if key == "total_functions" or not isinstance(value, list):
            continue
        for entry in value:
            if isinstance(entry, dict) and entry.get("bugs"):
                yield (key, entry)


# Max trace steps to show with code; if longer, show first + last steps only
MAX_TRACE_STEPS = 10
MAX_TRACE_HEAD = 5
MAX_TRACE_TAIL = 3


def _get_trace_with_code(trace: list | None, project_root: Path) -> list[dict]:
    """For each trace location (file:line), get the code at that line so the LLM can see it.
    Returns list of {"location": "file:line", "code": snippet or None}.
    Long traces are truncated to first + last steps to limit prompt size.
    """
    if not trace or not isinstance(trace, list):
        return []
    out = []
    for loc in trace:
        if not isinstance(loc, str):
            continue
        parts = loc.rsplit(":", 1)
        if len(parts) != 2:
            out.append({"location": loc, "code": None})
            continue
        file_rel, line_str = parts[0].strip(), parts[1].strip()
        try:
            line_num = int(line_str)
        except ValueError:
            out.append({"location": loc, "code": None})
            continue
        file_path = project_root / file_rel
        # Single line only at this trace step (no extra context), so "code at that step" is one line
        code = _get_line_snippet(file_path, line_num, context_lines=2)
        out.append({"location": loc, "code": code})
    if len(out) <= MAX_TRACE_STEPS:
        return out
    head = out[:MAX_TRACE_HEAD]
    tail = out[-MAX_TRACE_TAIL:]
    return head + [{"location": f"... ({len(out)} steps total) ...", "code": None}] + tail


def _get_line_snippet(file_path: Path, line_1based: int, context_lines: int = 2) -> str | None:
    """Get the code at the given line plus context (before/after). Returns None if file/line missing."""
    if not file_path.is_file() or line_1based is None:
        return None
    try:
        file_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    if line_1based < 1 or line_1based > len(file_lines):
        return None
    start = max(0, line_1based - 1 - context_lines)
    end = min(len(file_lines), line_1based + context_lines)
    snippet_lines = file_lines[start:end]
    return "\n".join(snippet_lines)


def collect_tasks(data: dict, project_root: Path) -> list[dict]:
    """Build list of tasks: one per function with bugs; each has source + bug_lines_detail."""
    tasks = []
    for category, entry in load_bugs(data):
        file_rel = entry.get("file") or ""
        function = entry.get("function") or ""
        bugs = entry.get("bugs") or []
        if not file_rel or not function:
            continue
        src_path = project_root / file_rel
        # Prefer line-based lookup to avoid wrong function when same name exists (static/macro)
        first_line = bugs[0].get("line") if bugs else None
        extracted = None
        if first_line is not None:
            extracted = extract_function_body_by_line(src_path, first_line)
        if extracted is None:
            extracted = extract_function_body(src_path, function)
        bug_lines_detail = []
        for b in bugs:
            line_num = b.get("line")
            code_snippet = _get_line_snippet(src_path, line_num) if line_num is not None else None
            trace_with_code = _get_trace_with_code(b.get("trace"), project_root)
            bug_lines_detail.append({
                "line": line_num,
                "message": b.get("message") or "",
                "warning_type": b.get("warning_type") or b.get("type") or "",
                "allocation_site": b.get("allocation_site"),
                "trace": b.get("trace"),
                "trace_with_code": trace_with_code,
                "suggested_fix": b.get("suggested_fix"),
                "code_at_line": code_snippet,
            })
        task = {
            "category": category,
            "file": file_rel,
            "function": function,
            "bugs": bugs,
            "source": None,
            "start_line": None,
            "end_line": None,
            "bug_lines_detail": bug_lines_detail,
            "project_name": project_root.name,
        }
        if extracted:
            start_1, end_1, source = extracted
            task["source"] = source
            task["start_line"] = start_1
            task["end_line"] = end_1
        tasks.append(task)
    return tasks


def build_prompt(task: dict) -> str:
    """Build prompt for one function and its reported bugs."""
    project_name = task.get("project_name") or "unknown"
    lines = [
        "You are a memory-safety expert. Analyze the following C function and the reported bug(s).",
        "",
        f"**Project:** {project_name}",
        f"**File:** {task['file']}",
        f"**Function:** {task['function']}",
        f"**Reported category:** {task['category']}",
        "",
        f"**Reported issues (numbered 1, 2, ... for reference):**",
    ]
    for idx, d in enumerate(task["bug_lines_detail"], 1):
        line_num = d.get("line", "?")
        msg = d.get("message", "")
        lines.append(f"  {idx}. Line {line_num}: {msg}")
        if d.get("allocation_site"):
            lines.append(f"    allocation_site: {d['allocation_site']}")
        for i, step in enumerate(d.get("trace_with_code") or [], 1):
            lines.append(f"    trace step {i}: {step['location']}")
            if step.get("code"):
                lines.append("    code at that step:")
                for code_line in step["code"].split("\n"):
                    lines.append(f"      {code_line}")
        if d.get("code_at_line"):
            lines.append(f"    code at line {line_num}:")
            for code_line in d["code_at_line"].split("\n"):
                lines.append(f"      {code_line}")
    lines.append("")
    if task.get("source"):
        lines.append("**Function source:**")
        lines.append("```c")
        start_1 = task.get("start_line") or 0
        bug_line_set = {d.get("line") for d in task["bug_lines_detail"]}
        for i, sl in enumerate(task["source"].split("\n")):
            ln = start_1 + i
            marker = "  // <-- reported bug" if ln in bug_line_set else ""
            lines.append(sl + marker)
        lines.append("```")
    else:
        lines.append("(Function source could not be extracted.)")
    n_issues = len(task["bug_lines_detail"])
    bug_type_desc = task["category"].replace("_", " ").lower()
    lines.extend([
        "",
        "Role: You are a senior static-analysis engineer specializing in C/C++ memory-safety.",
        "You review memory-leak bug reports and assess whether the reported findings correspond to actual memory-leak defects in the program.",
        "",
        f"Does this function actually have a {bug_type_desc}? Determine whether the reported issue is a genuine bug (true) or a false alarm (false).",
        "",
        "Decision policy:",
        "  - true: at least one reported issue plausibly corresponds to a real memory leak based on the shown code.",
        "  - false: all reported issues are not real defects based on the shown code.",
        "",
        "If only some issues are real, output true and list only the real ones by index (1 = first, 2 = second, ...).",
        "",
        "Respond with a single JSON object, no other text. Keep reason SHORT:",
        "  {\"verdict\": true | false, \"confidence\": 0.0-1.0, \"reason\": \"one short sentence\", \"bug_indices\": [1] or [2,3] or []}",
        "",
        "Output rules:",
        "  - bug_indices: 1-based indices of reported issues you consider real; [] when verdict=false.",
        "  - reason: ONE short sentence only. Do not quote code.",
    ])
    return "\n".join(lines)


def call_gemini(prompt: str, model: str, client=None, max_retries: int = 4) -> dict:
    """Call Gemini API with simple retry/backoff.

    Returns dict with 'content' (parsed JSON) and 'usage'.
    If client is provided, reuse it; otherwise create a new LLMClient.
    """
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        return {
            "content": {
                "verdict": "ERROR",
                "_error": "GEMINI_API_KEY not set",
            },
            "usage": {},
        }

    # Create the client only on the first call; subsequent run_verify calls reuse it.
    if client is None:
        from src.llm_client import LLMClient
        client = LLMClient(api_key=api_key, model=model)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            # A read timeout from the underlying requests library will raise an exception caught below.
            return client.query(prompt)
        except Exception as e:
            last_error = e
            msg = str(e)
            logger.warning(
                "Gemini HTTP API call failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                msg,
            )

            # Retry with backoff for transient errors (timeout / 502 / 503 / 504).
            if attempt < max_retries and any(
                kw in msg for kw in ("Read timed out", "502", "503", "504", "timeout")
            ):
                # Exponential backoff + jitter: 1s, 2s, 4s, ... to avoid overwhelming the endpoint.
                sleep_s = (2 ** (attempt - 1)) + random.random()
                time.sleep(sleep_s)
                continue
            else:
                # Non-transient error or last retry attempt; stop retrying.
                break

    # All retries exhausted; return ERROR (LLM call failure).
    err_str = str(last_error) if last_error else "Unknown error"
    logger.error("LLM call failed after %d retries: %s", max_retries, err_str)
    return {
        "content": {
            "verdict": "ERROR",
            "_error": err_str,
        },
        "usage": {},
    }


def _load_llm_results_from_partial(results_dir: Path) -> dict:
    """Load existing llm_verify results from llm_verify_bugs_partial.json if present.

    Returns mapping (category, file, function) -> llm_verify_dict so we can resume
    without re-querying the LLM for already-verified functions.
    """
    partial_file = results_dir / "llm_verify_bugs_partial.json"
    if not partial_file.is_file():
        return {}
    try:
        data = json.loads(partial_file.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(
            "Failed to read existing llm_verify_bugs_partial.json from %s; ignoring resume state.",
            partial_file,
            exc_info=True,
        )
        return {}

    llm_by_key: dict[tuple[str, str, str], dict] = {}
    for section_name in ("tp", "fp", "error"):
        section = data.get(section_name) or {}
        if not isinstance(section, dict):
            continue
        for cat_key, entries in section.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                llm_info = entry.get("llm_verify")
                if not isinstance(llm_info, dict):
                    continue
                key = (cat_key, entry.get("file"), entry.get("function"))
                if any(k is None for k in key):
                    continue
                # Latest result wins if duplicates exist
                llm_by_key[key] = copy.deepcopy(llm_info)
    return llm_by_key

def run_verify(
    results_dir: Path, project_root: Path, model: str, limit: int | None = None
) -> dict:
    """Load bugs, run Gemini per function, return original bugs JSON with llm_verify added to each entry."""
    bugs_file = results_dir / "memory_safety_bugs.json"
    if not bugs_file.is_file():
        return {}
    data = json.loads(bugs_file.read_text(encoding="utf-8"))
    project_root = Path(project_root).resolve()
    if not project_root.is_absolute():
        project_root = Path.cwd() / project_root
    # Load any existing partial llm_verify results so we can resume without
    # re-verifying the same (category, file, function) entries.
    llm_by_key = _load_llm_results_from_partial(results_dir)

    tasks = collect_tasks(data, project_root)
    if llm_by_key:
        existing_keys = set(llm_by_key.keys())
        tasks = [
            t
            for t in tasks
            if (t["category"], t["file"], t["function"]) not in existing_keys
        ]
    if limit is not None:
        tasks = tasks[:limit]
    def _normalize_verdict(v) -> str:
        if v is None:
            return "FP"
        if v is True:
            return "TP"
        if v is False:
            return "FP"
        s = str(v).strip().upper()
        if s in ("TRUE", "TP", "TRUE POSITIVE", "TRUE_POSITIVE"):
            return "TP"
        if s in ("FALSE", "FP", "FALSE POSITIVE", "FALSE_POSITIVE"):
            return "FP"
        if s in ("ERROR",):
            return "ERROR"
        return "FP"

    # Create LLMClient once and reuse for all functions
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    llm_client = None
    if api_key:
        from src.llm_client import LLMClient
        llm_client = LLMClient(api_key=api_key, model=model)

    # Pre-estimate cost (chars/3 for input, fixed tokens per call for output)
    est_input_tokens = 0
    for task in tasks:
        est_input_tokens += len(build_prompt(task)) // 3
    est_output_tokens = len(tasks) * 200
    input_ppm = getattr(llm_client, "input_price_per_million", 0.0) or 0.0 if llm_client else 0.0
    output_ppm = getattr(llm_client, "output_price_per_million", 0.0) or 0.0 if llm_client else 0.0
    estimated_cost_usd = (est_input_tokens / 1_000_000) * input_ppm + (est_output_tokens / 1_000_000) * output_ppm
    logger.info(
        "LLM verify: %d functions, estimated tokens in≈%d out≈%d, estimated cost ≈ $%.4f",
        len(tasks), est_input_tokens, est_output_tokens, estimated_cost_usd,
    )

    # (category, file, function) -> llm result (one verdict per function)
    # llm_by_key may already contain entries loaded from a previous partial run.
    actual_cost_usd = 0.0
    actual_prompt_tokens = 0
    actual_completion_tokens = 0

    logger.info("LLM verify: sequential mode (no batching)")

    def _verify_one_task(task):
        """Verify a single task and return (task, result_dict, usage_dict)."""
        prompt = build_prompt(task)
        out = call_gemini(prompt, model, client=llm_client)

        if not isinstance(out, dict):
            out = {
                "content": {"verdict": "ERROR", "_error": f"Unexpected LLM result type: {type(out).__name__}"},
                "usage": {},
            }

        usage = out.get("usage") or {}
        content = out.get("content") or {}

        if isinstance(content, list):
            first = content[0] if content else None
            content = first if isinstance(first, dict) else {
                "verdict": "ERROR", "_error": "LLM returned list content without dict payload",
            }

        if not content and not (usage.get("total_tokens") or usage.get("prompt_tokens") or usage.get("completion_tokens")):
            content = {"verdict": "ERROR", "_error": "LLM call failed or returned no content (likely HTTP/quota error)"}

        verdict = _normalize_verdict(content.get("verdict"))
        confidence = content.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None
        reason = content.get("reason", "")
        raw_bug_indices = content.get("bug_indices")
        bug_indices = []
        if isinstance(raw_bug_indices, list):
            n = len(task["bug_lines_detail"])
            for x in raw_bug_indices:
                try:
                    i = int(x)
                    if 1 <= i <= n and i not in bug_indices:
                        bug_indices.append(i)
                except (TypeError, ValueError):
                    pass
            bug_indices.sort()
        elif raw_bug_indices is not None:
            try:
                i = int(raw_bug_indices)
                if 1 <= i <= len(task["bug_lines_detail"]):
                    bug_indices = [i]
            except (TypeError, ValueError):
                pass
        should_keep = verdict not in ("FP",)
        err_msg = content.get("_error")
        if verdict == "ERROR" and err_msg:
            logger.error("LLM verify ERROR for %s:%s (%s): %s", task["file"], task["function"], task["category"], err_msg)

        result = {
            "verdict": verdict, "confidence": confidence, "reason": reason,
            "bug_indices": bug_indices, "should_keep": should_keep, "error": err_msg,
        }
        return task, result, usage

    # Process tasks sequentially (one LLM call at a time)
    for task in tasks:
        task, result, usage = _verify_one_task(task)
        actual_cost_usd += float(usage.get("cost", 0) or 0)
        actual_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        actual_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        key = (task["category"], task["file"], task["function"])
        llm_by_key[key] = result

        # Write partial JSON after each processed task
        out = _build_verify_output(data, llm_by_key)
        partial_file = results_dir / "llm_verify_bugs_partial.json"
        partial_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Check that every function was verified (catch LLM connection/API skips)
    out = _build_verify_output(data, llm_by_key)
    out["summary"]["estimated_cost_usd"] = round(estimated_cost_usd, 4)
    out["summary"]["estimated_prompt_tokens"] = est_input_tokens
    out["summary"]["estimated_completion_tokens"] = est_output_tokens
    out["summary"]["actual_cost_usd"] = round(actual_cost_usd, 4)
    out["summary"]["actual_prompt_tokens"] = actual_prompt_tokens
    out["summary"]["actual_completion_tokens"] = actual_completion_tokens
    logger.info(
        "LLM verify done: actual cost $%.4f (prompt %d + completion %d tokens)",
        actual_cost_usd, actual_prompt_tokens, actual_completion_tokens,
    )

    expected_keys = set()
    for cat, entry in load_bugs(data):
        expected_keys.add((cat, entry.get("file"), entry.get("function")))
    missing = [k for k in expected_keys if k not in llm_by_key]
    if missing:
        logger.warning(
            "LLM verify incomplete: %d function(s) not verified (e.g. connection/API error): %s",
            len(missing),
            [(f, fn) for (_, f, fn) in missing],
        )
        out["summary"]["total_functions_expected"] = len(expected_keys)
        out["summary"]["missing_count"] = len(missing)
        out["summary"]["missing_functions"] = [
            {"file": f, "function": fn, "category": cat} for (cat, f, fn) in missing
        ]
    return out


def _build_verify_output(data: dict, llm_by_key: dict) -> dict:
    """Build { summary, tp, fp, error } from original data + llm results."""
    tp_by_cat = {}
    fp_by_cat = {}
    error_by_cat = {}
    count_tp = count_fp = count_error = 0
    for cat_key, value in data.items():
        if cat_key == "total_functions" or not isinstance(value, list):
            continue
        if cat_key not in tp_by_cat:
            tp_by_cat[cat_key] = []
            fp_by_cat[cat_key] = []
            error_by_cat[cat_key] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            k = (cat_key, entry.get("file"), entry.get("function"))
            if k not in llm_by_key:
                continue
            entry = copy.deepcopy(entry)
            entry["llm_verify"] = llm_by_key[k]
            v = llm_by_key[k]["verdict"]
            if v == "TP":
                tp_by_cat[cat_key].append(entry)
                count_tp += 1
            elif v == "ERROR":
                error_by_cat[cat_key].append(entry)
                count_error += 1
            else:
                fp_by_cat[cat_key].append(entry)
                count_fp += 1
    return {
        "summary": {
            "total_functions_verified": count_tp + count_fp + count_error,
            "tp": count_tp,
            "fp": count_fp,
            "error": count_error,
            "should_keep": count_tp + count_error,
        },
        "tp": tp_by_cat,
        "fp": fp_by_cat,
        "error": error_by_cat,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify reported bugs with Gemini (pipeline step when USE_LLM_VERIFY_BUGS=true)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing memory_safety_bugs.json; llm_verify_bugs.json will be written here.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Project source root (for resolving file paths in bugs JSON).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-pro-preview",  # Gemini 3.1 Pro for warning validation
        help="Gemini model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N functions (for testing).",
    )
    parser.add_argument(
        "--print-first-prompt",
        action="store_true",
        help="Load first bug/function, build its prompt, print to stdout and exit (no Gemini call).",
    )
    parser.add_argument(
        "--print-prompt-for-function",
        type=str,
        metavar="NAME",
        help="Build and print prompt for the first task with this function name (e.g. apply_autocmds_group), then exit.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not (results_dir / "memory_safety_bugs.json").is_file():
        print(f"No memory_safety_bugs.json in {results_dir}", file=sys.stderr)
        sys.exit(0)

    if args.print_first_prompt or args.print_prompt_for_function:
        data = json.loads((results_dir / "memory_safety_bugs.json").read_text(encoding="utf-8"))
        project_root = Path(args.project).resolve()
        if not project_root.is_absolute():
            project_root = Path.cwd() / project_root
        tasks = collect_tasks(data, project_root)
        if not tasks:
            print("No tasks (no bugs or no valid entries).", file=sys.stderr)
            sys.exit(1)
        if args.print_prompt_for_function:
            name = args.print_prompt_for_function.strip()
            task = next((t for t in tasks if t.get("function") == name), None)
            if task is None:
                print(f"No task with function name '{name}'.", file=sys.stderr)
                sys.exit(1)
        else:
            task = tasks[0]
        print(build_prompt(task))
        sys.exit(0)

    out_data = run_verify(results_dir, args.project, args.model, limit=args.limit)
    if not out_data:
        sys.exit(0)
    out_file = results_dir / "llm_verify_bugs.json"
    out_file.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    s = out_data.get("summary", {})
    print(f"Wrote {out_file}: TP={s.get('tp', 0)}, FP={s.get('fp', 0)} (kept={s.get('should_keep', 0)})", file=sys.stderr)


if __name__ == "__main__":
    main()
