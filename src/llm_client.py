"""LLM-based Memory Safety Hint Generator.

This module generates memory safety hints (NOT bugs) using LLM.
Hints describe function SEMANTICS that help CodeQL understand custom functions.

Hint Types:
- ALLOCATOR: Function returns newly allocated heap memory
- DEALLOCATOR: Function frees memory passed as argument
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional, Callable
from shutil import which
import requests

from src.core.models import FunctionInfo, Hint, HintType, HintSet, CustomQuery

# Import templates from adapters to validate filters in the same structure as production
try:
    from src.analyzer.adapters import (
        ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE,
        ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE,
    )
except ImportError:
    ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE = ""
    ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE = ""

logger = logging.getLogger(__name__)



BATCH_HINT_GENERATION_PROMPT = """You are a memory safety expert analyzing C/C++ code to identify function semantics that help static analyzers detect memory leaks.

## Task
You will be given **one or more C/C++ functions** in a single request.
For **each function independently**, determine whether it is a memory allocator, deallocator, or neither.

The input will contain a block for each function, in this form:

```text
### Function: <func_name>
Return type: <return_type>
Parameters: <parameters>

```c
<code>
```

<optional context and Z3 feedback>
```

All function blocks will be concatenated into `{functions_block}`.

## Semantic Categories

### Allocator
Function returns **newly allocated heap memory** that caller must eventually free.

**Positive indicators:**
- Calls malloc/calloc/realloc/aligned_alloc/new/new[] and returns the result
- Calls another known allocator (e.g., g_malloc, xmalloc, kmalloc) and returns result
- Returns result of a wrapper function that allocates

**Negative indicators (NOT an allocator):**
- Returns pointer to static/global buffer
- Returns pointer to struct field or array member
- Returns one of the input arguments
- Allocates internally but doesn't return the allocated memory
- Returns stack-allocated memory (dangling pointer bug, but not allocator semantic)

### Deallocator
Function **frees/releases memory** passed as an argument.

**Positive indicators:**
- Calls free/delete/delete[]/g_free/kfree on an argument
- Calls another deallocator on an argument
- Wrapper around resource cleanup

## Analysis Guidelines

1. **Trace data flow:** Follow where return values come from and where arguments flow to.
2. **Consider all paths:** Check all branches and return statements.
3. **Indirect calls matter:** If function calls helper that allocates/frees, propagate that semantic.
4. **Be precise:** Only report semantics you can verify from the code.

## Output Format

Return a single JSON object whose **top-level keys are function names**.
For each function name, the value is an object with a `hints` array.

Each hint is a function summary with:
- `name`: the function name
- `role`: "Allocator" or "Deallocator"
- `target`: "return" for allocators (return value carries heap ownership), or "argN" for deallocators (the N-th argument is freed, 0-indexed)

### Example with two functions `foo` and `bar`

```json
{{
  "foo": {{
    "hints": [
      {{"name": "foo", "role": "Allocator", "target": "return"}},
      {{"name": "foo", "role": "Deallocator", "target": "arg0"}}
    ]
  }},
  "bar": {{
    "hints": []
  }}
}}
```

If no memory semantics apply to a function, still include that function name with `"hints": []`.

Now analyze **all functions in `{functions_block}`** and return the JSON result."""





def _first_callees(func, n: int) -> list[str]:
    """The first n direct callees in the order they are first called in the
    body (func.callees is a set, so its own order is arbitrary)."""
    code = getattr(func, "code", "") or ""
    def first_call(name: str) -> int:
        m = re.search(rf"\b{re.escape(name)}\s*\(", code)
        return m.start() if m else len(code)
    return sorted(func.callees or (), key=lambda c: (first_call(c), c))[:n]

def _thinking_budget(model_name: str) -> int:
    """Billed deliberation to allow, in tokens; switched off, phase 2 loses
    deallocators and the Pro models return nothing at all."""
    override = os.getenv("HINT_THINKING_BUDGET")
    if override is not None:
        return int(override)
    return 1024 if "flash" in (model_name or "").lower() else 128


class LLMClient:
    """Gemini client, over OpenRouter when its key is set and the public
    Generative Language HTTP API otherwise."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-3-flash-preview",
        base_url: str = None,
        max_retries: int = 3,
    ):
        """Initialize the client, preferring OpenRouter: Gemini caps
        gemini-3.1-pro at 250 requests a day below tier 2, which phase 6
        exhausts in a single project."""
        self.model_name = self._canonicalize_model_name(model or "gemini-3-flash-preview")
        self.max_retries = max_retries
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GENAI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.use_api_key = bool(self.api_key)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_compat_base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.use_openrouter = bool(self.openrouter_api_key)

        self.input_price_per_million, self.output_price_per_million = (
            self._resolve_pricing_for_model(self.model_name)
        )

        if self.use_openrouter:
            logger.info("LLMClient: using OpenRouter (model=%s).", self.model_name)
        elif self.use_api_key:
            logger.info("LLMClient: using the Gemini HTTP API (model=%s).", self.model_name)
        else:
            raise ValueError(
                "No credentials. Set OPENROUTER_API_KEY, or one of "
                "GEMINI_API_KEY / GOOGLE_GENAI_API_KEY / GOOGLE_API_KEY."
            )

    def query(self, prompt: str) -> dict:
        """Send the prompt over whichever backend the credentials selected."""
        if self.use_openrouter:
            result = self._call_by_openrouter(prompt)
        else:
            result = self._call_by_gemini_api(prompt)
        if not result:
            return {"content": {}, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}}
        
        content, usage = result
        if not content:
            return {"content": {}, "usage": usage}
        try:
            parsed = json.loads(content)
            return {"content": parsed, "usage": usage}
        except json.JSONDecodeError:
            cleaned = content.strip()
            if "Ġ" in cleaned or "Ċ" in cleaned:
                cleaned = cleaned.replace("Ġ", " ").replace("Ċ", "\n")
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
                cleaned = re.sub(r"\s*```\s*$", "", cleaned)
            try:
                return {"content": json.loads(cleaned), "usage": usage}
            except json.JSONDecodeError:
                pass
            for lo, hi in (("{", "}"), ("[", "]")):
                start, end = cleaned.find(lo), cleaned.rfind(hi)
                if 0 <= start < end:
                    try:
                        return {"content": json.loads(cleaned[start:end + 1]), "usage": usage}
                    except json.JSONDecodeError:
                        pass
            for lo, hi in (("{", "}"), ("[", "]")):
                end = cleaned.rfind(hi)
                if end < 0:
                    continue
                start = cleaned.rfind(lo, 0, end)
                while start >= 0:
                    try:
                        return {"content": json.loads(cleaned[start:end + 1]), "usage": usage}
                    except json.JSONDecodeError:
                        start = cleaned.rfind(lo, 0, start)
            logger.warning("LLM response was not valid JSON. Raw content: %s", content[:200])
            return {"content": {}, "usage": usage}

    def _resolve_pricing_for_model(self, model_name: str) -> tuple[float, float]:
        """Public list prices per million tokens as of February 2026;
        gemini-3.1-pro's holds to prompts of 200k tokens, and
        HINT_LLM_{INPUT,OUTPUT}_PRICE_PER_M override whatever is chosen."""
        name = (model_name or "").lower()

        input_price = 2.00
        output_price = 12.00
        if "gemini-3" in name and "flash" in name:
            input_price = 0.50
            output_price = 3.00

        try:
            env_in = os.getenv("HINT_LLM_INPUT_PRICE_PER_M")
            if env_in:
                input_price = float(env_in)
        except Exception:
            pass

        try:
            env_out = os.getenv("HINT_LLM_OUTPUT_PRICE_PER_M")
            if env_out:
                output_price = float(env_out)
        except Exception:
            pass

        logger.info(
            "LLM pricing for model '%s': input=$%.4f/M, output=$%.4f/M",
            model_name,
            input_price,
            output_price,
        )
        return input_price, output_price

    def _canonicalize_model_name(self, model: str) -> str:
        """Map user-friendly model aliases to the exact IDs expected by the
        public Gemini HTTP API (generativelanguage.googleapis.com)."""
        name = (model or "").strip()
        lower = name.lower()

        if lower in ("gemini-3.0-flash", "gemini-3-flash", "gemini-3.0-flash-preview"):
            return "gemini-3-flash-preview"

        return name

    def _call_by_openrouter(self, prompt: str) -> tuple[str | None, dict] | None:
        """Call OpenRouter's OpenAI-compatible chat-completions endpoint."""
        url = self.openai_compat_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not os.getenv("HINT_DISABLE_JSON_MODE"):
            body["response_format"] = {"type": "json_object"}
        body["reasoning"] = {"max_tokens": _thinking_budget(self.model_name)}

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=300)
                if resp.status_code >= 400:
                    logger.warning(
                        "OpenRouter call failed (attempt %d): %s %s",
                        attempt + 1, resp.status_code, resp.text[:200],
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                    continue

                data = resp.json()
                choices = data.get("choices") or []
                response_text = (choices[0].get("message", {}) or {}).get("content") if choices else None

                usage_meta = data.get("usage", {}) or {}
                prompt_tokens = int(usage_meta.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage_meta.get("completion_tokens", 0) or 0)
                total_tokens = int(usage_meta.get("total_tokens", prompt_tokens + completion_tokens) or 0)

                input_cost = (prompt_tokens / 1_000_000) * self.input_price_per_million
                output_cost = (completion_tokens / 1_000_000) * self.output_price_per_million
                total_cost = input_cost + output_cost

                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": total_cost,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                }
                return (response_text, usage)
            except Exception as exc:
                logger.warning("OpenRouter call failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        return (None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "input_cost": 0.0, "output_cost": 0.0})

    def _call_by_gemini_api(self, prompt: str) -> tuple[str | None, dict] | None:
        """Call Gemini over the HTTP API, counting thinking tokens as output
        because Gemini reports them apart from the answer."""
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        headers = {"Content-Type": "application/json"}

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": _thinking_budget(self.model_name)},
            },
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    params={"key": self.api_key},
                    json=body,
                    timeout=300,
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Gemini HTTP API call failed (attempt %d): %s %s",
                        attempt + 1,
                        resp.status_code,
                        resp.text[:200],
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                    continue

                data = resp.json()

                text_chunks: list[str] = []
                for cand in data.get("candidates", []):
                    content = cand.get("content", {})
                    for part in content.get("parts", []):
                        t = part.get("text")
                        if isinstance(t, str):
                            text_chunks.append(t)
                response_text = "".join(text_chunks) if text_chunks else None

                usage_meta = data.get("usageMetadata", {}) or {}
                prompt_tokens = int(usage_meta.get("promptTokenCount", 0) or 0)
                completion_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0) + int(
                    usage_meta.get("thoughtsTokenCount", 0) or 0
                )
                total_tokens = int(usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens) or 0)

                input_cost = (prompt_tokens / 1_000_000) * self.input_price_per_million
                output_cost = (completion_tokens / 1_000_000) * self.output_price_per_million
                total_cost = input_cost + output_cost

                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": total_cost,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                }

                return (response_text, usage)
            except Exception as exc:
                logger.warning("Gemini HTTP API call failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        return (None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "input_cost": 0.0, "output_cost": 0.0})




class HintGenerator:
    """Phase 2: classify candidate functions with the LLM, in batches, into
    allocator/deallocator summaries. Summaries come only from the LLM."""

    def __init__(self, llm_client: LLMClient = None, codeql_validator=None):
        self.llm = llm_client
        self.codeql_validator = codeql_validator
        self.function_costs: dict[str, dict] = {}
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.estimated_prompt_tokens: int = 0
        self.estimated_completion_tokens: int = 0
        self.estimated_total_tokens: int = 0
        self.estimated_calls: int = 0
        self.estimated_input_cost: float = 0.0
        self.estimated_output_cost: float = 0.0
        self.estimated_total_cost: float = 0.0
        self.pointer_typedef_aliases: set[str] = set()
        self.last_filter_classes: dict[str, list[str]] | None = None
        self.last_all_results: dict[str, list[Hint]] = {}
        self.processed_functions: set[str] = set()
        self.failed_functions: set[str] = set()

        self.custom_query_generator: CustomQueryGenerator | None = None

    def set_pointer_typedef_aliases(self, aliases: set[str] | None) -> None:
        self.pointer_typedef_aliases = set(aliases or set())

    def _has_pointer_io(self, func: FunctionInfo) -> bool:
        """Strict pointer-based IO filtering."""

        rt = func.return_type or ""
        if "*" in rt:
            return True

        for t in func.arg_types or []:
            if t and "*" in t:
                return True

        alias = self.pointer_typedef_aliases

        rt_token = (rt.split()[-1] if rt else None)
        if rt_token in alias:
            return True

        for t in func.arg_types or []:
            if not t:
                continue
            token = t.split()[-1]
            if token in alias:
                return True

        return False


    def generate_hints(
        self,
        functions: dict[str, FunctionInfo],
        previous_conflicts: list[str] = None,
        macro_names: set[str] = None,
        pointer_typedef_aliases: set[str] | None = None,
        progress_callback: Optional[Callable[[str, HintSet], None]] = None,
        existing_hints: HintSet | None = None,
        processed_functions: set[str] | None = None,
        batch_size: int = 1,
    ) -> HintSet:
        """Generate hints for all functions in codebase."""
        hint_set = existing_hints if existing_hints is not None else HintSet()
        self.processed_functions = set(processed_functions or set())
        self.failed_functions = set()
        macro_names = macro_names or set()
        if pointer_typedef_aliases is not None:
            self.set_pointer_typedef_aliases(pointer_typedef_aliases)

        conflict_map = {}
        if previous_conflicts:
            for conflict in previous_conflicts:
                if conflict.startswith("REMOVED "):
                    parts = conflict[8:].split(": ", 1)
                    if len(parts) == 2:
                        func_hint = parts[0]
                        reason = parts[1]
                        if "." in func_hint:
                            func_name = func_hint.split(".")[0]
                            if func_name not in conflict_map:
                                conflict_map[func_name] = []
                            conflict_map[func_name].append(reason)

        candidate_functions: dict[str, FunctionInfo] = {}
        filtered_entry_functions: list[str] = []
        filtered_pointer_functions: list[str] = []
        
        for fn, f in functions.items():
            if fn in ("main", "_main", "wmain") or "test" in fn.lower():
                filtered_entry_functions.append(fn)
                continue

            if fn in macro_names or self._has_pointer_io(f):
                candidate_functions[fn] = f
            else:
                filtered_pointer_functions.append(fn)

        self.last_filter_classes = {
            "kept": sorted(candidate_functions.keys()),
            "filtered_entry": sorted(filtered_entry_functions),
            "filtered_non_pointer": sorted(filtered_pointer_functions),
        }

        self.last_all_results = {}
        for fn in functions.keys():
            self.last_all_results[fn] = []

        total_filtered = len(filtered_entry_functions) + len(filtered_pointer_functions)
        if candidate_functions and total_filtered:
            logger.info(
                "LLM filter: %d/%d functions selected as LLM candidates; "
                "skipping %d functions (entry/test: %d, non-pointer: %d)",
                len(candidate_functions),
                len(functions),
                total_filtered,
                len(filtered_entry_functions),
                len(filtered_pointer_functions),
            )
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Functions NOT filtered (will be sent to LLM):")
                for fn in sorted(candidate_functions.keys()):
                    reason = "macro" if fn in macro_names else "pointer/alloc/free related"
                    logger.debug(f"  - {fn} ({reason})")
                
                if filtered_entry_functions:
                    logger.debug("Functions FILTERED as entry/test (skipped):")
                    for fn in sorted(filtered_entry_functions):
                        logger.debug(f"  - {fn}")

                if filtered_pointer_functions:
                    logger.debug("Functions FILTERED as non-pointer IO (skipped):")
                    for fn in sorted(filtered_pointer_functions):
                        logger.debug(f"  - {fn}")

        already_hinted_funcs: set[str] = set(hint_set.hints.keys())
        already_processed: set[str] = set(processed_functions or set())
        already_done_funcs: set[str] = already_hinted_funcs | already_processed
        remaining_candidates: dict[str, FunctionInfo] = {
            fn: f for fn, f in candidate_functions.items() if fn not in already_done_funcs
        }

        if existing_hints is not None:
            for fn, func_hints in existing_hints.hints.items():
                if fn in self.last_all_results:
                    self.last_all_results[fn] = list(func_hints)
                    self.processed_functions.add(fn)

        if self.llm and remaining_candidates:
            est_prompt_tokens = 0

            for func_name, func in remaining_candidates.items():
                context_parts = []
                for callee in _first_callees(func, 5):
                    if callee in functions:
                        callee_code = functions[callee].code
                        context_parts.append(f"// Called function:\n{callee_code}")
                context = "\n\n".join(context_parts) if context_parts else ""
                context_str = f"Context (called functions):\n{context}" if context else ""

                func_conflicts = conflict_map.get(func_name, [])
                conflict_feedback = ""
                if func_conflicts:
                    conflict_feedback = (
                        "## Z3 Validation Feedback\n\n"
                        "The following hints were rejected by Z3:\n"
                        + "\n".join(f"- {c}" for c in func_conflicts)
                    )

                params = list(zip(func.arg_types, func.arg_names))
                params_str = ", ".join(f"{t} {n}" for t, n in params) if params else "void"

                block = (
                    f"### Function: {func.name}\n"
                    f"Return type: {func.return_type or 'void'}\n"
                    f"Parameters: {params_str}\n\n"
                    f"```c\n{func.code}\n```\n\n"
                    f"{context_str}\n{conflict_feedback}"
                )

                est_prompt_tokens += max(1, len(block) // 3)

            try:
                est_batch_size = int(batch_size)
            except (TypeError, ValueError):
                est_batch_size = 1
            if est_batch_size < 1:
                est_batch_size = 1

            num_funcs_to_call = len(remaining_candidates)
            est_calls = (num_funcs_to_call + est_batch_size - 1) // est_batch_size if num_funcs_to_call > 0 else 0

            if est_calls > 0:
                envelope = len(BATCH_HINT_GENERATION_PROMPT) - len("{functions_block}")
                est_prompt_tokens += est_calls * max(1, envelope // 3)

                input_ppm = getattr(self.llm, "input_price_per_million", 0.0) or 0.0
                est_input_cost = (est_prompt_tokens / 1_000_000) * input_ppm

                EST_OUTPUT_TOKENS_PER_CALL = 800
                est_completion_tokens = est_calls * EST_OUTPUT_TOKENS_PER_CALL
                output_ppm = getattr(self.llm, "output_price_per_million", 0.0) or 0.0
                est_output_cost = (est_completion_tokens / 1_000_000) * output_ppm

                est_total_tokens = est_prompt_tokens + est_completion_tokens
                est_total_cost = est_input_cost + est_output_cost

                self.estimated_prompt_tokens = est_prompt_tokens
                self.estimated_completion_tokens = est_completion_tokens
                self.estimated_total_tokens = est_total_tokens
                self.estimated_calls = est_calls
                self.estimated_input_cost = est_input_cost
                self.estimated_output_cost = est_output_cost
                self.estimated_total_cost = est_total_cost

                logger.info(
                    "LLM pre-estimate (rough): input ≈ %d tokens, output ≈ %d tokens "
                    "across %d call(s) (batch size ≈ %d); estimated input cost ≈ $%.4f, "
                    "output cost ≈ $%.4f, total ≈ $%.4f (chars/3 heuristic + fixed %d output tokens/call).",
                    est_prompt_tokens,
                    est_completion_tokens,
                    est_calls,
                    est_batch_size,
                    est_input_cost,
                    est_output_cost,
                    est_total_cost,
                    EST_OUTPUT_TOKENS_PER_CALL,
                )

        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            batch_size = 1
        if batch_size < 1:
            batch_size = 1

        resume_index_str = os.getenv("HINT_RESUME_FROM_INDEX")
        resume_index: int | None = None
        if resume_index_str:
            try:
                resume_index = int(resume_index_str)
                if resume_index <= 0:
                    resume_index = None
            except ValueError:
                resume_index = None

        resume_from = os.getenv("HINT_RESUME_FROM_FUNC")

        candidate_names: list[str] = list(candidate_functions.keys())

        for func_name in candidate_names:
            if func_name in already_done_funcs:
                self.last_all_results[func_name] = list(
                    hint_set.hints.get(func_name, [])
                )
                self.processed_functions.add(func_name)
                if progress_callback is not None:
                    try:
                        progress_callback(func_name, hint_set)
                    except Exception as e:
                        logger.warning("progress_callback failed for %s: %s", func_name, e)

        names_to_call: list[str] = []
        current_idx = 0
        waiting_for_resume = bool(resume_from)

        for func_name in candidate_names:
            current_idx += 1

            if resume_index is not None and current_idx < resume_index:
                continue

            if waiting_for_resume:
                if func_name == resume_from:
                    waiting_for_resume = False
                else:
                    continue

            if func_name in already_done_funcs:
                continue

            names_to_call.append(func_name)

        if not names_to_call:
            return hint_set

        for i in tqdm(range(0, len(names_to_call), batch_size), desc="Generating hints (batched)"):
            batch_names = names_to_call[i : i + batch_size]
            batch_funcs = [candidate_functions[n] for n in batch_names]

            batch_conflicts: dict[str, list[str]] = {
                name: conflict_map.get(name, []) for name in batch_names
            }

            batch_results = self._llm_hints_batch(batch_funcs, functions, batch_conflicts)

            for func_name in batch_names:
                hints = batch_results.get(func_name, [])

                unique_hints: list[Hint] = []
                for llm_hint in hints:
                    existing = next(
                        (h for h in unique_hints
                         if h.hint_type == llm_hint.hint_type and h.target == llm_hint.target),
                        None,
                    )
                    if not existing:
                        unique_hints.append(llm_hint)

                for hint in unique_hints:
                    hint_set.add(hint)

                self.last_all_results[func_name] = list(unique_hints)
                if func_name not in self.failed_functions:
                    self.processed_functions.add(func_name)

                if progress_callback is not None:
                    try:
                        progress_callback(func_name, hint_set)
                    except Exception as e:
                        logger.warning("progress_callback failed for %s: %s", func_name, e)

        return hint_set

    @staticmethod
    def _parse_hint_type(h: dict) -> HintType | None:
        """Parse hint type from LLM output, supporting both old and new
        formats."""
        role = h.get("role", "")
        if role:
            role_upper = role.strip().upper()
            if role_upper == "ALLOCATOR":
                return HintType.ALLOCATOR
            if role_upper == "DEALLOCATOR":
                return HintType.DEALLOCATOR
            return None
        type_str = h.get("type", "")
        try:
            return HintType[type_str]
        except KeyError:
            return None

    @staticmethod
    def _parse_arg_index(h: dict, target: str) -> int:
        """Derive arg_index from target string."""
        if "arg_index" in h:
            try:
                return int(h["arg_index"])
            except (TypeError, ValueError):
                pass
        if target == "return":
            return -1
        if target.startswith("arg"):
            try:
                return int(target[3:])
            except ValueError:
                return 0
        return -1

    def _llm_hints_batch(
        self,
        funcs: list[FunctionInfo],
        all_functions: dict[str, FunctionInfo],
        batch_conflicts: dict[str, list[str]] | None = None,
    ) -> dict[str, list[Hint]]:
        """Generate hints for multiple functions in a single LLM call."""
        if not self.llm or not funcs:
            return {}

        batch_conflicts = batch_conflicts or {}

        func_blocks: list[str] = []
        for func in funcs:
            context_parts = []
            for callee in _first_callees(func, 5):
                if callee in all_functions:
                    callee_code = all_functions[callee].code
                    context_parts.append(f"// Called function:\n{callee_code}")

            context = "\n\n".join(context_parts) if context_parts else ""
            context_str = f"Context (called functions):\n{context}" if context else ""

            previous_conflicts = batch_conflicts.get(func.name, []) or []
            conflict_feedback = ""
            if previous_conflicts:
                conflict_feedback = f"""
## Z3 Validation Feedback for {func.name}

The following hints were **rejected** by Z3's constraint solving and path feasibility analysis:

{chr(10).join(f"- {c}" for c in previous_conflicts)}

The rejected hints are inconsistent with the code structure. Please:
1. Re-examine if the memory operation actually occurs in this function
2. Only re-submit hints you are confident are correct
"""

            params = list(zip(func.arg_types, func.arg_names))
            params_str = ", ".join(f"{t} {n}" for t, n in params) if params else "void"

            block = f"""
### Function: {func.name}
Return type: {func.return_type or "void"}
Parameters: {params_str}

```c
{func.code}
```

{context_str}
{conflict_feedback}
"""
            func_blocks.append(block.strip())


        functions_block = "\n\n".join(func_blocks)
        batch_prompt = BATCH_HINT_GENERATION_PROMPT.format(
            functions_block=functions_block,
        )

        logger.debug(
            "LLM batched prompt for %d functions:\n%s",
            len(funcs),
            batch_prompt,
        )

        result = self.llm.query(batch_prompt)
        usage = result.get("usage", {}) or {}
        content = result.get("content", {}) or {}

        batch_tokens = int(usage.get("total_tokens", 0) or 0)
        batch_cost = float(usage.get("cost", 0.0) or 0.0)
        self.total_cost += batch_cost
        self.total_tokens += batch_tokens

        # No usable JSON (a failed call or an unparsable reply): the batch is
        # marked failed and retried, rather than read as "neither".
        if not content or not isinstance(content, dict):
            logger.warning("Phase 2: no usable reply for a batch of %d functions; marked for retry", len(funcs))
            for func in funcs:
                self.failed_functions.add(func.name)
            return {}

        n_funcs = max(1, len(funcs))
        per_func_cost = batch_cost / n_funcs
        per_func_tokens = batch_tokens // n_funcs if batch_tokens > 0 else 0
        per_func_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0) // n_funcs if n_funcs > 0 else 0
        per_func_completion_tokens = int(usage.get("completion_tokens", 0) or 0) // n_funcs if n_funcs > 0 else 0
        per_func_input_cost = float(usage.get("input_cost", 0.0) or 0.0) / n_funcs if n_funcs > 0 else 0.0
        per_func_output_cost = float(usage.get("output_cost", 0.0) or 0.0) / n_funcs if n_funcs > 0 else 0.0

        hints_by_func: dict[str, list[Hint]] = {}

        fn_map = content

        for func in funcs:
            func_name = func.name
            if func_name not in fn_map:
                logger.warning("Phase 2: reply omitted %s; marked for retry", func_name)
                self.failed_functions.add(func_name)
                continue
            func_entry = fn_map.get(func_name) or {}
            func_hints_raw = func_entry.get("hints", []) or [] if isinstance(func_entry, dict) else []

            parsed_hints: list[Hint] = []
            for h in func_hints_raw:
                if not isinstance(h, dict):
                    continue
                hint_type = self._parse_hint_type(h)
                if hint_type is None:
                    continue

                target = h.get("target", "return")
                arg_index = self._parse_arg_index(h, target)

                parsed_hints.append(
                    Hint(
                        function_name=func_name,
                        hint_type=hint_type,
                        target=target,
                        arg_index=arg_index,
                        reason=h.get("reason", ""),
                    )
                )

            hints_by_func[func_name] = parsed_hints

            self.function_costs[func_name] = {
                "cost": per_func_cost,
                "tokens": per_func_tokens,
                "prompt_tokens": per_func_prompt_tokens,
                "completion_tokens": per_func_completion_tokens,
                "input_cost": per_func_input_cost,
                "output_cost": per_func_output_cost,
            }

        return hints_by_func
    
    def get_cost_summary(self) -> dict:
        """Get cost summary with per-function and per-hint breakdowns."""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "estimated_prompt_tokens": getattr(self, "estimated_prompt_tokens", 0),
            "estimated_calls": getattr(self, "estimated_calls", 0),
            "estimated_input_cost": getattr(self, "estimated_input_cost", 0.0),
            "function_costs": self.function_costs,
        }
    
"""
CustomQueryGenerator: Generates custom CodeQL queries for functions with special semantics.

This module uses an LLM to analyze C/C++ functions and generate targeted CodeQL queries
for bug finding or false positive suppression based on memory safety hints and parameter semantics.
"""


class CodeQLValidator:
    """Validator that uses CodeQL CLI to compile and validate queries."""
    
    def __init__(self, codeql_binary: str = "codeql", codeql_dir: Path | None = None, 
                 database_path: Path | None = None, timeout: int = 30):
        """Initialize CodeQL validator."""
        self.codeql_binary = codeql_binary
        self.codeql_dir = codeql_dir
        self.database_path = database_path
        self.timeout = timeout
        
        if not which(codeql_binary):
            raise ValueError(
                f"CodeQL binary '{codeql_binary}' not found in PATH. "
                "Please install CodeQL or provide the correct path."
            )
    
    def validate_query(self, query_code: str) -> Tuple[bool, str]:
        if not query_code or not query_code.strip():
            return False, "Query code is empty"

        if self.codeql_dir:
            base = self.codeql_dir / "qlpacks" / "codeql" / "cpp-queries"
        else:
            base = Path.home() / ".codeql" / "packages" / "codeql" / "cpp-queries"
        if base.exists():
            versions = sorted([d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")], reverse=True)
            query_dir = versions[0] / "Critical" if versions else base
        else:
            query_dir = base
        query_file = query_dir / "tmp.ql"

        try:
            query_file.write_text(query_code)

            cmd = [self.codeql_binary, "query", "compile", "--check-only", str(query_file)]

            env = os.environ.copy()
            if self.codeql_dir:
                env["CODEQL_HOME"] = str(self.codeql_dir)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=str(query_dir),
            )

            if result.returncode == 0:
                logger.info("CodeQL query validation passed successfully")
                return True, ""

            error_msg = result.stderr.strip() or result.stdout.strip()
            if not error_msg:
                error_msg = f"CodeQL query compile failed with return code {result.returncode}"

            error_msg = error_msg.replace(str(query_file), "tmp.ql")
            error_msg = error_msg.replace(str(query_dir), "")

            return False, error_msg

        except Exception as e:
            return False, f"Validation error: {str(e)}"

        finally:
            try:
                query_file.unlink(missing_ok=True)
            except Exception:
                pass


class CustomQueryGenerator:
    """Generates custom CodeQL queries for functions with special semantics."""
    
    def __init__(self, llm: Any = None, codeql_validator: Any = None):
        """Initialize the query generator."""
        self.llm = llm
        self.codeql_validator = codeql_validator
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.function_costs: Dict[str, Dict[str, float]] = {}
        
        if not self.codeql_validator:
            try:
                self.codeql_validator = CodeQLValidator()
                logger.info("CodeQL validator created automatically for query validation")
            except Exception as e:
                logger.debug(
                    f"CodeQL validator not available (CodeQL not in PATH): {e}. "
                    "Validation will use enhanced quick checks only."
                )
                self.codeql_validator = None
    
    
    def generate_custom_query_for_function(
        self,
        func: Any,
        hints: List[Any],
    ) -> Optional[CustomQuery]:
        """Generate a custom query for a function if it has special semantics."""
        if not self.llm:
            return None
        
        hints_text, arg_semantics_text = self._format_hints_and_mad(func, hints)
        prompt = self._build_query_generation_prompt(func, hints_text, arg_semantics_text, hints)
        
        logger.debug(
            "Generating custom query for function: %s",
            getattr(func, "name", "unknown")
        )
        
        result = self.llm.query(prompt) or {}
        usage = result.get("usage", {}) or {}
        content = result.get("content", {}) or {}
        
        self._track_cost(getattr(func, "name", ""), usage)
        
        is_special = bool(content.get("is_special", False))
        reason = (content.get("reason", "") or "LLM determined function needs custom filters").strip()
        
        func_name = getattr(func, "name", "")
        if is_special:
            logger.info(
                "Function '%s' is SPECIAL: %s",
                func_name,
                reason
            )
        else:
            logger.info(
                "Function '%s' is NOT special: %s",
                func_name,
                reason
            )

        never_freed_block = content.get("memory_never_freed_filter") or {}
        may_not_be_freed_block = content.get("memory_may_not_be_freed_filter") or {}

        def _has_filter_block(block: Any) -> bool:
            return isinstance(block, dict) and any(
                (block.get("predicates_code") or "").strip()
                or (block.get("use_expr") or "").strip()
            )

        has_new_filters = any(_has_filter_block(b) for b in (never_freed_block, may_not_be_freed_block))

        validated = False
        validation_error = ""

        if is_special and has_new_filters:
            func_name = getattr(func, "name", "")

            if _has_filter_block(never_freed_block):
                never_freed_block = self._validate_and_fix_filter_block(
                    func_name=func_name,
                    filter_type="memory_never_freed_filter",
                    block=never_freed_block,
                    original_prompt=prompt,
                )

            if _has_filter_block(may_not_be_freed_block):
                may_not_be_freed_block = self._validate_and_fix_filter_block(
                    func_name=func_name,
                    filter_type="memory_may_not_be_freed_filter",
                    block=may_not_be_freed_block,
                    original_prompt=prompt,
                )

        return CustomQuery(
            function_name=getattr(func, "name", ""),
            reason=reason,
            is_special=is_special,
            validated=validated,
            validation_error=validation_error,
            memory_never_freed_filter=never_freed_block if is_special else {},
            memory_may_not_be_freed_filter=may_not_be_freed_block if is_special else {},
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of LLM API costs and token usage."""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "function_costs": self.function_costs,
        }
    
    
    def _format_hints_and_mad(
        self,
        func: Any,
        hints: List[Any]
    ) -> Tuple[str, str]:
        """Convert hints and MAD (Memory Access Description) into readable text
        blocks."""
        hints_details: List[str] = []
        all_arg_semantics: Dict[int, List[str]] = {}
        
        for hint in hints or []:
            hint_type_name = getattr(
                getattr(hint, "hint_type", None),
                "name",
                str(getattr(hint, "hint_type", ""))
            )
            target = getattr(hint, "target", "")
            reason = getattr(hint, "reason", "")
            
            detail = f"- **{hint_type_name}** on `{target}`"
            if reason:
                detail += f": {reason}"
            hints_details.append(detail)
            
            arg_sem = getattr(hint, "arg_semantics", {}) or {}
            for arg_idx, semantic_desc in arg_sem.items():
                try:
                    arg_i = int(arg_idx)
                except (ValueError, TypeError):
                    continue
                all_arg_semantics.setdefault(arg_i, []).append(str(semantic_desc))
        
        hints_text = "\n".join(hints_details) if hints_details else "No hints available"
        
        arg_semantics_text = ""
        if all_arg_semantics:
            arg_lines: List[str] = []
            arg_names = getattr(func, "arg_names", []) or []
            arg_types = getattr(func, "arg_types", []) or []
            
            for arg_idx in sorted(all_arg_semantics.keys()):
                descs = "; ".join(sorted(set(all_arg_semantics[arg_idx])))
                arg_name = arg_names[arg_idx] if arg_idx < len(arg_names) else f"arg{arg_idx}"
                arg_type = arg_types[arg_idx] if arg_idx < len(arg_types) else "unknown"
                arg_lines.append(f"- **Argument {arg_idx}** (`{arg_name}: {arg_type}`): {descs}")
            
            arg_semantics_text = "\n".join(arg_lines)
        
        return hints_text, arg_semantics_text
    
    
    def _build_query_generation_prompt(
        self,
        func: Any,
        hints_text: str,
        arg_semantics_text: str,
        hints: Optional[List[Any]] = None
    ) -> str:
        """Build the LLM prompt for deciding specialness and generating a
        query."""
        name = getattr(func, "name", "")
        code = getattr(func, "code", "")
        ret = getattr(func, "return_type", "") or "void"
        arg_names = getattr(func, "arg_names", []) or []
        arg_types = getattr(func, "arg_types", []) or []
        
        params_str = ", ".join(
            [f"{t} {n}" for t, n in zip(arg_types, arg_names)]
        ) if arg_names else "void"
        
        is_allocator = False
        is_deallocator = False
        allocator_targets = []
        deallocator_targets = []
        
        if hints:
            for hint in hints:
                hint_type = getattr(hint, "hint_type", None)
                if hint_type:
                    hint_type_name = getattr(hint_type, "name", str(hint_type))
                    target = getattr(hint, "target", "")
                    
                    if hint_type_name == "ALLOCATOR":
                        is_allocator = True
                        allocator_targets.append(target)
                    elif hint_type_name == "DEALLOCATOR":
                        is_deallocator = True
                        deallocator_targets.append(target)
        
        semantic_parts = []
        if is_allocator:
            targets_str = ", ".join(allocator_targets) if allocator_targets else "return value"
            semantic_parts.append(f"ALLOCATOR (on {targets_str})")
        if is_deallocator:
            targets_str = ", ".join(deallocator_targets) if deallocator_targets else "arguments"
            semantic_parts.append(f"DEALLOCATOR (on {targets_str})")
        
        semantic_info = " | ".join(semantic_parts) if semantic_parts else "Neither allocator nor deallocator"
        
        PROMPT = f'''You are an expert CodeQL query generator specializing in C/C++ memory safety analysis.

Your task has TWO parts:
1. Determine if this function has SPECIAL SEMANTICS that cause false positive warnings
2. If special, generate FILTER PLUGIN PREDICATES (not full queries) to suppress safe patterns

IMPORTANT:
- You are generating ONLY predicate definitions that will be inserted into an existing CodeQL query pack.
- DO NOT generate full queries.
- DO NOT generate metadata headers (no @name/@id/...).
- DO NOT generate select clauses.
- DO NOT redefine aggregator predicates like dfFiltered/uafFiltered/leakFiltered/mayNotBeFreedFiltered.

- DO NOT write import statements inside predicates_code.
- If your predicates require additional CodeQL modules (e.g., DataFlow, IR, StackVariableReachability, etc.),
  you MUST declare them in a separate field called "required_imports".
- "required_imports" must be a list of import lines WITHOUT the word "import".
  Example:
      "required_imports": ["semmle.code.cpp.dataflow.new.DataFlow", "semmle.code.cpp.ir.IR"]
- DO NOT include "cpp" in required_imports (it is already imported).
- Only list modules that are actually used in your predicates_code.


==================================================
FUNCTION ANALYSIS
==================================================

Function Name: `{name}`
Return Type: `{ret}`
Parameters: `{params_str}`
Memory Semantics: {semantic_info}

Source Code:
```c
{code}
```

==================================================
MEMORY SAFETY HINTS (Z3-Validated)
==================================================
{hints_text}

==================================================
DECISION FRAMEWORK
==================================================

A function is SPECIAL if standard CodeQL memory queries produce false positives because:
- It performs partial cleanup (does not free the main object)
- It frees different substructures depending on index/flag
- It is reference-counted
- It is idempotent due to internal state checks (e.g., sets pointer to NULL)
- It uses arena/pool logic (returns memory to a pool)
- Multiple calls are intentionally safe

Your goal is to generate FILTER PLUGIN PREDICATES that identify SAFE cases for:
- DOUBLE FREE
- USE AFTER FREE
- MEMORY LEAK (never freed / may not be freed)

If the function is NOT special, return is_special=false and leave all filters empty.

==================================================
REAL FILTER REQUIREMENT (NO BLANKET SUPPRESSION)
==================================================

A filter MUST identify a *provably safe* pattern. It is NOT allowed to suppress warnings
just because the allocation/free/use comes from function `{name}`.

❌ FORBIDDEN (NOT A FILTER):
Any predicate that returns true using only "target name is `{name}`" (or wrappers)
without any additional safety proof.

Examples that are FORBIDDEN:
- exists(FunctionCall c | c = alloc and c.getTarget().getName() = "{name}")
- exists(FunctionCall c | c = alloc and c.getTarget().hasGlobalName("{name}")
- exists(FunctionCall c | c = dealloc and c.getTarget().getName() = "{name}")
- Any leak/UAF/DF filter whose only condition is matching `{name}`

Why forbidden:
- `{name}` may be an allocator-like function that returns owned memory to the caller.
  Blanket suppression would hide real leaks in callers.

✅ REQUIRED (WHAT A REAL FILTER MUST PROVE):

For MEMORY NEVER FREED filters (AllocationExpr alloc):
You MUST prove ownership transfer or safe escape, e.g. at least ONE of:
- The call result is returned by the enclosing function
- The call result is assigned/stored into a field/global/pointer-deref (heap escape)
- The call result is passed into an owner-taking API (ONLY if such API is explicitly named in hints)

If you cannot prove a concrete ownership transfer/escape, leave the memory_never_freed_filter empty.

For MEMORY MAY NOT BE FREED filters (ControlFlowNode def):
You MUST bind def to an allocation expression and prove escape/return/ownership transfer.

For DOUBLE FREE / USE AFTER FREE filters:
You MUST prove the second event is safe due to semantics (different index/flag, idempotence, refcount, etc.).
Never suppress based only on function name.

==================================================
PIPELINE SIGNATURE RULE (MANDATORY)
==================================================

Your filters are PLUGINS called by existing aggregators in the main query.
Therefore, your predicates MUST match the aggregator's expected parameters and ONLY use those parameters.

DOUBLE FREE plugin signature (EXACT):
  predicate <PluginName>(DeallocationExpr srcDealloc, DataFlow::Node sinkNode, Expr sinkFreedExpr) {{{{ ... }}}}

USE-AFTER-FREE plugin signature (EXACT — this pipeline uses 2 parameters):
  predicate <PluginName>(DeallocationExpr dealloc, DataFlow::Node sinkNode) {{{{ ... }}}}

MEMORY NEVER FREED plugin signature (EXACT):
  // main query uses: not leakFiltered(alloc)
  predicate <PluginName>(AllocationExpr alloc) {{{{ ... }}}}

MEMORY MAY NOT BE FREED plugin signature (EXACT):
  // main query uses: not mayNotBeFreedFiltered(def)
  // where def is a ControlFlowNode
  predicate <PluginName>(ControlFlowNode def) {{{{ ... }}}}

RULES:
- If you need extra locals (call1/call2/etc.), bind them INSIDE exists(...).
- use_expr MUST reference ONLY pipeline parameters:
  - never-freed: (alloc)
  - may-not-be-freed: (def)
- NEVER output use_expr containing free variables like call1/call2.
- NEVER redefine leakFiltered/mayNotBeFreedFiltered (aggregators are owned by the main query).

==================================================
HARD BAN LIST (COMPILATION FAILURES)
==================================================

- getExpr()
- getASubExpression()
- getEnclosingFunctionCall()
- Casting FunctionCall to DeallocationExpr
- Redefining aggregator predicates
- Any use_expr that references undefined variables (e.g., call1/call2)
- For memory_may_not_be_freed_filter: def.(ExprCfgNode).getExpr() [ExprCfgNode does not exist]
- For memory_may_not_be_freed_filter: DataFlow::node(def) [ControlFlowNode is not compatible with DataFlow::Node]
- For memory_may_not_be_freed_filter: def.getExpr() [must use def.(AnalysedExpr).getExpr() instead]

Note: If the MAIN query already uses sinkNode.asExpr(), you may also use sinkNode.asExpr() in plugins.
Otherwise, do not invent new conversions.

For memory_may_not_be_freed_filter: ALWAYS use def.(AnalysedExpr).getExpr() to extract expressions from ControlFlowNode.

==================================================
SCOPING RULE (MANDATORY)
==================================================

Your filters MUST be scoped ONLY to this function `{name}` (or its direct wrappers if explicitly mentioned in hints).
Do NOT generate generic filters that could apply to unrelated functions.

==================================================
EXAMPLE WHEN GENERATING A MEMORY NEVER FREED FILTER
(SINGLE-PREDICATE PLUGIN, NO HELPER PREDICATES)
==================================================

Context: the enhanced "never freed" query contains:
  not leakFiltered(alloc)

So your plugin must have signature:
  predicate <PluginName>(AllocationExpr alloc)

Example:

predicate leakNeverFreedFilterExample_{name}(AllocationExpr alloc) {{{{
  exists(FunctionCall c, ReturnStmt ret |
    // allocation originates from calling a special allocator {name}
    c = alloc and
    c.getTarget().getName() = "{name}" and

    // Example safe-suppression condition: ownership is transferred out via return
    ret.getEnclosingFunction() = c.getEnclosingFunction() and
    ret.getExpr() = c
  )
}}}}

use_expr example:
leakNeverFreedFilterExample_{name}(alloc)

==================================================
EXAMPLE WHEN GENERATING A MEMORY MAY NOT BE FREED FILTER
(SINGLE-PREDICATE PLUGIN, NO HELPER PREDICATES)
==================================================

Context: the enhanced "may not be freed" query contains:
  not mayNotBeFreedFiltered(def)
where `def` is a ControlFlowNode (allocation definition point).

So your plugin must have signature:
  predicate <PluginName>(ControlFlowNode def)

CRITICAL: To extract an expression from a ControlFlowNode, you MUST use:
  e = def.(AnalysedExpr).getExpr()

DO NOT use:
  - def.(ExprCfgNode).getExpr()  [ExprCfgNode does not exist]
  - DataFlow::node(def).asExpr()  [ControlFlowNode is not compatible with DataFlow::Node]
  - def.getExpr()  [ControlFlowNode does not have getExpr() directly]

Example:

predicate leakMayNotBeFreedFilterExample_{name}(ControlFlowNode def) {{{{
  exists(Expr e, FunctionCall c, Assignment a |
    // CRITICAL: Use def.(AnalysedExpr).getExpr() to extract expression from ControlFlowNode
    e = def.(AnalysedExpr).getExpr() and
    c = e and
    c.getTarget().getName() = "{name}" and

    // Example safe-suppression condition: allocation escapes to a field/global
    a.getRValue() = c and
    not a.getLValue().(VariableAccess).getTarget() instanceof StackVariable
  )
}}}}

use_expr example:
leakMayNotBeFreedFilterExample_{name}(def)

==================================================
OUTPUT FORMAT (STRICT JSON)
==================================================

Return ONLY a valid JSON object with these exact fields:
{{{{
  "is_special": boolean,
  "reason": "Clear explanation of why this function is/isn't special",

  "memory_never_freed_filter": {{{{
    "required_imports": ["module.path.IfNeeded"],
    "predicates_code": "ONLY predicate definitions or empty string",
    "use_expr": "Single boolean expression or empty string"
  }}}},

  "memory_may_not_be_freed_filter": {{{{
    "required_imports": ["module.path.IfNeeded"],
    "predicates_code": "ONLY predicate definitions or empty string",
    "use_expr": "Single boolean expression or empty string"
  }}}}
}}}}

GUIDELINES:
- If the function is not special, set is_special=false and leave all predicates_code/use_expr as empty strings.
- If special only for some bug types, fill only those blocks.
- For each non-empty block:
  - predicates_code MUST contain ONLY predicate definitions (no imports, no select, no metadata, no aggregator).
  - use_expr MUST call your predicate using ONLY in-scope pipeline parameters.

==================================================
SELF-CHECK BEFORE RETURNING JSON
==================================================

Before output:
- Ensure NO getExpr/getASubExpression/getEnclosingFunctionCall appears.
- Ensure you did NOT redefine leakFiltered/mayNotBeFreedFiltered.
- Ensure your use_expr contains ONLY the pipeline parameters.
- Ensure any local variables are bound inside exists(...).
- Ensure required_imports lists ONLY modules actually used.
- Ensure you did NOT include "import" keyword in required_imports.
- Ensure JSON is valid and contains only the required fields.


Now analyze the function and respond with JSON only.
'''
        
        return PROMPT
    
    
    def _insert_imports_into_template(self, template: str, required_imports: set[str]) -> str:
        """Insert additional imports into a CodeQL query template."""
        if not required_imports:
            return template
        
        import_lines = sorted(required_imports)
        additional_imports = "\n".join(f"import {imp}" for imp in import_lines)
        
        last_import_pos = -1
        last_import_end = -1
        
        import_patterns = [
            "import semmle.code.cpp.dataflow.new.DataFlow",
            "import semmle.code.cpp.controlflow.StackVariableReachability",
            "import MemoryFreed",
            "import cpp",
        ]
        
        for pattern in import_patterns:
            pos = template.rfind(pattern)
            if pos != -1:
                end_pos = template.find("\n", pos)
                if end_pos != -1:
                    if pos > last_import_pos:
                        last_import_pos = pos
                        last_import_end = end_pos
        
        if last_import_pos != -1 and last_import_end != -1:
            blank_line_pos = template.find("\n\n", last_import_end)
            if blank_line_pos != -1:
                return (
                    template[:blank_line_pos + 1] +
                    additional_imports + "\n" +
                    template[blank_line_pos + 1:]
                )
            else:
                return (
                    template[:last_import_end] +
                    "\n" + additional_imports + "\n" +
                    template[last_import_end:]
                )
        
        return additional_imports + "\n\n" + template

    def _build_filter_wrapper_query(
        self,
        filter_type: str,
        predicates_code: str,
        use_expr: str,
        required_imports: Optional[set[str]] = None,
    ) -> str:
        """Build a validation query by inserting the filter into the actual
        production template."""
        if required_imports is None:
            required_imports = set()
        
        if filter_type == "memory_never_freed_filter":
            template = ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE
            preds_part = predicates_code if predicates_code else ""
            body = use_expr if use_expr else "false"
            leak_agg = (
                "predicate leakFiltered(AllocationExpr alloc) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = template
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + leak_agg
            return merged
            
        elif filter_type == "memory_may_not_be_freed_filter":
            template = ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE
            preds_part = predicates_code if predicates_code else ""
            body = use_expr if use_expr else "false"
            may_not_be_freed_agg = (
                "predicate mayNotBeFreedFiltered(ControlFlowNode def) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = template
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + may_not_be_freed_agg
            return merged
            
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")
    
    def _validate_and_fix_filter_block(
        self,
        func_name: str,
        filter_type: str,
        block: Dict[str, Any],
        original_prompt: str,
    ) -> Dict[str, Any]:
        """Validate a filter block (predicates_code + use_expr) and fix if
        needed."""
        predicates_code = (block.get("predicates_code") or "").strip()
        use_expr = (block.get("use_expr") or "").strip()
        
        required_imports: set[str] = set()
        imports = block.get("required_imports", [])
        if isinstance(imports, list):
            for imp in imports:
                if isinstance(imp, str) and imp.strip():
                    required_imports.add(imp.strip())
        
        if not predicates_code and not use_expr:
            return block
        
        wrapper_query = self._build_filter_wrapper_query(filter_type, predicates_code, use_expr, required_imports)
        
        logger.info(
            "Validating filter block '%s' for '%s' with merged query:\n%s",
            filter_type,
            func_name,
            wrapper_query
        )
        
        if self.codeql_validator:
            is_valid, error_msg = self.codeql_validator.validate_query(wrapper_query)
        else:
            is_valid, error_msg = False, "No CodeQL validator available - cannot validate query"
        
        if is_valid:
            logger.info(
                "Filter block '%s' for '%s' passed validation",
                filter_type,
                func_name
            )
            return {
                **block,
                "validated": True,
                "validation_error": "",
            }
        
        logger.warning(
            "Filter block '%s' for '%s' failed validation: %s. Attempting LLM fix...",
            filter_type,
            func_name,
            error_msg[:200]
        )
        
        for attempt in range(3):
            specific_guidance = ""
            if filter_type == "memory_may_not_be_freed_filter":
                if "ExprCfgNode" in error_msg or "DataFlow::node" in error_msg or "ControlFlowNode" in error_msg:
                    specific_guidance = """

**CRITICAL FIX GUIDANCE FOR memory_may_not_be_freed_filter:**
The parameter `def` is a ControlFlowNode. To extract an expression from it, you MUST use:
  e = def.(AnalysedExpr).getExpr()

DO NOT use:
  - def.(ExprCfgNode).getExpr()  [ExprCfgNode does not exist]
  - DataFlow::node(def).asExpr()  [ControlFlowNode is not compatible with DataFlow::Node]
  - def.getExpr()  [ControlFlowNode does not have getExpr() directly]

If your code uses any of these incorrect patterns, replace them with def.(AnalysedExpr).getExpr().
"""
            
            fix_prompt = f"""You previously generated a CodeQL filter for the function '{func_name}' (filter type: {filter_type}), but it has a compilation error when inserted into the production template.

**ERROR:**
{error_msg}
{specific_guidance}
**CURRENT FILTER CODE:**
```codeql
{predicates_code}
```

**USE EXPRESSION:**
```codeql
{use_expr}
```

**FULL QUERY (production template with your filter inserted - this is how your code is actually used):**
```codeql
{wrapper_query}
```

Please review the FULL QUERY above to understand the complete context. Your filter code (predicates_code and use_expr) is inserted into this production template. Fix ONLY the predicates_code and/or use_expr to resolve the compilation error. The fix must work when inserted into the production template shown above.

Return the corrected filter in JSON format:
{{
  "predicates_code": "corrected predicate definitions here",
  "use_expr": "corrected use expression here"
}}"""
            
            result = self.llm.query(fix_prompt) or {}
            content = result.get("content", {}) or {}
            
            new_predicates = (content.get("predicates_code") or predicates_code).strip()
            new_use_expr = (content.get("use_expr") or use_expr).strip()
            
            if not new_predicates and not new_use_expr:
                logger.warning("LLM did not return corrected filter code in attempt %d/3", attempt + 1)
                continue
            
            new_wrapper = self._build_filter_wrapper_query(filter_type, new_predicates, new_use_expr, required_imports)
            
            logger.info(
                "Validating fixed filter block '%s' for '%s' (attempt %d/3) with merged query:\n%s",
                filter_type,
                func_name,
                attempt + 1,
                new_wrapper
            )
            
            if self.codeql_validator:
                is_valid, error_msg = self.codeql_validator.validate_query(new_wrapper)
            else:
                is_valid, error_msg = False, "No CodeQL validator available - cannot validate query"
            
            if is_valid:
                logger.info(
                    "✓ Fixed filter block '%s' for '%s' passed validation",
                    filter_type,
                    func_name
                )
                return {
                    **block,
                    "predicates_code": new_predicates,
                    "use_expr": new_use_expr,
                    "validated": True,
                    "validation_error": "",
                }
            
            logger.warning(
                "Fix attempt %d/3 for '%s' still has error: %s",
                attempt + 1,
                filter_type,
                error_msg[:200]
            )
        
        logger.error(
            "Failed to fix filter block '%s' for '%s' after 3 fix attempts. Last error: %s",
            filter_type,
            func_name,
            error_msg[:200]
        )
        return {
            **block,
            "validated": False,
            "validation_error": f"Validation failed after 3 fix attempts: {error_msg[:200]}",
        }
    
    
    def _track_cost(self, func_name: str, usage: Dict[str, Any]) -> None:
        """Track API costs and token usage for analytics."""
        cost = float(usage.get("cost", 0.0) or 0.0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        
        self.total_cost += cost
        self.total_tokens += total_tokens
        
        entry = self.function_costs.get(func_name)
        if not entry:
            entry = {
                "cost": 0.0,
                "tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
            }
            self.function_costs[func_name] = entry
        
        entry["cost"] += cost
        entry["tokens"] += total_tokens
        entry["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        entry["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        entry["input_cost"] += float(usage.get("input_cost", 0.0) or 0.0)
        entry["output_cost"] += float(usage.get("output_cost", 0.0) or 0.0)