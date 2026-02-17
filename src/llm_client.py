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
import tempfile
import time
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional
from shutil import which
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    VERTEX_AI_AVAILABLE = True
except ImportError:  # pragma: no cover - environment may not have Vertex SDK
    vertexai = None
    GenerativeModel = None
    VERTEX_AI_AVAILABLE = False



from src.core.models import FunctionInfo, Hint, HintType, HintSet, CustomQuery

# Import templates from adapters to validate filters in the same structure as production
try:
    from src.analyzer.adapters import (
        ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE,
        ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE,
        ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE,
        ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE,
    )
except ImportError:
    # Fallback if adapters not available (shouldn't happen in normal usage)
    ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE = ""
    ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE = ""
    ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE = ""
    ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE = ""

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Template for Hint Generation
# =============================================================================

HINT_GENERATION_PROMPT = """You are a memory safety expert analyzing C/C++ code to identify function semantics that help static analyzers detect memory bugs.

## Task
Analyze this function and identify its MEMORY SEMANTICS relevant to allocation and deallocation.

**Function:** `{func_name}`
**Return type:** `{return_type}`
**Parameters:** `{parameters}`

```c
{code}
```
{context}

## Semantic Categories

### 1. ALLOCATOR
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

**Specify:** Use "return" if allocated memory is returned, or "argN" if allocated memory is written to an output parameter (e.g., `int alloc(void **out)` writes to arg0).

### 2. DEALLOCATOR
Function **frees/releases memory** passed as an argument.

**Positive indicators:**
- Calls free/delete/delete[]/g_free/kfree on an argument
- Calls another deallocator on an argument
- Wrapper around resource cleanup

**Specify:** Which argument (0-indexed) gets freed. If multiple arguments are freed, report each separately.

## Analysis Guidelines

1. **Trace data flow:** Follow where return values come from and where arguments flow to.
2. **Consider all paths:** Check all branches and return statements.
3. **Indirect calls matter:** If function calls helper that allocates/frees, propagate that semantic.
4. **Be precise:** Only report semantics you can verify from the code.
5. **Provide evidence:** Your reason should cite specific code elements (function calls, return statements, etc.)

## Output Format

Return a JSON object with hints array. Each hint must have:
- `type`: One of ALLOCATOR, DEALLOCATOR
- `target`: "return" for return value, or "argN" for argument N
- `arg_index`: 0-based index (-1 for return value, 0+ for arguments)
- `reason`: Brief evidence from the code (cite specific lines/calls)
```json
{{
    "hints": [
{{"type": "ALLOCATOR", "target": "return", "arg_index": -1, "reason": "line 5: returns malloc(size) result"}},
{{"type": "ALLOCATOR", "target": "arg0", "arg_index": 0, "reason": "line 6: writes calloc() result to *out parameter"}},
{{"type": "DEALLOCATOR", "target": "arg0", "arg_index": 0, "reason": "line 8: calls free(ptr)"}}
    ]
}}
```

If no memory semantics apply, return: `{{"hints": []}}`

Now analyze the function above and return the JSON result."""



# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Gemini LLM client using service account / ADC (no API key)."""

    def __init__(
        self,
        api_key: str = None,  # kept for compatibility; ignored
        model: str = "gemini-2.5-pro",
        base_url: str = None,  # kept for compatibility; unused
        project_id: str | None = None,
        location: str = "us-central1",
        max_retries: int = 3,
    ):
        """Initialize Gemini client via Vertex AI using ADC / service account.

        Authentication flows:
        - Recommended: set GOOGLE_APPLICATION_CREDENTIALS to the service account JSON.
        - Or run under a GCP environment with default service account.

        Args:
            api_key: Ignored (present for backwards compatibility).
            model: Gemini model name, e.g. "gemini-2.5-pro".
            base_url: Unused for Vertex AI; kept for compatibility.
            project_id: Optional explicit GCP project (fallback when key file missing project_id).
            location: Vertex AI region (default us-central1).
            max_retries: How many retries when calling Gemini.
        """
        self.model_name = model or "gemini-2.5-pro"
        self.location = location
        self.max_retries = max_retries
        self.explicit_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

        # Ensure GOOGLE_APPLICATION_CREDENTIALS is set up front for clearer error early.
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not self.credentials_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS must point to a service account JSON file."
            )
        
        # Initialize Vertex AI once during initialization for better performance
        if VERTEX_AI_AVAILABLE:
            project_id = self.explicit_project
            if not project_id:
                try:
                    with open(self.credentials_path, "r", encoding="utf-8") as f:
                        creds_data = json.load(f)
                    project_id = creds_data.get("project_id")
                except Exception as exc:
                    raise ValueError(
                        f"Failed to read project_id from service account file: {exc}"
                    ) from exc
            
            if not project_id:
                raise ValueError("GCP project_id not found. Set GOOGLE_CLOUD_PROJECT or include in key.")
            
            vertexai.init(project=project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
        else:
            self.model = None
        
        # Cost tracking
        # Pricing per million tokens (approximate, update based on actual Gemini pricing)
        # Default pricing for Gemini models (adjust as needed)
        self.input_price_per_million = 1.25  # $1.25 per million input tokens
        self.output_price_per_million = 10.00  # $10.00 per million output tokens

    def query(self, prompt: str) -> dict:
        """Send query, prefer Vertex AI when service account credentials are configured.
        
        Returns:
            dict with 'content' key containing parsed JSON response and 'usage' key with token usage
        """
        result = self._call_by_vertex_ai(prompt)
        if not result:
            return {"content": {}, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}}
        
        content, usage = result
        if not content:
            return {"content": {}, "usage": usage}
        try:
            parsed = json.loads(content)
            return {"content": parsed, "usage": usage}
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON. Raw content: %s", content[:200])
            return {"content": {}, "usage": usage}

    def _call_by_vertex_ai(self, prompt: str) -> tuple[str | None, dict] | None:
        """Use Vertex AI SDK to call Gemini with service account credentials.
        
        Returns:
            Tuple of (response_text, usage_dict) or None if failed.
            usage_dict contains: prompt_tokens, completion_tokens, total_tokens, cost
        """
        if not VERTEX_AI_AVAILABLE:
            raise ImportError(
                "google-cloud-aiplatform is not installed. "
                "Install it with: pip install google-cloud-aiplatform>=1.38"
            )

        if not self.model:
            raise RuntimeError("Vertex AI model not initialized. Check initialization errors.")

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                    },
                )
                
                # Extract usage metadata
                usage_metadata = getattr(response, "usage_metadata", None)
                prompt_tokens = 0
                completion_tokens = 0
                
                if usage_metadata:
                    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
                
                total_tokens = prompt_tokens + completion_tokens
                
                # Calculate cost
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
                
                return (response.text, usage)
            except Exception as exc:
                logger.warning("Vertex AI call failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        
        # Return empty usage on failure
        return (None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "input_cost": 0.0, "output_cost": 0.0})


# =============================================================================
# Hint Generator
# =============================================================================

class HintGenerator:
    """Generate memory safety hints using LLM with heuristic fallback.

    The generator:
    1. Uses heuristics for known functions (fast, reliable)
    2. Uses LLM for unknown functions (slower, more flexible)
    3. Combines both for comprehensive coverage
    """

    def __init__(self, llm_client: LLMClient = None, codeql_validator=None):
        self.llm = llm_client
        self.codeql_validator = codeql_validator
        # Cost tracking: per function and per hint
        self.function_costs: dict[str, dict] = {}  # function_name -> cost info
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        # Provided by Pipeline (tree-sitter based): typedef alias names that are pointer types.
        self.pointer_typedef_aliases: set[str] = set()
        # Snapshot of last filtering decision for external consumers (e.g., Pipeline)
        self.last_filter_classes: dict[str, list[str]] | None = None

        # Custom CodeQL query generator for "special" functions
        self.custom_query_generator: CustomQueryGenerator | None = None

    def set_pointer_typedef_aliases(self, aliases: set[str] | None) -> None:
        self.pointer_typedef_aliases = set(aliases or set())

    def _has_pointer_io(self, func: FunctionInfo) -> bool:
        """Strict pointer-based IO filtering."""

        # 1) direct pointer in signature
        rt = func.return_type or ""
        if "*" in rt:
            return True

        for t in func.arg_types or []:
            if t and "*" in t:
                return True

        # 2) typedef-based pointer alias
        alias = self.pointer_typedef_aliases

        # return type is alias AND alias defined as pointer typedef
        rt_token = (rt.split()[-1] if rt else None)
        if rt_token in alias:
            return True

        # arg type is alias AND alias defined as pointer typedef
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
    ) -> HintSet:
        """Generate hints for all functions in codebase.

        Args:
            functions: Dict of function name -> FunctionInfo (includes converted macros)
            previous_conflicts: List of conflict messages from previous validation
                              (format: "REMOVED func_name.HintType.name: reason")
            macro_names: Set of function names that are from macros (always included, not filtered)

        Returns:
            HintSet with all generated hints
        """
        hint_set = HintSet()
        macro_names = macro_names or set()
        if pointer_typedef_aliases is not None:
            self.set_pointer_typedef_aliases(pointer_typedef_aliases)

        # Parse conflicts to map function names to their conflict reasons
        conflict_map = {}
        if previous_conflicts:
            for conflict in previous_conflicts:
                # Parse "REMOVED func_name.HintType.name: reason"
                if conflict.startswith("REMOVED "):
                    parts = conflict[8:].split(": ", 1)
                    if len(parts) == 2:
                        func_hint = parts[0]
                        reason = parts[1]
                        # Extract function name (before the dot)
                        if "." in func_hint:
                            func_name = func_hint.split(".")[0]
                            if func_name not in conflict_map:
                                conflict_map[func_name] = []
                            conflict_map[func_name].append(reason)

        # Filter: only send pointer-related functions to the LLM to reduce tokens.
        # Always include macros (they may expand to pointer operations).
        # Additionally, skip entry/test-style functions (main/_main/wmain, *test*)
        # up front so candidate count == actual LLM call count.
        # Keep `functions` as the full codebase so context lookup still works.
        candidate_functions: dict[str, FunctionInfo] = {}
        filtered_entry_functions: list[str] = []   # main/_main/wmain or *test*
        filtered_pointer_functions: list[str] = [] # no pointer IO and not macro
        
        for fn, f in functions.items():
            # Skip main/test functions entirely for hint generation
            if fn in ("main", "_main", "wmain") or "test" in fn.lower():
                filtered_entry_functions.append(fn)
                continue

            if fn in macro_names or self._has_pointer_io(f):
                candidate_functions[fn] = f
            else:
                filtered_pointer_functions.append(fn)

        # Persist filter snapshot for external reporting (Pipeline will use this)
        self.last_filter_classes = {
            "kept": sorted(candidate_functions.keys()),
            "filtered_entry": sorted(filtered_entry_functions),
            "filtered_non_pointer": sorted(filtered_pointer_functions),
        }

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
            
            # Debug: log all filtered and non-filtered functions
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

        # ------------------------------------------------------------------
        # Pre-estimate total LLM prompt tokens and input cost BEFORE calling LLM
        # ------------------------------------------------------------------
        if self.llm and candidate_functions:
            est_prompt_tokens = 0
            est_calls = 0

            for func_name, func in candidate_functions.items():
                # Build approximate prompt text (same structure as _llm_hints)
                context_parts = []
                for callee in list(func.callees)[:5]:
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

                prompt = HINT_GENERATION_PROMPT.format(
                    func_name=func.name,
                    return_type=func.return_type or "void",
                    parameters=params_str,
                    code=func.code,
                    context=context_str + conflict_feedback,
                )

                # Rough token estimate: ~4 characters per token
                est_tokens_this = max(1, len(prompt) // 4)
                est_prompt_tokens += est_tokens_this
                est_calls += 1

            if est_calls > 0:
                ppm = getattr(self.llm, "input_price_per_million", 0.0) or 0.0
                est_input_cost = (est_prompt_tokens / 1_000_000) * ppm
                logger.info(
                    "LLM pre-estimate: about %d prompt tokens across %d call(s), "
                    "estimated input cost ≈ $%.4f (excluding completion tokens)",
                    est_prompt_tokens,
                    est_calls,
                    est_input_cost,
                )

        for func_name, func in tqdm(candidate_functions.items(), desc="Generating hints"):
            # Get conflicts for this function if any
            func_conflicts = conflict_map.get(func_name, [])
            hints = self._generate_for_function(func, functions, func_conflicts)

            for hint in hints:
                hint_set.add(hint)

        return hint_set

    def regenerate_hints_for_functions(
        self,
        functions: dict[str, FunctionInfo],
        conflict_functions: set[str],
        previous_conflicts: list[str] = None,
        macro_names: set[str] = None,
        pointer_typedef_aliases: set[str] | None = None,
    ) -> HintSet:
        """Regenerate hints only for functions that had conflicts.

        Args:
            functions: Dict of all function name -> FunctionInfo
            conflict_functions: Set of function names that had conflicts
            previous_conflicts: List of conflict messages from previous validation
                              (format: "REMOVED func_name.HintType.name: reason")
            macro_names: Set of function names that are from macros (always included, not filtered)

        Returns:
            HintSet with regenerated hints for conflict functions only
        """
        hint_set = HintSet()
        macro_names = macro_names or set()
        if pointer_typedef_aliases is not None:
            self.set_pointer_typedef_aliases(pointer_typedef_aliases)

        # Parse conflicts to map function names to their conflict reasons
        conflict_map = {}
        if previous_conflicts:
            for conflict in previous_conflicts:
                # Parse "REMOVED func_name.HintType.name: reason"
                if conflict.startswith("REMOVED "):
                    parts = conflict[8:].split(": ", 1)
                    if len(parts) == 2:
                        func_hint = parts[0]
                        reason = parts[1]
                        # Extract function name (before the dot)
                        if "." in func_hint:
                            func_name = func_hint.split(".")[0]
                            if func_name not in conflict_map:
                                conflict_map[func_name] = []
                            conflict_map[func_name].append(reason)

        # Pre-estimate tokens/cost for regeneration subset
        if self.llm and conflict_functions:
            est_prompt_tokens = 0
            est_calls = 0

            for func_name in conflict_functions:
                if func_name not in functions:
                    continue

                func = functions[func_name]
                # Respect pointer IO filter for non-macros (same as actual loop)
                if func_name not in macro_names and not self._has_pointer_io(func):
                    continue

                context_parts = []
                for callee in list(func.callees)[:5]:
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

                prompt = HINT_GENERATION_PROMPT.format(
                    func_name=func.name,
                    return_type=func.return_type or "void",
                    parameters=params_str,
                    code=func.code,
                    context=context_str + conflict_feedback,
                )

                est_tokens_this = max(1, len(prompt) // 4)
                est_prompt_tokens += est_tokens_this
                est_calls += 1

            if est_calls > 0:
                ppm = getattr(self.llm, "input_price_per_million", 0.0) or 0.0
                est_input_cost = (est_prompt_tokens / 1_000_000) * ppm
                logger.info(
                    "LLM pre-estimate (regeneration): about %d prompt tokens across %d call(s), "
                    "estimated input cost ≈ $%.4f (excluding completion tokens)",
                    est_prompt_tokens,
                    est_calls,
                    est_input_cost,
                )

        # Only process conflict functions (all of these have already passed the initial LLM filter)
        filtered_conflict_funcs = []
        for func_name in conflict_functions:
            if func_name not in functions:
                continue

            func = functions[func_name]
            # Always include macros. For regular functions, apply pointer-only filter.
            if func_name not in macro_names and not self._has_pointer_io(func):
                filtered_conflict_funcs.append(func_name)
                continue
            
            # Get conflicts for this function if any
            func_conflicts = conflict_map.get(func_name, [])
            hints = self._generate_for_function(func, functions, func_conflicts)

            for hint in hints:
                hint_set.add(hint)
        
        # Debug: log filtered conflict functions
        if filtered_conflict_funcs and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Conflict functions FILTERED during regeneration (skipped):")
            for fn in sorted(filtered_conflict_funcs):
                logger.debug(f"  - {fn}")

        return hint_set

    def _generate_for_function(
        self,
        func: FunctionInfo,
        all_functions: dict[str, FunctionInfo],
        previous_conflicts: list[str] = None,
    ) -> list[Hint]:
        """Generate hints for a single function.

        Args:
            func: Function to analyze
            all_functions: All functions in codebase
            previous_conflicts: List of conflict reasons for this function from previous iteration
        """
        hints: list[Hint] = []

        #LLM-based hints, merged without duplicates.
        llm_hints = self._llm_hints(func, all_functions, previous_conflicts)
        for llm_hint in llm_hints:
            existing = next(
                (h for h in hints if h.hint_type == llm_hint.hint_type and h.target == llm_hint.target),
                None,
            )
            if not existing:
                hints.append(llm_hint)

        return hints

    def _llm_hints(
        self,
        func: FunctionInfo,
        all_functions: dict[str, FunctionInfo],
        previous_conflicts: list[str] = None,
    ) -> list[Hint]:
        """Generate hints using LLM.

        Args:
            func: Function to analyze
            all_functions: All functions in codebase
            previous_conflicts: List of conflict reasons from previous validation iteration
        """
        if not self.llm:
            return []

        context_parts = []
        for callee in list(func.callees)[:5]:
            if callee in all_functions:
                callee_code = all_functions[callee].code
                context_parts.append(f"// Called function:\n{callee_code}")

        context = "\n\n".join(context_parts) if context_parts else ""
        context_str = f"Context (called functions):\n{context}" if context else ""

        # Add Z3 validation feedback if available
        conflict_feedback = ""
        if previous_conflicts:
            conflict_feedback = f"""
## Z3 Validation Feedback

The following hints were **rejected** by Z3's constraint solving and path feasibility analysis:

{chr(10).join(f"- {c}" for c in previous_conflicts)}

The rejected hints are inconsistent with the code structure. Please:
1. Re-examine if the memory operation actually occurs in this function
2. Only re-submit hints you are confident are correct
"""
        params = list(zip(func.arg_types, func.arg_names))
        params_str = ", ".join(f"{t} {n}" for t, n in params) if params else "void"

        prompt = HINT_GENERATION_PROMPT.format(
            func_name=func.name,
            return_type=func.return_type or "void",
            parameters=params_str,
            code=func.code,
            context=context_str + conflict_feedback,
        )

        # Log the complete prompt for debugging
        logger.debug(f"LLM prompt for function {func.name}:\n{'='*80}\n{prompt}\n{'='*80}")

        result = self.llm.query(prompt)
        usage = result.get("usage", {})
        content = result.get("content", {})
        
        # Track cost for this function
        func_cost = usage.get("cost", 0.0)
        func_tokens = usage.get("total_tokens", 0)
        self.total_cost += func_cost
        self.total_tokens += func_tokens
        
        self.function_costs[func.name] = {
            "cost": func_cost,
            "tokens": func_tokens,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "input_cost": usage.get("input_cost", 0.0),
            "output_cost": usage.get("output_cost", 0.0),
        }
        
        hints = []

        for h in content.get("hints", []):
            hint_type_str = h.get("type", "")

            try:
                hint_type = HintType[hint_type_str]
            except KeyError:
                continue

            # Extract arg_semantics (convert string keys to int keys)
            arg_semantics_raw = h.get("arg_semantics", {})
            arg_semantics = {}
            if isinstance(arg_semantics_raw, dict):
                for k, v in arg_semantics_raw.items():
                    try:
                        arg_idx = int(k)
                        arg_semantics[arg_idx] = str(v)
                    except (ValueError, TypeError):
                        pass

            hints.append(Hint(
                function_name=func.name,
                hint_type=hint_type,
                target=h.get("target", "return"),
                arg_index=h.get("arg_index", -1),
                reason=h.get("reason", "LLM analysis"),
                arg_semantics=arg_semantics,
            ))

        return hints
    
    def get_cost_summary(self) -> dict:
        """Get cost summary with per-function and per-hint breakdowns.
        
        Returns:
            dict with total costs, per-function costs, and per-hint costs
        """
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
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
        """
        Initialize CodeQL validator.
        
        Args:
            codeql_binary: Path to CodeQL binary (default: "codeql")
            codeql_dir: Optional path to CodeQL installation directory
            database_path: Optional path to CodeQL database (if available, can improve validation)
            timeout: Timeout in seconds for validation (default: 30)
        """
        self.codeql_binary = codeql_binary
        self.codeql_dir = codeql_dir
        self.database_path = database_path
        self.timeout = timeout
        
        # Check if CodeQL is available
        if not which(codeql_binary):
            raise ValueError(
                f"CodeQL binary '{codeql_binary}' not found in PATH. "
                "Please install CodeQL or provide the correct path."
            )
    
    def validate_query(self, query_code: str) -> Tuple[bool, str]:
        if not query_code or not query_code.strip():
            return False, "Query code is empty"

        query_dir = Path("/home/huihuihuang/Hint/codeql/qlpacks/codeql/cpp-queries/1.5.8/Critical")
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
        """
        Initialize the query generator.
        
        Args:
            llm: Object with method query(prompt: str) -> dict
                 Expected dict keys:
                 - "content": dict with:
                       is_special: bool
                       reason: str
                       double_free_filter: {predicates_code, use_expr}
                       use_after_free_filter: {predicates_code, use_expr}
                       memory_never_freed_filter: {predicates_code, use_expr}
                       memory_may_not_be_freed_filter: {predicates_code, use_expr}
                 - "usage": dict with cost/token breakdown (optional)
            codeql_validator: Optional validator object (if None, will try to create one automatically)
        """
        self.llm = llm
        self.codeql_validator = codeql_validator
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.function_costs: Dict[str, Dict[str, float]] = {}
        
        # Auto-create CodeQL validator if not provided and CodeQL is available
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
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def generate_custom_query_for_function(
        self,
        func: Any,
        hints: List[Any],
    ) -> Optional[CustomQuery]:
        """
        Generate a custom query for a function if it has special semantics.
        
        Args:
            func: FunctionInfo object containing function metadata
            hints: List of Hint objects with memory safety information
        
        Returns:
            CustomQuery if the LLM marks it as special, else a CustomQuery with empty code
        """
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
        
        # Log the special status and reason
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

        # New-style per-bug-type filters
        df_block = content.get("double_free_filter") or {}
        uaf_block = content.get("use_after_free_filter") or {}
        never_freed_block = content.get("memory_never_freed_filter") or {}
        may_not_be_freed_block = content.get("memory_may_not_be_freed_filter") or {}

        # Decide if we have any new-style filter content
        def _has_filter_block(block: Any) -> bool:
            return isinstance(block, dict) and any(
                (block.get("predicates_code") or "").strip()
                or (block.get("use_expr") or "").strip()
            )

        has_new_filters = any(_has_filter_block(b) for b in (df_block, uaf_block, never_freed_block, may_not_be_freed_block))

        validated = False
        validation_error = ""

        if is_special and has_new_filters:
            # New behavior: validate each filter type separately at generation time
            func_name = getattr(func, "name", "")
            
            # Validate and fix double_free_filter
            if _has_filter_block(df_block):
                df_block = self._validate_and_fix_filter_block(
                    func_name=func_name,
                    filter_type="double_free_filter",
                    block=df_block,
                    original_prompt=prompt,
                )
            
            # Validate and fix use_after_free_filter
            if _has_filter_block(uaf_block):
                uaf_block = self._validate_and_fix_filter_block(
                    func_name=func_name,
                    filter_type="use_after_free_filter",
                    block=uaf_block,
                    original_prompt=prompt,
                )
            
            # Validate and fix memory_never_freed_filter
            if _has_filter_block(never_freed_block):
                never_freed_block = self._validate_and_fix_filter_block(
                    func_name=func_name,
                    filter_type="memory_never_freed_filter",
                    block=never_freed_block,
                    original_prompt=prompt,
                )
            
            # Validate and fix memory_may_not_be_freed_filter
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
            double_free_filter=df_block if is_special else {},
            use_after_free_filter=uaf_block if is_special else {},
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
    
    # -------------------------------------------------------------------------
    # Formatting helpers
    # -------------------------------------------------------------------------
    
    def _format_hints_and_mad(
        self,
        func: Any,
        hints: List[Any]
    ) -> Tuple[str, str]:
        """
        Convert hints and MAD (Memory Access Description) into readable text blocks.
        
        Args:
            func: Function object
            hints: List of hint objects
            
        Returns:
            Tuple of (hints_text, arg_semantics_text)
        """
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
    
    # -------------------------------------------------------------------------
    # Prompt engineering
    # -------------------------------------------------------------------------
    
    def _build_query_generation_prompt(
        self,
        func: Any,
        hints_text: str,
        arg_semantics_text: str,
        hints: Optional[List[Any]] = None
    ) -> str:
        """
        Build the LLM prompt for deciding specialness and generating a query.
        
        This prompt uses a MAD-driven approach to guide the LLM in creating
        targeted CodeQL queries based on function semantics.
        """
        name = getattr(func, "name", "")
        code = getattr(func, "code", "")
        ret = getattr(func, "return_type", "") or "void"
        arg_names = getattr(func, "arg_names", []) or []
        arg_types = getattr(func, "arg_types", []) or []
        
        params_str = ", ".join(
            [f"{t} {n}" for t, n in zip(arg_types, arg_names)]
        ) if arg_names else "void"
        
        # Extract allocator/deallocator information from hints
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
        
        # Build semantic description
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
  - double-free: (srcDealloc, sinkNode, sinkFreedExpr)
  - UAF: (dealloc, sinkNode)
  - never-freed: (alloc)
  - may-not-be-freed: (def)
- NEVER output use_expr containing free variables like call1/call2.
- NEVER redefine dfFiltered/uafFiltered/leakFiltered/mayNotBeFreedFiltered (aggregators are owned by the main query).

==================================================
HARD BAN LIST (COMPILATION FAILURES)
==================================================

- getExpr()
- getASubExpression()
- getEnclosingFunctionCall()
- Casting FunctionCall to DeallocationExpr
- Redefining aggregator predicates
- Any use_expr that references undefined variables (e.g., call1/call2; or srcDealloc/sinkFreedExpr inside UAF)
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
HOW TO BIND SINK DEALLOCATION (DOUBLE FREE ONLY)
==================================================

In a double-free plugin you may need sink-side DeallocationExpr.
The main query typically provides isFree(...) already (do NOT redefine it).
To bind sink DeallocationExpr without using '_' placeholders, use:

exists(DeallocationExpr sinkDealloc, DataFlow::Node dummy0 |
  isFree(dummy0, sinkNode, sinkFreedExpr, sinkDealloc)
  ...
)

==================================================
EXAMPLE WHEN GENERATING A DOUBLE FREE FILTER
(SINGLE-PREDICATE PLUGIN, NO HELPER PREDICATES)
==================================================

predicate dfFilterExample_{name}(DeallocationExpr srcDealloc, DataFlow::Node sinkNode, Expr sinkFreedExpr) {{{{
  exists(DeallocationExpr sinkDealloc, FunctionCall c1, FunctionCall c2, Expr p1, Expr p2, Expr idx1, Expr idx2, DataFlow::Node dummy0 |
    isFree(dummy0, sinkNode, sinkFreedExpr, sinkDealloc) and

    c1 = srcDealloc and
    c2 = sinkDealloc and
    c1.getTarget().getName() = "{name}" and
    c2.getTarget().getName() = "{name}" and

    // same pointer, different constant index => safe pair
    p1 = c1.getArgument(0) and
    p2 = c2.getArgument(0) and
    idx1 = c1.getArgument(1) and
    idx2 = c2.getArgument(1) and
    p1.toString() = p2.toString() and
    idx1.isConstant() and idx2.isConstant() and
    idx1.getValueText() != idx2.getValueText()
  )
}}}}

use_expr example:
dfFilterExample_{name}(srcDealloc, sinkNode, sinkFreedExpr)

==================================================
EXAMPLE WHEN GENERATING A USE AFTER FREE FILTER
(SINGLE-PREDICATE PLUGIN, NO HELPER PREDICATES)
==================================================

predicate uafFilterExample_{name}(DeallocationExpr dealloc, DataFlow::Node sinkNode) {{{{
  exists(Expr use, FunctionCall freeCall, Expr freedPtr |
    // main query uses sinkNode.asExpr(), so plugin may use it too
    sinkNode.asExpr() = use and

    freeCall = dealloc and
    freeCall.getTarget().getName() = "{name}" and
    freedPtr = freeCall.getArgument(0) and

    // Example safe-suppression condition: the sink "use" is not about the freed pointer
    not (
      exists(Variable v |
        freedPtr.(VariableAccess).getTarget() = v and
        use.(VariableAccess).getTarget() = v
      )
      or
      freedPtr.toString() = use.toString()
    )
  )
}}}}

use_expr example:
uafFilterExample_{name}(dealloc, sinkNode)

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

  "double_free_filter": {{{{
    "required_imports": ["module.path.IfNeeded"],
    "predicates_code": "ONLY predicate definitions (no imports, no select, no metadata, no aggregator). Empty string if not needed.",
    "use_expr": "Single boolean expression or empty string"
  }}}},

  "use_after_free_filter": {{{{
    "required_imports": ["module.path.IfNeeded"],
    "predicates_code": "ONLY predicate definitions or empty string",
    "use_expr": "Single boolean expression or empty string"
  }}}},

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
- Ensure you did NOT redefine dfFiltered/uafFiltered/leakFiltered/mayNotBeFreedFiltered.
- Ensure your use_expr contains ONLY the pipeline parameters.
- Ensure any local variables are bound inside exists(...).
- Ensure required_imports lists ONLY modules actually used.
- Ensure you did NOT include "import" keyword in required_imports.
- Ensure JSON is valid and contains only the required fields.


Now analyze the function and respond with JSON only.
'''
        
        return PROMPT
    
    # -------------------------------------------------------------------------
    # Validation and fixing
    # -------------------------------------------------------------------------
    
    def _insert_imports_into_template(self, template: str, required_imports: set[str]) -> str:
        """
        Insert additional imports into a CodeQL query template.
        
        Args:
            template: The CodeQL query template string
            required_imports: Set of import module paths (without 'import' keyword)
        
        Returns:
            Template with imports inserted after existing imports
        """
        if not required_imports:
            return template
        
        # Sort imports for consistent output
        import_lines = sorted(required_imports)
        additional_imports = "\n".join(f"import {imp}" for imp in import_lines)
        
        # Find the last import statement in the template
        # Look for common import patterns
        last_import_pos = -1
        last_import_end = -1
        
        # Try to find the last import line
        import_patterns = [
            "import semmle.code.cpp.dataflow.new.DataFlow",
            "import semmle.code.cpp.controlflow.StackVariableReachability",
            "import MemoryFreed",
            "import cpp",
        ]
        
        for pattern in import_patterns:
            pos = template.rfind(pattern)
            if pos != -1:
                # Find the end of this import line
                end_pos = template.find("\n", pos)
                if end_pos != -1:
                    if pos > last_import_pos:
                        last_import_pos = pos
                        last_import_end = end_pos
        
        if last_import_pos != -1 and last_import_end != -1:
            # Find the blank line after imports (if any)
            blank_line_pos = template.find("\n\n", last_import_end)
            if blank_line_pos != -1:
                # Insert after the blank line
                return (
                    template[:blank_line_pos + 1] +
                    additional_imports + "\n" +
                    template[blank_line_pos + 1:]
                )
            else:
                # No blank line, insert after the last import with a newline
                return (
                    template[:last_import_end] +
                    "\n" + additional_imports + "\n" +
                    template[last_import_end:]
                )
        
        # Fallback: prepend imports (shouldn't happen with proper templates)
        return additional_imports + "\n\n" + template

    def _build_filter_wrapper_query(
        self,
        filter_type: str,
        predicates_code: str,
        use_expr: str,
        required_imports: Optional[set[str]] = None,
    ) -> str:
        """
        Build a validation query by inserting the filter into the actual production template.
        
        This mirrors how adapters.py merges filters into templates, ensuring validation
        happens with the exact same structure that will be used in production.
        """
        if required_imports is None:
            required_imports = set()
        
        if filter_type == "double_free_filter":
            template = ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE
            # Merge predicates_code and build dfFiltered aggregator (same as adapters.py)
            preds_part = predicates_code if predicates_code else ""
            body = use_expr if use_expr else "false"
            df_agg = (
                "predicate dfFiltered(DeallocationExpr srcDealloc, "
                "DataFlow::Node sinkNode, Expr sinkFreedExpr) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = template
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + df_agg
            return merged
            
        elif filter_type == "use_after_free_filter":
            template = ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE
            # Merge predicates_code and build uafFiltered aggregator
            preds_part = predicates_code if predicates_code else ""
            body = use_expr if use_expr else "false"
            uaf_agg = (
                "predicate uafFiltered(DeallocationExpr dealloc, DataFlow::Node sinkNode) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = template
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + uaf_agg
            return merged
            
        elif filter_type == "memory_never_freed_filter":
            # Memory never freed filter (uses leakFiltered(AllocationExpr alloc))
            template = ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE
            preds_part = predicates_code if predicates_code else ""
            body = use_expr if use_expr else "false"
            leak_agg = (
                "predicate leakFiltered(AllocationExpr alloc) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = template
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + leak_agg
            return merged
            
        elif filter_type == "memory_may_not_be_freed_filter":
            # Memory may not be freed filter (uses mayNotBeFreedFiltered(ControlFlowNode def))
            template = ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE
            preds_part = predicates_code if predicates_code else ""
            body = use_expr if use_expr else "false"
            may_not_be_freed_agg = (
                "predicate mayNotBeFreedFiltered(ControlFlowNode def) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = template
            # Insert additional imports if any
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
        """
        Validate a filter block (predicates_code + use_expr) and fix if needed.
        
        Returns:
            Updated block dict with validated/corrected predicates_code and use_expr.
        """
        predicates_code = (block.get("predicates_code") or "").strip()
        use_expr = (block.get("use_expr") or "").strip()
        
        # Collect required imports (use set to avoid duplicates)
        required_imports: set[str] = set()
        imports = block.get("required_imports", [])
        if isinstance(imports, list):
            for imp in imports:
                if isinstance(imp, str) and imp.strip():
                    required_imports.add(imp.strip())
        
        if not predicates_code and not use_expr:
            # Empty block, nothing to validate
            return block
        
        # Build wrapper query for validation
        wrapper_query = self._build_filter_wrapper_query(filter_type, predicates_code, use_expr, required_imports)
        
        # Log the full query being validated
        logger.info(
            "Validating filter block '%s' for '%s' with merged query:\n%s",
            filter_type,
            func_name,
            wrapper_query
        )
        
        # Validate the wrapper query
        is_valid, error_msg = self._validate_codeql_query(wrapper_query)
        
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
        
        # Try to fix with LLM (up to 2 attempts)
        for attempt in range(2):
            # Add specific guidance for memory_may_not_be_freed_filter errors
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
                logger.warning("LLM did not return corrected filter code in attempt %d/2", attempt + 1)
                continue
            
            # Rebuild wrapper and validate again (preserve required_imports from original block)
            new_wrapper = self._build_filter_wrapper_query(filter_type, new_predicates, new_use_expr, required_imports)
            
            # Log the fixed query being validated
            logger.info(
                "Validating fixed filter block '%s' for '%s' (attempt %d/2) with merged query:\n%s",
                filter_type,
                func_name,
                attempt + 1,
                new_wrapper
            )
            
            is_valid, error_msg = self._validate_codeql_query(new_wrapper)
            
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
                "Fix attempt %d/2 for '%s' still has error: %s",
                attempt + 1,
                filter_type,
                error_msg[:200]
            )
        
        # All attempts failed
        logger.error(
            "Failed to fix filter block '%s' for '%s' after 3 attempts. Last error: %s",
            filter_type,
            func_name,
            error_msg[:200]
        )
        return {
            **block,
            "validated": False,
            "validation_error": f"Validation failed after 3 attempts: {error_msg[:200]}",
        }
    
    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------
    
    def _quick_lint_codeql_query(self, query_code: str) -> Tuple[bool, str]:
        """
        Perform quick syntactic checks on the query before full validation.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query_code or not query_code.strip():
            return False, "Query code is empty"
        
        # Check for metadata block
        if "/**" not in query_code:
            return False, "Missing metadata comment block (/** ... */)"
        
        # Check required metadata tags
        required_tags = ["@name", "@description", "@kind", "@id"]
        for tag in required_tags:
            if tag not in query_code:
                return False, f"Missing required metadata tag: {tag}"
        
        # Check for cpp import
        if "import cpp" not in query_code:
            return False, "Missing 'import cpp' statement"
        
        # Check basic query structure
        has_from = "from " in query_code
        has_where = "where " in query_code or "where\n" in query_code
        has_select = "select " in query_code
        
        if not (has_from and has_where and has_select):
            return False, "Missing required query structure (from/where/select)"
        
        # Check for forbidden imports
        forbidden = [
            "semmle.code.cpp.dataflow",
            "DataFlow",
            "TaintTracking",
            "FlowAfterFree",
        ]
        for bad in forbidden:
            if bad in query_code:
                return False, f"Forbidden import or library detected: {bad}"
        
        return True, ""
    
    def _validate_codeql_query(self, query_code: str) -> Tuple[bool, str]:
        """
        Validate the query using the CodeQL validator if available.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.codeql_validator:
            # No validator configured -> perform enhanced quick validation
            # This catches common CodeQL syntax errors that quick lint misses
            return self._enhanced_quick_validation(query_code)
        
        return self.codeql_validator.validate_query(query_code)
    
    @staticmethod
    def create_codeql_validator(codeql_binary: str = "codeql", codeql_dir: Path | None = None) -> Optional[Any]:
        """
        Create a CodeQL validator that uses the CodeQL CLI to compile queries.
        
        Args:
            codeql_binary: Path to CodeQL binary (default: "codeql")
            codeql_dir: Optional path to CodeQL installation directory
        
        Returns:
            CodeQLValidator instance if CodeQL is available, None otherwise
        """
        try:
            return CodeQLValidator(codeql_binary=codeql_binary, codeql_dir=codeql_dir)
        except Exception as e:
            logger.warning(f"Failed to create CodeQL validator: {e}")
            return None
    
    def _enhanced_quick_validation(self, query_code: str) -> Tuple[bool, str]:
        """
        Enhanced validation that catches common CodeQL syntax errors when
        no full validator is available.
        
        This addresses the issue where validation was completely skipped when
        codeql_validator was None. Previously, _validate_codeql_query would
        return True, "" without any checks, allowing invalid queries to pass.
        
        This enhanced validation catches:
        - hasGlobalOrInstantiatedName() misuse on Function type
        - Common method name errors (getFunctionName, getMethodName)
        - Incorrect FunctionCall usage (calling methods directly instead of via getTarget())
        - Missing imports
        
        Note: This is better than nothing, but a real CodeQL validator that
        can compile queries is preferred for comprehensive validation.
        """
        # Check for common CodeQL API misuse patterns
        # Pattern 1: hasGlobalOrInstantiatedName on wrong type
        # This method doesn't exist on Function, only on Variable, etc.
        if "hasGlobalOrInstantiatedName" in query_code:
            # Check if it's being called on Function (common mistake)
            lines = query_code.split('\n')
            for i, line in enumerate(lines, 1):
                if "hasGlobalOrInstantiatedName" in line and ("Function" in line or ".getTarget()" in line):
                    return False, (
                        f"Line {i}: hasGlobalOrInstantiatedName() cannot be used on Function type. "
                        "Use getName() or getQualifiedName() instead. "
                        "Example: call.getTarget().getName() = \"function_name\""
                    )
        
        # Pattern 1b: hasGlobalOrStdName might not exist on Function either
        # Check if it's being used incorrectly
        if "hasGlobalOrStdName" in query_code:
            lines = query_code.split('\n')
            for i, line in enumerate(lines, 1):
                if "hasGlobalOrStdName" in line and (".getTarget()" in line or "Function" in line):
                    # This might be valid, but if it fails, suggest alternatives
                    # We'll let it pass here but note it might need getName() instead
                    pass  # Allow it, but the actual CodeQL compiler will catch if invalid
        
        # Pattern 2: Common method name errors
        method_errors = {
            "getFunctionName()": "Function doesn't have getFunctionName(). Use getName() or getQualifiedName()",
            "getMethodName()": "Function doesn't have getMethodName(). Use getName() or getQualifiedName()",
        }
        for bad_method, suggestion in method_errors.items():
            if bad_method in query_code:
                return False, f"Invalid method: {bad_method}. {suggestion}"
        
        # Pattern 3: Check for proper FunctionCall usage
        # If using FunctionCall, should use getTarget() to get Function
        if "FunctionCall" in query_code and "getTarget()" not in query_code:
            # This might be okay, but warn if they're trying to call methods directly
            if ".getName()" in query_code or ".getQualifiedName()" in query_code:
                # Check if it's being called on FunctionCall directly (wrong)
                lines = query_code.split('\n')
                for i, line in enumerate(lines, 1):
                    if "FunctionCall" in line and any(m in line for m in [".getName()", ".getQualifiedName()"]):
                        if "getTarget()" not in line:
                            return False, (
                                f"Line {i}: Cannot call getName()/getQualifiedName() directly on FunctionCall. "
                                "Use call.getTarget().getName() instead."
                            )
        
        # Pattern 4: Check for proper Literal usage
        # getValue() returns a string, should compare with strings
        if "Literal" in query_code and "getValue()" in query_code:
            # Check if comparing with non-string (common mistake)
            if "getValue() != " in query_code or "getValue() = " in query_code:
                # This is usually okay, but check for obvious type mismatches
                pass  # Hard to detect without full parsing
        
        # Pattern 5: Check for missing imports that are commonly needed
        # If using certain types, need specific imports
        if "FunctionCall" in query_code and "import cpp" not in query_code:
            return False, "Missing 'import cpp' - required for FunctionCall"
        
        # All checks passed
        return True, ""
    
    # -------------------------------------------------------------------------
    # Cost tracking
    # -------------------------------------------------------------------------
    
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