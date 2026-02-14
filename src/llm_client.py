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

        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Service account key file not found: {self.credentials_path}\n"
                "Please check the path and try again."
            )

        project_id = self.explicit_project
        if not project_id:
            try:
                with open(self.credentials_path, "r", encoding="utf-8") as f:
                    creds_data = json.load(f)
                project_id = creds_data.get("project_id")
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"Failed to read project_id from service account file: {exc}"
                ) from exc

        if not project_id:
            raise ValueError("GCP project_id not found. Set GOOGLE_CLOUD_PROJECT or include in key.")

        # Initialize Vertex AI for every call to ensure fresh config if env changes.
        vertexai.init(project=project_id, location=self.location)
        model = GenerativeModel(self.model_name)

        for attempt in range(self.max_retries):
            try:
                response = model.generate_content(
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
        """
        Validate a CodeQL query by writing it to a temp file and running codeql query compile.
        
        Args:
            query_code: The CodeQL query code to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query_code or not query_code.strip():
            return False, "Query code is empty"
        
        # Create a temporary directory for the query file (needed for qlpack context)
        with tempfile.TemporaryDirectory() as temp_dir:
            query_dir = Path(temp_dir)
            query_file = query_dir / "query.ql"
            
            try:
                # Write query to file
                query_file.write_text(query_code)
                
                # Create qlpack.yml with required library dependencies
                qlpack_file = query_dir / "qlpack.yml"
                qlpack_content = """name: temp-query-pack
version: 1.0.0
libraryPathDependencies: codeql/cpp-all
"""
                qlpack_file.write_text(qlpack_content)
                
                # Run codeql query compile to validate the query
                # Note: query compile validates syntax without needing a database
                # The libraryPathDependencies in qlpack.yml provides the dbscheme
                cmd = [self.codeql_binary, "query", "compile", "--check-only", str(query_file)]
                
                # Set environment if codeql_dir is provided
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
                else:
                    # Return the error message - LLM will use this to fix the query
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    if not error_msg:
                        error_msg = f"CodeQL query compile failed with return code {result.returncode}"
                    
                    # Clean up error message - remove temp file paths
                    error_msg = error_msg.replace(str(query_file), "query.ql")
                    error_msg = error_msg.replace(str(query_dir), "")
                    
                    return False, error_msg
                    
            except Exception as e:
                return False, f"Validation error: {str(e)}"


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
                       double_free_filter: {predicates_code, use_expr, query_code?}
                       use_after_free_filter: {predicates_code, use_expr, query_code?}
                       memory_never_freed_filter: {predicates_code, use_expr, query_code?}
                       memory_may_not_be_freed_filter: {predicates_code, use_expr, query_code?}
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
        prompt = self._build_query_generation_prompt(func, hints_text, arg_semantics_text)
        
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

        # New-style per-bug-type filters
        df_block = content.get("double_free_filter") or {}
        uaf_block = content.get("use_after_free_filter") or {}
        never_freed_block = content.get("memory_never_freed_filter") or {}
        may_not_be_freed_block = content.get("memory_may_not_be_freed_filter") or {}

        # Legacy single-query mode (query_code) – kept for backward compatibility.
        legacy_query_code = (content.get("query_code", "") or "").strip()

        # Decide if we have any new-style filter content
        def _has_filter_block(block: Any) -> bool:
            return isinstance(block, dict) and any(
                (block.get("predicates_code") or "").strip()
                or (block.get("use_expr") or "").strip()
                or (block.get("query_code") or "").strip()
            )

        has_new_filters = any(_has_filter_block(b) for b in (df_block, uaf_block, never_freed_block, may_not_be_freed_block))

        validated = False
        validation_error = ""
        query_code = ""

        if is_special and not has_new_filters and legacy_query_code:
            # Old behavior: validate a full custom query_code.
            query_code, is_special, reason, validated, validation_error = self._validate_and_fix_query(
                func_name=getattr(func, "name", ""),
                query_code=legacy_query_code,
                reason=reason,
                original_prompt=prompt,
            )
        elif is_special and has_new_filters:
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
            query_code=query_code if is_special else "",
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
        arg_semantics_text: str
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
        
        PROMPT = f'''You are an expert CodeQL query generator specializing in C/C++ memory safety analysis.

Your task has TWO parts:
1. Determine if this function has SPECIAL SEMANTICS that cause false positive warnings
2. If special, generate FILTER PLUGIN PREDICATES (not full queries) to suppress safe patterns

IMPORTANT:
- You are generating ONLY predicate definitions that will be inserted into an existing CodeQL query pack.
- DO NOT generate full queries.
- DO NOT generate metadata headers (no @name/@id/...).
- DO NOT generate import statements.
- DO NOT generate select clauses.
- DO NOT redefine aggregator predicates like dfFiltered/uafFiltered/leakFiltered/mayNotBeFreedFiltered.

==================================================
FUNCTION ANALYSIS
==================================================

Function Name: `{name}`
Return Type: `{ret}`
Parameters: `{params_str}`

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

Note: If the MAIN query already uses sinkNode.asExpr(), you may also use sinkNode.asExpr() in plugins.
Otherwise, do not invent new conversions.

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

Example:

predicate leakMayNotBeFreedFilterExample_{name}(ControlFlowNode def) {{{{
  exists(Expr e, FunctionCall c, Assignment a |
    // bind the definition node to an expression that is a call to the special function {name}
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
  "reason": "Clear explanation of why this function is/isn't special, based on the actual code and hints",

  "double_free_filter": {{{{
    "predicates_code": "ONLY predicate definitions for dfFiltered(...) plugins (no imports, no select, no metadata, no aggregator). Empty string if not needed.",
    "use_expr": "A single boolean expression like dfFilterXxx(srcDealloc, sinkNode, sinkFreedExpr), or empty string",
    "query_code": ""
  }}}},

  "use_after_free_filter": {{{{
    "predicates_code": "ONLY predicate definitions for uafFiltered(...) plugins. Empty string if not needed.",
    "use_expr": "A single boolean expression like uafFilterXxx(dealloc, sinkNode), or empty string",
    "query_code": ""
  }}}},

  "memory_never_freed_filter": {{{{
    "predicates_code": "ONLY predicate definitions for leakFiltered(alloc) plugins. Empty string if not needed.",
    "use_expr": "A single boolean expression like leakFilterXxx(alloc), or empty string",
    "query_code": ""
  }}}},

  "memory_may_not_be_freed_filter": {{{{
    "predicates_code": "ONLY predicate definitions for mayNotBeFreedFiltered(def) plugins. Empty string if not needed.",
    "use_expr": "A single boolean expression like mayNotBeFreedFilterXxx(def), or empty string",
    "query_code": ""
  }}}}
}}}}

GUIDELINES:
- If the function is not special, set is_special=false and leave all predicates_code/use_expr as empty strings.
- If special only for some bug types, fill only those blocks.
- For each non-empty block:
  - predicates_code MUST contain ONLY predicate definitions (no imports, no select, no metadata, no aggregator).
  - use_expr MUST call your predicate using ONLY in-scope pipeline parameters.
  - query_code MUST be "".

==================================================
SELF-CHECK BEFORE RETURNING JSON
==================================================

Before output:
- Ensure NO getExpr/getASubExpression/getEnclosingFunctionCall appears.
- Ensure you did NOT redefine dfFiltered/uafFiltered/leakFiltered/mayNotBeFreedFiltered.
- Ensure your use_expr contains ONLY the pipeline parameters (no call1/call2).
- Ensure any local variables are bound within exists(...).
- Ensure JSON is valid and contains only the required fields.

Now analyze the function and respond with JSON only.
'''
        
        return PROMPT
    
    # -------------------------------------------------------------------------
    # Validation and fixing
    # -------------------------------------------------------------------------
    
    def _build_filter_wrapper_query(
        self,
        filter_type: str,
        predicates_code: str,
        use_expr: str,
    ) -> str:
        """
        Build a validation query by inserting the filter into the actual production template.
        
        This mirrors how adapters.py merges filters into templates, ensuring validation
        happens with the exact same structure that will be used in production.
        """
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
        
        if not predicates_code and not use_expr:
            # Empty block, nothing to validate
            return block
        
        # Build wrapper query for validation
        wrapper_query = self._build_filter_wrapper_query(filter_type, predicates_code, use_expr)
        
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
            fix_prompt = f"""You previously generated a CodeQL filter for the function '{func_name}' (filter type: {filter_type}), but it has a compilation error when inserted into the production template.

**ERROR:**
{error_msg[:500]}

**CURRENT FILTER CODE:**
```codeql
{predicates_code}
```

**USE EXPRESSION:**
```codeql
{use_expr}
```

**FULL QUERY (production template with your filter inserted):**
```codeql
{wrapper_query[:1500]}...
```

Please fix ONLY the predicates_code and/or use_expr to resolve the compilation error. The fix must work when inserted into the production template shown above. Return the corrected filter in JSON format:
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
            
            # Rebuild wrapper and validate again
            new_wrapper = self._build_filter_wrapper_query(filter_type, new_predicates, new_use_expr)
            
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
    
    def _validate_and_fix_query(
        self,
        func_name: str,
        query_code: str,
        reason: str,
        original_prompt: str,
    ) -> Tuple[str, bool, str, bool, str]:
        """
        Validate generated CodeQL query and attempt to fix errors.
        
        Returns:
            Tuple of (query_code, is_special, reason, validated, validation_error)
            - validated: True if query passed validation, False otherwise
            - validation_error: Error message if validation failed, empty string if passed
        """
        # Quick lint check first
        ok, lint_msg = self._quick_lint_codeql_query(query_code)
        if not ok:
            logger.warning(
                "Generated query for '%s' failed quick lint: %s",
                func_name,
                lint_msg
            )
            is_valid = False
            error_msg = lint_msg
        else:
            is_valid, error_msg = self._validate_codeql_query(query_code)
            if not is_valid:
                logger.warning(
                    "Generated query for '%s' failed validation: %s",
                    func_name,
                    error_msg
                )
            else:
                logger.info("Generated query for '%s' passed CodeQL validation", func_name)
        
        if is_valid:
            return query_code, True, reason, True, ""
        
        logger.info(
            "Attempting to fix the query for '%s' with LLM assistance (error: %s)...",
            func_name,
            error_msg[:100]  # Truncate long error messages
        )
        
        # Try up to 2 fix attempts (3 total attempts)
        for attempt in range(2):
            fix_prompt = self._build_fix_prompt(error_msg, query_code)
            
            # Build a focused fix prompt that includes the error and current code
            focused_prompt = f"""You previously generated a CodeQL query, but it has an error.

**ERROR:**
{error_msg}

**CURRENT QUERY CODE (with error):**
```codeql
{query_code}
```

Please fix ONLY the error above. Return the corrected query in JSON format:
{{
  "is_special": true,
  "reason": "Fixed: {error_msg[:50]}...",
  "query_code": "corrected CodeQL query here"
}}"""
            
            result = self.llm.query(focused_prompt) or {}
            content = result.get("content", {}) or {}
            
            new_is_special = bool(content.get("is_special", True))
            new_code = (content.get("query_code", "") or "").strip()
            new_reason = (content.get("reason", reason) or reason).strip()
            
            if not (new_is_special and new_code):
                error_msg = "LLM did not return a valid special query in retry."
                logger.warning(error_msg)
                break
            
            ok, lint_msg = self._quick_lint_codeql_query(new_code)
            if not ok:
                error_msg = lint_msg
                logger.warning("Fix attempt %d/2 failed lint: %s", attempt + 1, error_msg)
                continue
            
            is_valid, error_msg = self._validate_codeql_query(new_code)
            if is_valid:
                logger.info("✓ Fixed query for '%s' passed validation", func_name)
                return new_code, True, new_reason, True, ""
            
            logger.warning("Fix attempt %d/2 still has error: %s", attempt + 1, error_msg)
        
        # All attempts failed
        logger.error(
            "Failed to generate valid query for '%s' after 3 attempts",
            func_name
        )
        return (
            "",
            False,
            f"Query generation failed after 3 attempts. Last error: {error_msg}",
            False,
            error_msg
        )
    
    def _build_fix_prompt(self, error_msg: str, query_code: str = "") -> str:
        """Build a prompt to fix a failing CodeQL query.
        
        Note: This method signature is kept for compatibility, but the actual
        fix prompt is now built inline in _validate_and_fix_query for better
        context.
        """
        return f"""
========================================
QUERY ERROR - FIX REQUIRED
========================================

Your previous CodeQL filter query has a compilation or validation error:

**ERROR:**
{error_msg}

Please fix the query following these STRICT rules:

**1. Imports:**
- ONLY use `import cpp`
- DO NOT import DataFlow, TaintTracking, or any dataflow libraries

**2. Query Structure:**
- Keep `@kind problem` (NOT path-problem)
- Keep all required metadata tags (@name, @description, @kind, @id)

**3. Message Format (CRITICAL):**
The message MUST contain BOTH:
- The exact phrase: `SAFE_PAIR`
- The exact phrase: `second call at line <line_number>`

Example: "SAFE_PAIR: Different indices used (second call at line 123)"

**4. Use Robust AST Matching:**
- Use concrete types: FunctionCall, Expr, Literal, Variable
- Avoid complex predicates that might fail
- Test conditions carefully

Return the corrected query in the same JSON format:
{{
  "is_special": boolean,
  "reason": "explanation",
  "query_code": "corrected CodeQL query"
}}
"""
    
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