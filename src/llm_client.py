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
import time

from tqdm import tqdm

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    VERTEX_AI_AVAILABLE = True
except ImportError:  # pragma: no cover - environment may not have Vertex SDK
    vertexai = None
    GenerativeModel = None
    VERTEX_AI_AVAILABLE = False



from src.core.models import FunctionInfo, Hint, HintType, HintSet

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

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client
        # Cost tracking: per function and per hint
        self.function_costs: dict[str, dict] = {}  # function_name -> cost info
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        # Provided by Pipeline (tree-sitter based): typedef alias names that are pointer types.
        self.pointer_typedef_aliases: set[str] = set()
        # Snapshot of last filtering decision for external consumers (e.g., Pipeline)
        self.last_filter_classes: dict[str, list[str]] | None = None

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

            hints.append(Hint(
                function_name=func.name,
                hint_type=hint_type,
                target=h.get("target", "return"),
                arg_index=h.get("arg_index", -1),
                reason=h.get("reason", "LLM analysis"),
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