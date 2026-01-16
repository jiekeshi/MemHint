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
IMPORTANT: A function can be an ALLOCATOR even if it returns void.
Do NOT rely on the function's return type when deciding allocator semantics.

**Positive indicators:**
- Calls malloc/calloc/realloc/aligned_alloc/new/new[] and returns the result
- Calls another known allocator (e.g., g_malloc, xmalloc, kmalloc) and returns result
- Returns result of a wrapper function that allocates
- Allocates memory and stores it into a local variable, even if that variable is not returned or passed out (still an ALLOCATOR)
- Calls strdup/wcsdup/_strdup/wmemdup or similar duplication functions that allocate using malloc internally

**Negative indicators (NOT an allocator):**
- Returns pointer to static/global buffer
- Returns pointer to struct field or array member
- Returns one of the input arguments
- Returns stack-allocated memory (dangling pointer bug, but not allocator semantic)

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
- `arg_index`: (required for DEALLOCATOR) 0-based argument index
- `reason`: Brief evidence from the code (cite specific lines/calls)
```json
{{
    "hints": [
        {{"type": "ALLOCATOR", "target": "return", "reason": "line 5: returns malloc(size) result"}},
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

    def query(self, prompt: str) -> dict:
        """Send query, prefer Vertex AI when service account credentials are configured."""
        content = self._call_by_vertex_ai(prompt)
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON. Raw content: %s", content[:200])
            return {}

    def _call_by_vertex_ai(self, prompt: str) -> str | None:
        """Use Vertex AI SDK to call Gemini with service account credentials."""
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
                return response.text
            except Exception as exc:
                logger.warning("Vertex AI call failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        return None


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

    def generate_hints(
        self,
        functions: dict[str, FunctionInfo],
        previous_conflicts: list[str] = None
    ) -> HintSet:
        """Generate hints for all functions in codebase.

        Args:
            functions: Dict of function name -> FunctionInfo
            previous_conflicts: List of conflict messages from previous validation
                              (format: "REMOVED func_name.HintType.name: reason")

        Returns:
            HintSet with all generated hints
        """
        hint_set = HintSet()
        
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

        for func_name, func in tqdm(functions.items()):
            # Skip main and test functions
            if func_name in ("main", "_main", "wmain") or "test" in func_name.lower():
                continue

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
        previous_conflicts: list[str] = None
    ) -> HintSet:
        """Regenerate hints only for functions that had conflicts.
        
        Args:
            functions: Dict of all function name -> FunctionInfo
            conflict_functions: Set of function names that had conflicts
            previous_conflicts: List of conflict messages from previous validation
                              (format: "REMOVED func_name.HintType.name: reason")
        
        Returns:
            HintSet with regenerated hints for conflict functions only
        """
        hint_set = HintSet()
        
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

        # Only process conflict functions
        for func_name in conflict_functions:
            if func_name not in functions:
                continue
            # Skip main and test functions
            if func_name in ("main", "_main", "wmain") or "test" in func_name.lower():
                continue
            
            func = functions[func_name]
            # Get conflicts for this function if any
            func_conflicts = conflict_map.get(func_name, [])
            hints = self._generate_for_function(func, functions, func_conflicts)
            
            for hint in hints:
                hint_set.add(hint)
        
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
The following hints were rejected by Z3 validation:
{chr(10).join(f"- {c}" for c in previous_conflicts)}

Z3 does not think there is a memory leak (or the rejected semantic) here. Please reconsider your analysis.
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

        result = self.llm.query(prompt)
        hints = []

        for h in result.get("hints", []):
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