"""LLM-based Memory Safety Hint Generator.

This module generates memory safety hints (NOT bugs) using LLM.
Hints describe function SEMANTICS that help CodeQL understand custom functions.

Hint Types:
- ALLOCATOR: Function returns newly allocated heap memory
- DEALLOCATOR: Function frees memory passed as argument
- NULLABLE: Function may return NULL
- WRITES_BUFFER: Function writes to buffer argument
- SIZE_PARAM: Parameter specifies buffer size
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
Analyze this function and identify its MEMORY SEMANTICS and its behavioral properties relevant to memory safety.

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

### 2. DEALLOCATOR
Function **frees/releases memory** passed as an argument.

**Positive indicators:**
- Calls free/delete/delete[]/g_free/kfree on an argument
- Calls another deallocator on an argument
- Wrapper around resource cleanup

**Specify:** Which argument (0-indexed) gets freed. If multiple arguments are freed, report each separately.

### 3. NULLABLE
Function's return value **may be NULL** under some conditions.

**Positive indicators:**
- Explicit `return NULL`, `return 0`, or `return nullptr`
- Returns result of malloc/calloc (which may return NULL)
- Returns result of another nullable function
- Has error handling path that returns NULL
- Lookup/search function that may not find element

**Note:** All ALLOCATORs are implicitly NULLABLE (malloc can fail), so only report NULLABLE separately if the function is NOT an allocator but may still return NULL.

### 4. WRITES_BUFFER
Function **writes data to a buffer argument** (potential overflow risk).

**Positive indicators:**
- Calls strcpy/strcat/sprintf/memcpy/memmove targeting an argument
- Uses loop to write into argument buffer
- Assigns to dereferenced pointer argument: `*buf = ...` or `buf[i] = ...`

**Specify:** Which argument (0-indexed) is the destination buffer.

### 5. SIZE_PARAM
A parameter **specifies buffer size or max length** (used for bounds checking).

**Positive indicators:**
- Parameter used as limit in strncpy/snprintf/memcpy size argument
- Parameter used as loop bound when writing to buffer
- Parameter named size/len/count/max/capacity/n/num_bytes

**Specify:** Which parameter (0-indexed) is the size.

## Analysis Guidelines

1. **Trace data flow:** Follow where return values come from and where arguments flow to.
2. **Consider all paths:** Check all branches and return statements.
3. **Indirect calls matter:** If function calls helper that allocates/frees, propagate that semantic.
4. **Be precise:** Only report semantics you can verify from the code.
5. **Provide evidence:** Your reason should cite specific code elements (function calls, return statements, etc.)

## Output Format

Return a JSON object with hints array. Each hint must have:
- `type`: One of ALLOCATOR, DEALLOCATOR, NULLABLE, WRITES_BUFFER, SIZE_PARAM
- `target`: "return" for return value, or "argN" for argument N
- `arg_index`: (required for DEALLOCATOR, WRITES_BUFFER, SIZE_PARAM) 0-based argument index
- `reason`: Brief evidence from the code (cite specific lines/calls)
```json
{{
    "hints": [
        {{"type": "ALLOCATOR", "target": "return", "reason": "line 5: returns malloc(size) result"}},
        {{"type": "DEALLOCATOR", "target": "arg0", "arg_index": 0, "reason": "line 8: calls free(ptr)"}},
        {{"type": "NULLABLE", "target": "return", "reason": "line 3: returns NULL if size==0"}},
        {{"type": "WRITES_BUFFER", "target": "arg0", "arg_index": 0, "reason": "line 6: memcpy(dest, src, n)"}},
        {{"type": "SIZE_PARAM", "target": "arg2", "arg_index": 2, "reason": "parameter 'n' limits memcpy length"}}
    ]
}}
```

If no memory semantics apply, return: `{{"hints": []}}`

Now analyze the function above and return the JSON result."""

# =============================================================================
# Known Functions (Heuristic Fallback)
# =============================================================================

KNOWN_ALLOCATORS = {
    "malloc", "calloc", "realloc", "strdup", "strndup",
    "aligned_alloc", "memalign", "posix_memalign",
    "g_malloc", "g_malloc0", "g_new", "g_new0",
    "kmalloc", "kzalloc", "vmalloc",
    "xmalloc", "xcalloc", "xrealloc",
}

KNOWN_DEALLOCATORS = {
    "free", "cfree", "g_free", "kfree", "vfree", "xfree",
}

KNOWN_BUFFER_WRITERS = {
    "strcpy": 0, "strncpy": 0, "strcat": 0, "strncat": 0,
    "sprintf": 0, "snprintf": 0, "vsprintf": 0, "vsnprintf": 0,
    "memcpy": 0, "memmove": 0, "memset": 0,
    "gets": 0, "fgets": 0, "read": 1, "recv": 1,
}

KNOWN_SIZE_PARAMS = {
    "strncpy": 2, "strncat": 2, "snprintf": 1, "vsnprintf": 1,
    "memcpy": 2, "memmove": 2, "memset": 2,
    "fgets": 1, "read": 2, "recv": 2,
}


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
        functions: dict[str, FunctionInfo]
    ) -> HintSet:
        """Generate hints for all functions in codebase.

        Args:
            functions: Dict of function name -> FunctionInfo

        Returns:
            HintSet with all generated hints
        """
        hint_set = HintSet()

        for func_name, func in tqdm(functions.items()):
            # Skip main and test functions
            if func_name in ("main", "_main", "wmain") or "test" in func_name.lower():
                continue

            hints = self._generate_for_function(func, functions)

            for hint in hints:
                hint_set.add(hint)

        return hint_set

    def _generate_for_function(
        self,
        func: FunctionInfo,
        all_functions: dict[str, FunctionInfo],
    ) -> list[Hint]:
        """Generate hints for a single function."""
        hints = []

        llm_hints = self._llm_hints(func, all_functions)
        for llm_hint in llm_hints:
            existing = next(
                (h for h in hints
                    if h.hint_type == llm_hint.hint_type and h.target == llm_hint.target),
                None
            )
            if not existing:
                hints.append(llm_hint)

        return hints

    def _llm_hints(
        self,
        func: FunctionInfo,
        all_functions: dict[str, FunctionInfo],
    ) -> list[Hint]:
        """Generate hints using LLM."""
        if not self.llm:
            return []

        context_parts = []
        for callee in list(func.callees)[:5]:
            if callee in all_functions:
                callee_code = all_functions[callee].code
                context_parts.append(f"// Called function:\n{callee_code}")

        context = "\n\n".join(context_parts) if context_parts else ""
        context_str = f"Context (called functions):\n{context}" if context else ""

        params = list(zip(func.arg_types, func.arg_names))
        params_str = ", ".join(f"{t} {n}" for t, n in params) if params else "void"

        prompt = HINT_GENERATION_PROMPT.format(
            func_name=func.name,
            return_type=func.return_type or "void",
            parameters=params_str,
            code=func.code,
            context=context_str,
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