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
import re
import time

from src.core.models import FunctionInfo, Hint, HintType, HintSet

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Template for Hint Generation
# =============================================================================

HINT_GENERATION_PROMPT = """You are a memory safety expert analyzing C/C++ code.

Analyze this function and identify its MEMORY SEMANTICS (not bugs):

Function: {func_name}
Return type: {return_type}
Parameters: {parameters}

```c
{code}
```

{context}

Identify if this function has any of these semantics:

1. **ALLOCATOR**: Returns newly allocated heap memory to caller
   - Direct allocation: calls malloc/calloc/realloc/new and returns result
   - Wrapper: wraps another allocator function
   - NOT allocator if: returns static buffer, struct field, or doesn't return allocated memory

2. **DEALLOCATOR**: Frees memory passed as argument
   - Direct: calls free/delete on an argument
   - Wrapper: wraps another deallocator function
   - Specify which argument index (0-based) is freed

3. **NULLABLE**: May return NULL
   - Explicit: has code path that returns NULL/0/nullptr
   - Implicit: returns result of function that may return NULL

4. **WRITES_BUFFER**: Writes to a buffer argument
   - Like strcpy, memcpy, sprintf, etc.
   - Specify which argument is the destination buffer

5. **SIZE_PARAM**: Has parameter that specifies buffer size
   - Like strncpy's 'n' parameter
   - Specify which parameter is the size

Return JSON (only include applicable hints):
{{
    "hints": [
        {{"type": "ALLOCATOR", "target": "return", "reason": "wraps malloc"}},
        {{"type": "DEALLOCATOR", "target": "arg0", "arg_index": 0, "reason": "calls free on first arg"}},
        {{"type": "NULLABLE", "target": "return", "reason": "returns NULL on error"}},
        {{"type": "WRITES_BUFFER", "target": "arg0", "arg_index": 0, "reason": "copies to dest buffer"}},
        {{"type": "SIZE_PARAM", "target": "arg2", "arg_index": 2, "reason": "specifies max bytes"}}
    ]
}}

If no hints apply, return {{"hints": []}}
"""


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
    """LLM client for hint generation using Vertex AI."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        project_id: str = None,
        location: str = "us-central1",
        max_retries: int = 3,
    ):
        self.model_name = model
        self.location = location
        self.max_retries = max_retries
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self._client = None

    def _init_client(self):
        """Initialize Vertex AI client lazily."""
        if self._client is not None:
            return True

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path:
                logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set")
                return False

            # Get project ID from credentials if not set
            if not self.project_id and credentials_path:
                try:
                    with open(credentials_path) as f:
                        creds = json.load(f)
                        self.project_id = creds.get("project_id")
                except Exception:
                    pass

            if not self.project_id:
                logger.warning("GCP project ID not found")
                return False

            vertexai.init(project=self.project_id, location=self.location)
            self._client = GenerativeModel(self.model_name)
            return True

        except ImportError:
            logger.warning("google-cloud-aiplatform not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to init Vertex AI: {e}")
            return False

    def query(self, prompt: str) -> dict:
        """Query LLM and return JSON response."""
        if not self._init_client():
            return {}

        for attempt in range(self.max_retries):
            try:
                response = self._client.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
            except Exception as e:
                logger.debug(f"LLM query failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        return {}


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

        for func_name, func in functions.items():
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

        # First, apply heuristics (fast and reliable)
        heuristic_hints = self._heuristic_hints(func)
        hints.extend(heuristic_hints)

        # Then, optionally use LLM for additional analysis
        if self.llm:
            llm_hints = self._llm_hints(func, all_functions)
            # Merge, avoiding duplicates
            for llm_hint in llm_hints:
                existing = next(
                    (h for h in hints
                     if h.hint_type == llm_hint.hint_type and h.target == llm_hint.target),
                    None
                )
                if not existing:
                    hints.append(llm_hint)

        return hints

    def _heuristic_hints(self, func: FunctionInfo) -> list[Hint]:
        """Generate hints using heuristics for known patterns."""
        hints = []
        code_lower = func.code.lower()

        # Check if function wraps known allocator
        for alloc in KNOWN_ALLOCATORS:
            if f"{alloc}(" in func.code and func.return_type and '*' in func.return_type:
                # Check if it returns the allocation result
                if f"return" in code_lower:
                    hints.append(Hint(
                        function_name=func.name,
                        hint_type=HintType.ALLOCATOR,
                        target="return",
                        reason=f"Wraps {alloc} and returns pointer"
                    ))
                    # Allocators are implicitly nullable
                    hints.append(Hint(
                        function_name=func.name,
                        hint_type=HintType.NULLABLE,
                        target="return",
                        reason="Allocator may return NULL"
                    ))
                    break

        # Check if function wraps known deallocator
        for dealloc in KNOWN_DEALLOCATORS:
            for i, arg_name in enumerate(func.arg_names):
                if f"{dealloc}({arg_name})" in func.code or f"{dealloc}( {arg_name} )" in func.code:
                    hints.append(Hint(
                        function_name=func.name,
                        hint_type=HintType.DEALLOCATOR,
                        target=f"arg{i}",
                        arg_index=i,
                        reason=f"Calls {dealloc} on argument {arg_name}"
                    ))
                    break

        # Check for NULL return
        if func.return_type and '*' in func.return_type:
            if re.search(r'return\s+(NULL|0|nullptr)\s*;', func.code):
                hints.append(Hint(
                    function_name=func.name,
                    hint_type=HintType.NULLABLE,
                    target="return",
                    reason="Explicitly returns NULL"
                ))

        # Check for known buffer writers
        for writer, buf_idx in KNOWN_BUFFER_WRITERS.items():
            if f"{writer}(" in func.code:
                hints.append(Hint(
                    function_name=func.name,
                    hint_type=HintType.WRITES_BUFFER,
                    target=f"arg{buf_idx}",
                    arg_index=buf_idx,
                    reason=f"Calls buffer writer {writer}"
                ))

        return hints

    def _llm_hints(
        self,
        func: FunctionInfo,
        all_functions: dict[str, FunctionInfo],
    ) -> list[Hint]:
        """Generate hints using LLM."""
        if not self.llm:
            return []

        # Build context from callees
        context_parts = []
        for callee in list(func.callees)[:5]:  # Limit context size
            if callee in all_functions:
                callee_code = all_functions[callee].code
                if len(callee_code) < 500:  # Only include small functions
                    context_parts.append(f"// Called function:\n{callee_code}")
        context = "\n\n".join(context_parts) if context_parts else ""
        context_str = f"Context (called functions):\n{context}" if context else ""

        # Format parameters
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