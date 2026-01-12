"""OpenAI LLM client for memory safety annotation generation."""

import json
import logging
import os
import time

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    VERTEX_AI_AVAILABLE = True
except ImportError:  # pragma: no cover - environment may not have Vertex SDK
    vertexai = None
    GenerativeModel = None
    VERTEX_AI_AVAILABLE = False

from src.core.models import (
    FunctionInfo, Annotation, AnnotationType, AnnotationSet, CounterExample
)

logger = logging.getLogger(__name__)


# Prompt 1: Function-level annotations (allocator/deallocator identification)
PROMPT_FUNC_ANNOTATION = """You are a memory safety expert analyzing C/C++ code.

Analyze this function for memory management patterns:

Function: {func_name}
```c
{code}
```

{context}

Identify:
1. **Is this function an ALLOCATOR?** (returns newly allocated heap memory to caller)
   - Direct: malloc, calloc, realloc, new
   - Wrapper that returns allocated memory
   - NOT an allocator if it just uses malloc internally without returning it

2. **Is this function a DEALLOCATOR?** (frees memory passed as argument)
   - Direct: free, delete
   - Wrapper that frees its argument

3. **Ownership semantics**:
   - Does it transfer ownership to caller via return?
   - Does it take ownership of an argument?

4. **Null safety**:
   - Can return NULL?
   - Requires non-null arguments?

Return JSON:
{{
    "is_allocator": true/false,
    "allocation_type": "malloc"|"calloc"|"wrapper"|null,
    "is_deallocator": true/false,
    "freed_arg_index": 0|1|2|null,
    "may_return_null": true/false,
    "transfers_ownership_to_caller": true/false,
    "reasoning": "brief explanation"
}}
"""

# Prompt 2: Bug detection (hints only, not merged with analyzer results)
PROMPT_BUG_DETECTION = """You are a memory safety expert. Analyze this function for MEMORY BUGS:

```c
{code}
```

Look for these specific bugs:

1. **MEMORY LEAK**: Allocated memory that may not be freed on some path
2. **USE-AFTER-FREE**: Memory accessed after being freed
3. **DOUBLE-FREE**: Memory freed more than once
4. **NULL-DEREFERENCE**: Pointer used without null check

Return JSON:
{{
    "bugs": [
        {{
            "type": "MEMORY_LEAK"|"USE_AFTER_FREE"|"DOUBLE_FREE"|"NULL_DEREFERENCE",
            "variable": "variable_name",
            "alloc_line": line_number_or_null,
            "free_line": line_number_or_null,
            "use_line": line_number_or_null,
            "condition": "when bug occurs",
            "confidence": 0.0-1.0,
            "explanation": "why this is a bug"
        }}
    ],
    "is_safe": true/false
}}
"""

# Prompt 3: Validate annotation
PROMPT_VALIDATE = """Verify memory annotation for function `{func_name}`:

Annotation: {annotation_type} on {target}
```c
{code}
```

Check for FALSE POSITIVES:
1. Returns field of input structure (borrowed, not new allocation)
2. Returns static/global buffer
3. Returns cached/pooled memory (not caller's responsibility)
4. Memory managed by container/arena
5. Reference counted object

Return JSON:
{{
    "is_valid": true/false,
    "false_positive_reason": "struct_field"|"static_buffer"|"cached"|"container"|"refcount"|null,
    "explanation": "reason"
}}
"""

# Prompt 3: Refine annotation based on conflict
PROMPT_REFINE = """Refine annotation based on symbolic execution conflict:

Function: {func_name}
Current annotation: {annotation_type} on {target}
Conflict type: {conflict_type}
Conflict reason: {conflict_reason}

```c
{code}
```

What should happen to this annotation?

Return JSON:
{{
    "action": "remove"|"modify"|"keep",
    "new_annotation_type": "ALLOC_SOURCE"|"FREE_SINK"|"STATIC_BUFFER"|"BORROWED_REF"|null,
    "new_target": "return"|"arg0"|etc|null,
    "confidence": 0.0-1.0,
    "explanation": "reason"
}}
"""


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


class AnnotationGenerator:
    """Generate memory safety annotations using LLM.

    The LLM generates two types of annotations:
    1. Function-level properties (allocator/deallocator/ownership)
       - Used by static analyzers to understand custom memory functions
    2. Bug detection hints (potential leaks/UAF/double-free)
       - Logged for reference but NOT merged with analyzer results
       - Static analyzer is the sole source of truth for bugs
    """

    def __init__(self, client: LLMClient):
        self.client = client

    def generate(
        self, func: FunctionInfo, all_funcs: dict[str, FunctionInfo], max_context: int = 8000
    ) -> list[Annotation]:
        """Generate all annotations for a function.

        Returns both function annotations and bug hints.
        Bug hints are marked with is_bug_annotation() = True.
        """
        annotations = []

        # Part 1: Function-level annotations (used by static analyzer)
        func_anns = self._generate_func_annotations(func, all_funcs, max_context)
        annotations.extend(func_anns)

        # Part 2: Bug detection hints (for reference/logging only)
        bug_hints = self._detect_bugs(func)
        annotations.extend(bug_hints)

        return annotations

    def _generate_func_annotations(
        self, func: FunctionInfo, all_funcs: dict[str, FunctionInfo], max_context: int
    ) -> list[Annotation]:
        """Generate function-level annotations (allocator/deallocator)."""
        # Build callee context
        context_parts = []
        for callee in func.callees:
            if callee in all_funcs:
                context_parts.append(all_funcs[callee].code)
        context = "\n\n".join(context_parts)[:max_context]
        context_str = f"Called functions:\n```c\n{context}\n```" if context else ""

        prompt = PROMPT_FUNC_ANNOTATION.format(
            func_name=func.name, code=func.code, context=context_str
        )
        result = self.client.query(prompt)

        annotations = []

        # Allocator annotation
        if result.get("is_allocator") and func.name not in ("main", "_main", "wmain"):
            annotations.append(Annotation(
                function_name=func.name,
                annotation_type=AnnotationType.ALLOC_SOURCE,
                target="return",
                reason=f"allocation_type: {result.get('allocation_type')}"
            ))

        # Deallocator annotation
        if result.get("is_deallocator"):
            arg_idx = result.get("freed_arg_index", 0)
            annotations.append(Annotation(
                function_name=func.name,
                annotation_type=AnnotationType.FREE_SINK,
                target=f"arg{arg_idx}",
                reason="deallocator function"
            ))

        # Null return annotation
        if result.get("may_return_null"):
            annotations.append(Annotation(
                function_name=func.name,
                annotation_type=AnnotationType.MUST_CHECK_NULL,
                target="return",
                reason="may return null"
            ))

        # Ownership transfer
        if result.get("transfers_ownership_to_caller"):
            annotations.append(Annotation(
                function_name=func.name,
                annotation_type=AnnotationType.OWNERSHIP_RETURN,
                target="return",
                reason="transfers ownership"
            ))

        return annotations

    def _detect_bugs(self, func: FunctionInfo) -> list[Annotation]:
        """Detect potential memory bugs in function.

        These are hints only - NOT merged with static analyzer results.
        The static analyzer is the sole source of truth for bugs.
        """
        prompt = PROMPT_BUG_DETECTION.format(code=func.code)
        result = self.client.query(prompt)

        annotations = []

        bugs = result.get("bugs", [])
        for bug in bugs:
            bug_type = bug.get("type", "")
            variable = bug.get("variable", "")
            confidence = bug.get("confidence", 0.5)

            # Only report high-confidence bugs
            if confidence < 0.6:
                continue

            # Map bug type to annotation type
            ann_type = None
            if bug_type == "MEMORY_LEAK":
                ann_type = AnnotationType.POTENTIAL_LEAK
            elif bug_type == "USE_AFTER_FREE":
                ann_type = AnnotationType.USE_AFTER_FREE
            elif bug_type == "DOUBLE_FREE":
                ann_type = AnnotationType.DOUBLE_FREE
            elif bug_type == "NULL_DEREFERENCE":
                ann_type = AnnotationType.NULL_DEREF

            if ann_type:
                annotations.append(Annotation(
                    function_name=func.name,
                    annotation_type=ann_type,
                    target=variable,
                    reason=bug.get("explanation", ""),
                    line_number=bug.get("use_line") or bug.get("free_line") or bug.get("alloc_line"),
                    confidence=confidence,
                    condition=bug.get("condition", "")
                ))

        return annotations

    def validate(
        self, func: FunctionInfo, ann: Annotation
    ) -> tuple[bool, str]:
        """Validate an annotation."""
        prompt = PROMPT_VALIDATE.format(
            func_name=func.name,
            annotation_type=ann.annotation_type.name,
            target=ann.target,
            code=func.code
        )
        result = self.client.query(prompt)
        return result.get("is_valid", True), result.get("explanation", "")

    def refine(self, ann: Annotation, func: FunctionInfo, cex: CounterExample) -> tuple[str, AnnotationType, str]:
        """Refine annotation based on counter-example. Returns (action, new_type, new_target)."""
        prompt = PROMPT_REFINE.format(
            func_name=func.name,
            annotation_type=ann.annotation_type.name,
            target=ann.target,
            conflict_type=cex.conflict_type.name,
            conflict_reason=cex.reason,
            code=func.code,
        )
        result = self.client.query(prompt)

        action = result.get("action", "keep")
        new_type_str = result.get("new_annotation_type")
        new_type = None
        if new_type_str:
            try:
                new_type = AnnotationType[new_type_str]
            except KeyError:
                pass
        new_target = result.get("new_target", ann.target)

        return action, new_type, new_target

    def analyze_issue(self, code: str, issue_type: str, location: str, message: str) -> dict:
        """Analyze a potential memory safety issue."""
        prompt = f"""Analyze this code for a potential {issue_type} bug:

```c
{code}
```

Warning location: {location}
Message: {message}

Is this a real bug or false positive?

Return JSON:
{{
    "is_real_bug": true/false,
    "confidence": 0.0-1.0,
    "explanation": "analysis",
    "suggested_fix": "how to fix"
}}
"""
        return self.client.query(prompt)