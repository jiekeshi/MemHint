"""Core data structures for HINT pipeline - Memory Safety Analysis.

We define memory safety hints that guide static analyzers:
- ALLOCATOR: Returns newly allocated heap memory
- DEALLOCATOR: Frees memory at argument N

Pipeline flow:
1. LLM generates Hints (function semantics)
2. Z3 validates Hints (filter impossible annotations)
3. CodeQL + Hints scans for bugs
4. Z3 filters spurious warnings (path feasibility)

LLM's role: Annotate functions with memory safety hints
Static Analyzer's role: Detect bugs based on these hints
Z3's role: Validate hints and filter infeasible warnings
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Literal
import datetime

# =============================================================================
# Memory Safety Hint Types (LLM generates these)
# =============================================================================

class HintType(Enum):
    """Memory safety hints for static analysis.

    These describe FUNCTION SEMANTICS, not bugs.
    LLM identifies these patterns in custom/wrapper functions.

    Examples:
        ALLOCATOR: my_malloc, create_buffer, pool_alloc
        DEALLOCATOR: my_free, destroy_buffer, pool_release
    """
    # Memory management (for detecting leak/UAF/double-free)
    ALLOCATOR = auto()        # Returns newly allocated heap memory
    DEALLOCATOR = auto()      # Frees memory at argument N


class MemoryIssueType(Enum):
    """Types of memory safety bugs that CodeQL detects."""
    MEMORY_LEAK = auto()        # Allocated memory never freed
    USE_AFTER_FREE = auto()     # Accessing freed memory
    DOUBLE_FREE = auto()        # Freeing already freed memory
    MEMORY_LEAK_FILTERED = auto() # Allocated memory never freed with a filtered condition
    USE_AFTER_FREE_FILTERED = auto() # Accessing freed memory with a filtered condition
    DOUBLE_FREE_FILTERED = auto() # Freeing already freed memory with a filtered condition


# Mapping: Which hints help detect which bugs
HINT_TO_BUGS = {
    HintType.ALLOCATOR: [MemoryIssueType.MEMORY_LEAK, MemoryIssueType.USE_AFTER_FREE,
                         MemoryIssueType.DOUBLE_FREE, MemoryIssueType.MEMORY_LEAK_FILTERED, MemoryIssueType.USE_AFTER_FREE_FILTERED, MemoryIssueType.DOUBLE_FREE_FILTERED],
    HintType.DEALLOCATOR: [MemoryIssueType.MEMORY_LEAK, MemoryIssueType.USE_AFTER_FREE,
                           MemoryIssueType.DOUBLE_FREE, MemoryIssueType.MEMORY_LEAK_FILTERED, MemoryIssueType.USE_AFTER_FREE_FILTERED, MemoryIssueType.DOUBLE_FREE_FILTERED],
}


# =============================================================================
# Function Information (from parsing)
# =============================================================================

@dataclass
class FunctionInfo:
    """Function information extracted from source code."""
    name: str
    code: str
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    callees: set[str] = field(default_factory=set)
    callers: set[str] = field(default_factory=set)
    arg_names: list[str] = field(default_factory=list)
    arg_types: list[str] = field(default_factory=list)
    return_type: str = ""
    return_expressions: set[str] = field(default_factory=set)  # Return statements in function


# =============================================================================
# Hints (LLM generates these)
# =============================================================================

@dataclass
class Hint:
    """A memory safety hint for a function.

    This describes function SEMANTICS, not bugs.
    LLM generates these to help CodeQL understand custom functions.
    """
    function_name: str
    hint_type: HintType
    target: str = "return"    # "return", "arg0", "arg1", etc.
    arg_index: int = -1       # For DEALLOCATOR: which argument
    reason: str = ""
    arg_semantics: dict[int, str] = field(default_factory=dict)  # arg_index -> semantic description
    # Example: {0: "dictionary object pointer to clear", 1: "memory index parameter for d"}


@dataclass
class HintSet:
    """Collection of memory safety hints for a codebase."""
    hints: dict[str, list[Hint]] = field(default_factory=dict)

    def add(self, hint: Hint) -> None:
        """Add a hint, avoiding duplicates. Merge arg_semantics if hint already exists."""
        func_name = hint.function_name
        if func_name not in self.hints:
            self.hints[func_name] = []
        # Check for duplicates
        for existing in self.hints[func_name]:
            if existing.hint_type == hint.hint_type and existing.target == hint.target:
                # Merge arg_semantics from the new hint into existing hint
                existing.arg_semantics.update(hint.arg_semantics)
                return
        self.hints[func_name].append(hint)

    def remove(self, func_name: str, hint_type: HintType) -> bool:
        """Remove hints of a specific type from a function."""
        if func_name in self.hints:
            orig_len = len(self.hints[func_name])
            self.hints[func_name] = [
                h for h in self.hints[func_name] if h.hint_type != hint_type
            ]
            return len(self.hints[func_name]) < orig_len
        return False

    def get_by_type(self, hint_type: HintType) -> list[tuple[str, Hint]]:
        """Get all functions with a specific hint type."""
        result = []
        for func_name, hints in self.hints.items():
            for h in hints:
                if h.hint_type == hint_type:
                    result.append((func_name, h))
        return result

    def get_allocators(self) -> list[tuple[str, int]]:
        """Get all allocator function names with their output index.

        Returns:
            List of (function_name, arg_index) where:
            - arg_index = -1 means return value
            - arg_index >= 0 means output parameter at that index
        """
        result = []
        for fn, h in self.get_by_type(HintType.ALLOCATOR):
            arg_idx = h.arg_index if h.arg_index is not None else -1
            result.append((fn, arg_idx))
        return result

    def get_deallocators(self) -> list[tuple[str, int]]:
        """Get all deallocator function names with their freed argument index.

        Returns:
            List of (function_name, arg_index) where arg_index is the 0-based
            index of the argument that gets freed.
        """
        result = []
        for fn, h in self.get_by_type(HintType.DEALLOCATOR):
            arg_idx = h.arg_index if h.arg_index is not None and h.arg_index >= 0 else 0
            result.append((fn, arg_idx))
        return result

    def to_json(self) -> dict:
        """Export as JSON format."""
        result = {"hints": {}}
        for func_name, hints in self.hints.items():
            result["hints"][func_name] = [
                {
                    "type": h.hint_type.name,
                    "target": h.target,
                    "arg_index": h.arg_index,
                    "reason": h.reason,
                    "arg_semantics": h.arg_semantics,  # Include parameter semantics
                }
                for h in hints
            ]
        return result

    @classmethod
    def from_json(cls, data: dict) -> "HintSet":
        """Import from JSON format."""
        hint_set = cls()
        for func_name, hints in data.get("hints", {}).items():
            for h in hints:
                # Convert arg_semantics from list of [index, desc] pairs to dict if needed
                arg_semantics = h.get("arg_semantics", {})
                if isinstance(arg_semantics, list):
                    # Handle old format: list of [index, description] pairs
                    arg_semantics = {int(idx): desc for idx, desc in arg_semantics}
                elif not isinstance(arg_semantics, dict):
                    # Ensure it's a dict with int keys
                    arg_semantics = {int(k): v for k, v in arg_semantics.items()} if arg_semantics else {}
                
                hint_set.add(Hint(
                    function_name=func_name,
                    hint_type=HintType[h["type"]],
                    target=h.get("target", "return"),
                    arg_index=h.get("arg_index", -1),
                    reason=h.get("reason", ""),
                    arg_semantics=arg_semantics,
                ))
        return hint_set

    def summary(self) -> str:
        """Return a summary of hints."""
        n_alloc = len(self.get_allocators())
        n_dealloc = len(self.get_deallocators())

        parts = []
        if n_alloc: parts.append(f"{n_alloc} allocators")
        if n_dealloc: parts.append(f"{n_dealloc} deallocators")

        return f"Hints: {len(self.hints)} functions ({', '.join(parts) if parts else 'none'})"

    def __len__(self) -> int:
        return len(self.hints)


"""
Custom query objects for LLM-generated CodeQL filter queries.

This module defines:
- CustomQuery: a single filter query produced by the LLM to identify safe code patterns
- CustomQuerySet: a container for both "special" (query-producing) and
  "non-special" (reason-only) evaluations, with backward-compatible JSON IO.

Design goals
------------
1) Backward compatible with older JSON dumps.
2) Focus exclusively on false positive suppression (filter queries).
3) Clear separation between:
   - CodeQL filter predicates that produce filter evidence
   - Filter predicates for identifying safe patterns
"""

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class SuppressRule:
    """
    A lightweight rule to suppress/mark CodeQL warnings in the pipeline.

    This is intentionally NOT CodeQL code. It allows pipeline-level filtering
    without requiring CodeQL itself to support suppression.

    Common patterns:
    - Match on rule_id (e.g., cpp/double-free)
    - Match on function_name
    - Match on callee name and argument patterns (stringified)
    """
    name: str
    reason: str
    
    # Matching criteria (all optional; if multiple specified, treat as AND)
    rule_id_contains: Optional[str] = None
    function_name_equals: Optional[str] = None
    callee_global_name_equals: Optional[str] = None

    # Optional argument pattern checks (string compare, keep it cheap and robust)
    # Example: {"0": "d", "1": "0"} means arg0.toString()=="d" and arg1.toString()=="0"
    arg_to_string_equals: Dict[str, str] = field(default_factory=dict)

    # Optional: "must differ" constraints for pairs of calls
    # Example: {"1": True} means arg1 must be different across two calls (e.g., htidx)
    arg_must_differ_for_pair: Dict[str, bool] = field(default_factory=dict)


@dataclass
class CustomQuery:
    """
    A custom CodeQL filter query generated for a function with special semantics.

    Purpose: Generate evidence queries that identify SAFE code patterns which
    the pipeline can use to filter out false positive warnings.

    Suppression approach:
    Filter predicates: CodeQL predicates that identify safe patterns (per-bug-type filters)
    """

    # Core identity
    function_name: str
    reason: str = ""

    # Classification
    is_special: bool = True

    # Validation bookkeeping
    validated: bool = False
    validation_error: str = ""

    # When this object was created (ISO8601, UTC)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )

    # New-style per-bug-type filter blocks as returned by the LLM.
    # Each block is a small dict, typically with:
    #   - "predicates_code": CodeQL predicate definitions
    #   - "use_expr": how to call those predicates from the main filtered query
    double_free_filter: Dict[str, Any] = field(default_factory=dict)
    use_after_free_filter: Dict[str, Any] = field(default_factory=dict)
    memory_never_freed_filter: Dict[str, Any] = field(default_factory=dict)
    memory_may_not_be_freed_filter: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomQuerySet:
    """
    Collection of custom filter queries and evaluations.

    Stores two categories:
    - queries: special functions (include filter predicates)
    - non_special: non-special functions with reason only
    """
    queries: Dict[str, CustomQuery] = field(default_factory=dict)
    non_special: Dict[str, str] = field(default_factory=dict)

    def add(self, query: CustomQuery) -> None:
        """Add a custom query or non-special evaluation."""
        if query.is_special:
            self.queries[query.function_name] = query
        else:
            self.non_special[query.function_name] = query.reason

    def get(self, function_name: str) -> Optional[CustomQuery]:
        """Get a custom query by function name."""
        return self.queries.get(function_name)

    def __len__(self) -> int:
        return len(self.queries) + len(self.non_special)

    def summary(self) -> str:
        """Get a summary string of the query set."""
        return (
            f"Evaluated {len(self)} functions: "
            f"{len(self.queries)} special, {len(self.non_special)} non-special"
        )

    # -------------------------------------------------------------------------
    # JSON IO (backward compatible)
    # -------------------------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        """
        Export as JSON.

        Structure:
        {
          "queries": {
            "<func>": {
              "double_free_filter": {...},
              "use_after_free_filter": {...},
              "memory_never_freed_filter": {...},
              "memory_may_not_be_freed_filter": {...},
              "reason": "...",
              "created_at": "..."
            },
            ...
          },
          "non_special": {
            "<func>": { "reason": "..." },
            ...
          }
        }
        """
        out: Dict[str, Any] = {"queries": {}, "non_special": {}}

        for func_name, q in self.queries.items():
            # New structured format: per-bug-type filter blocks for this function.
            df_block: Dict[str, Any] = dict(q.double_free_filter or {})
            uaf_block: Dict[str, Any] = dict(q.use_after_free_filter or {})
            never_freed_block: Dict[str, Any] = dict(q.memory_never_freed_filter or {})
            may_not_be_freed_block: Dict[str, Any] = dict(q.memory_may_not_be_freed_filter or {})

            out["queries"][func_name] = {
                "double_free_filter": df_block,
                "use_after_free_filter": uaf_block,
                "memory_never_freed_filter": never_freed_block,
                "memory_may_not_be_freed_filter": may_not_be_freed_block,
                "reason": q.reason,
                "created_at": q.created_at,
            }

        for func_name, reason in self.non_special.items():
            out["non_special"][func_name] = {"reason": reason}

        return out

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CustomQuerySet":
        """
        Import from JSON with backward compatibility.

        Supports older formats where:
        - non_special[func] was a string
        - queries[func] only had {reason} (legacy format)
        """
        qs = cls()

        # --- special queries ---
        for func_name, q in (data.get("queries") or {}).items():
            reason = ""
            validated = False
            validation_error = ""
            created_at = ""
            df_block: Dict[str, Any] = {}
            uaf_block: Dict[str, Any] = {}
            never_freed_block: Dict[str, Any] = {}
            may_not_be_freed_block: Dict[str, Any] = {}
            leak_block: Dict[str, Any] = {}  # Legacy field

            if isinstance(q, str):
                # Extremely old format: value was the query code (no longer supported)
                # Skip this entry as we no longer support query_code
                continue
            elif isinstance(q, dict):
                # New structured format with per-bug-type filters
                if (
                    "double_free_filter" in q
                    or "use_after_free_filter" in q
                    or "memory_never_freed_filter" in q
                    or "memory_may_not_be_freed_filter" in q
                ):
                    df_block = dict(q.get("double_free_filter") or {})
                    uaf_block = dict(q.get("use_after_free_filter") or {})
                    never_freed_block = dict(q.get("memory_never_freed_filter") or {})
                    may_not_be_freed_block = dict(q.get("memory_may_not_be_freed_filter") or {})

                    # Extract validation info from filter blocks
                    validated = bool(df_block.get("validated", False) or 
                                   uaf_block.get("validated", False) or
                                   never_freed_block.get("validated", False) or
                                   may_not_be_freed_block.get("validated", False))
                    validation_error = (df_block.get("validation_error", "") or 
                                      uaf_block.get("validation_error", "") or
                                      never_freed_block.get("validation_error", "") or
                                      may_not_be_freed_block.get("validation_error", "") or "")

                    reason = q.get("reason", "") or ""
                    created_at = q.get("created_at", "") or ""
                else:
                    # Legacy dict format - just read reason and created_at
                    reason = q.get("reason", "") or ""
                    validated = bool(q.get("validated", False))
                    validation_error = q.get("validation_error", "") or ""
                    created_at = q.get("created_at", "") or ""

            cq = CustomQuery(
                function_name=func_name,
                reason=reason,
                is_special=True,
                validated=validated,
                validation_error=validation_error,
                created_at=created_at or datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                double_free_filter=df_block,
                use_after_free_filter=uaf_block,
                memory_never_freed_filter=never_freed_block,
                memory_may_not_be_freed_filter=may_not_be_freed_block,
            )
            qs.queries[func_name] = cq

        # --- non-special ---
        for func_name, info in (data.get("non_special") or {}).items():
            if isinstance(info, str):
                qs.non_special[func_name] = info
            elif isinstance(info, dict):
                qs.non_special[func_name] = info.get("reason", "") or ""
            else:
                qs.non_special[func_name] = ""

        return qs

    def save(self, filepath: str) -> None:
        """Save to JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "CustomQuerySet":
        """Load from JSON file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_json(data)

# =============================================================================
# Analysis Results (CodeQL generates these, Z3 filters them)
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum

@dataclass
class Warning:
    """A warning from static analyzer (CodeQL/Infer)."""
    file_path: str
    line_number: int
    function_name: str
    warning_type: str
    message: str
    issue_type: MemoryIssueType = MemoryIssueType.MEMORY_LEAK
    allocation_site: str = ""
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "warning_type": self.warning_type,
            "message": self.message,
            "issue_type": self.issue_type.name,  # enum → string
            "allocation_site": self.allocation_site,
            "trace": self.trace,
        }

    @staticmethod
    def from_dict(d: dict) -> "Warning":
        return Warning(
            file_path=d["file_path"],
            line_number=d["line_number"],
            function_name=d["function_name"],
            warning_type=d["warning_type"],
            message=d["message"],
            issue_type=MemoryIssueType[d["issue_type"]],  # string → enum
            allocation_site=d.get("allocation_site", ""),
            trace=d.get("trace", []),
        )



@dataclass
class Evidence:
    """Evidence for a confirmed bug (after Z3 validation)."""
    warning: Warning
    concrete_trace: list[str] = field(default_factory=list)
    root_cause: str = ""
    suggested_fix: str = ""
    z3_validated: bool = True


@dataclass
class AnalysisResult:
    """Final analysis result."""
    confirmed_bugs: list[Evidence]
    hints: HintSet
    iterations: int = 1
    spurious_filtered: int = 0

    def bugs_by_type(self) -> dict[MemoryIssueType, list[Evidence]]:
        """Group bugs by type."""
        result: dict[MemoryIssueType, list[Evidence]] = {}
        for bug in self.confirmed_bugs:
            t = bug.warning.issue_type
            if t not in result:
                result[t] = []
            result[t].append(bug)
        return result

    def summary(self) -> str:
        """Return analysis summary."""
        by_type = self.bugs_by_type()
        type_lines = [f"  - {t.name}: {len(bugs)}" for t, bugs in by_type.items()]
        type_str = "\n".join(type_lines) if type_lines else "  None"

        return f"""
=== HINT Analysis Result ===
Total bugs: {len(self.confirmed_bugs)}
By type:
{type_str}
{self.hints.summary()}
Iterations: {self.iterations}
Spurious warnings filtered: {self.spurious_filtered}
"""


# =============================================================================
# Validation Results (Z3 generates these)
# =============================================================================

@dataclass
class ValidationResult:
    """Result of Z3 validation for a hint or warning."""
    is_valid: bool
    reason: str
    counterexample: Optional[dict] = None  # Variable assignments if invalid


@dataclass
class PathFeasibilityResult:
    """Result of Z3 path feasibility check for a warning."""
    is_feasible: bool
    reason: str
    path_condition: Optional[str] = None
    variable_assignments: Optional[dict] = None