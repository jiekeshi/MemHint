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


class HintType(Enum):
    """Memory safety hints for static analysis.

    These describe FUNCTION SEMANTICS, not bugs.
    LLM identifies these patterns in custom/wrapper functions.

    Examples:
        ALLOCATOR: my_malloc, create_buffer, pool_alloc
        DEALLOCATOR: my_free, destroy_buffer, pool_release
    """
    # Memory management (for detecting leak/UAF/double-free)
    ALLOCATOR = auto()
    DEALLOCATOR = auto()


class MemoryIssueType(Enum):
    """Types of memory safety bugs detected by the pipeline."""
    MEMORY_LEAK = auto()
    MEMORY_LEAK_FILTERED = auto()
    OTHER = auto()




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
    return_expressions: set[str] = field(default_factory=set)



@dataclass
class Hint:
    """A memory safety hint for a function.

    This describes function SEMANTICS, not bugs.
    LLM generates these to help CodeQL understand custom functions.
    """
    function_name: str
    hint_type: HintType
    target: str = "return"
    arg_index: int = -1
    reason: str = ""
    arg_semantics: dict[int, str] = field(default_factory=dict)
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
        for existing in self.hints[func_name]:
            if existing.hint_type == hint.hint_type and existing.target == hint.target:
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
        """Get all allocator function names with their output index."""
        result = []
        for fn, h in self.get_by_type(HintType.ALLOCATOR):
            arg_idx = h.arg_index if h.arg_index is not None else -1
            result.append((fn, arg_idx))
        return result

    def get_deallocators(self) -> list[tuple[str, int]]:
        """Get all deallocator function names with their freed argument index."""
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
                    "name": func_name,
                    "role": "Allocator" if h.hint_type == HintType.ALLOCATOR else "Deallocator",
                    "target": h.target,
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
                role = h.get("role", "")
                if role:
                    role_upper = role.strip().upper()
                    hint_type = HintType.ALLOCATOR if role_upper == "ALLOCATOR" else HintType.DEALLOCATOR
                else:
                    hint_type = HintType[h["type"]]

                target = h.get("target", "return")

                if target == "return":
                    arg_index = -1
                elif target.startswith("arg"):
                    try:
                        arg_index = int(target[3:])
                    except ValueError:
                        arg_index = 0
                else:
                    arg_index = h.get("arg_index", -1)

                hint_set.add(Hint(
                    function_name=func_name,
                    hint_type=hint_type,
                    target=target,
                    arg_index=arg_index,
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
from typing import Any, Dict, Optional


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

    validated: bool = False
    validation_error: str = ""

    # When this object was created (ISO8601, UTC)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )

    # New-style per-bug-type filter blocks as returned by the LLM.
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


    def to_json(self) -> Dict[str, Any]:
        """Export as JSON."""
        out: Dict[str, Any] = {"queries": {}, "non_special": {}}

        for func_name, q in self.queries.items():
            never_freed_block: Dict[str, Any] = dict(q.memory_never_freed_filter or {})
            may_not_be_freed_block: Dict[str, Any] = dict(q.memory_may_not_be_freed_filter or {})

            out["queries"][func_name] = {
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
        """Import from JSON with backward compatibility."""
        qs = cls()

        for func_name, q in (data.get("queries") or {}).items():
            reason = ""
            validated = False
            validation_error = ""
            created_at = ""
            df_block: Dict[str, Any] = {}
            uaf_block: Dict[str, Any] = {}
            never_freed_block: Dict[str, Any] = {}
            may_not_be_freed_block: Dict[str, Any] = {}
            leak_block: Dict[str, Any] = {}

            if isinstance(q, str):
                continue
            elif isinstance(q, dict):
                if (
                    "memory_never_freed_filter" in q
                    or "memory_may_not_be_freed_filter" in q
                ):
                    never_freed_block = dict(q.get("memory_never_freed_filter") or {})
                    may_not_be_freed_block = dict(q.get("memory_may_not_be_freed_filter") or {})

                    validated = bool(never_freed_block.get("validated", False) or
                                   may_not_be_freed_block.get("validated", False))
                    validation_error = (never_freed_block.get("validation_error", "") or
                                      may_not_be_freed_block.get("validation_error", "") or "")

                    reason = q.get("reason", "") or ""
                    created_at = q.get("created_at", "") or ""
                else:
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
                memory_never_freed_filter=never_freed_block,
                memory_may_not_be_freed_filter=may_not_be_freed_block,
            )
            qs.queries[func_name] = cq

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
            "issue_type": self.issue_type.name,
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
            issue_type=MemoryIssueType[d["issue_type"]],
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
    skipped_phases_4_5: bool = False
    llm_verify_summary: dict | None = None

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
        lines = ["", "=== HINT Analysis Result ==="]
        if self.skipped_phases_4_5:
            lines.append("Skipped Phase 4-5 (used existing memory_safety_bugs.json).")
            if self.llm_verify_summary:
                s = self.llm_verify_summary
                total = s.get("total_functions_verified", 0)
                tp = s.get("tp", 0)
                fp = s.get("fp", 0)
                err = s.get("error", 0)
                kept = s.get("should_keep", tp + err)
                lines.append(f"Phase 6 LLM verify: {total} verified, TP={tp}, FP={fp}, ERROR={err} (kept={kept}).")
            lines.append("")
        by_type = self.bugs_by_type()
        type_lines = [f"  - {t.name}: {len(bugs)}" for t, bugs in by_type.items()]
        type_str = "\n".join(type_lines) if type_lines else "  None"
        lines.extend([
            f"Total bugs: {len(self.confirmed_bugs)}",
            "By type:",
            type_str,
            self.hints.summary(),
            f"Iterations: {self.iterations}",
            f"Spurious warnings filtered: {self.spurious_filtered}",
            "",
        ])
        return "\n".join(lines)



@dataclass
class ValidationResult:
    """Result of Z3 validation for a hint or warning."""
    is_valid: bool
    reason: str


@dataclass
class PathFeasibilityResult:
    """Result of Z3 path feasibility check for a warning."""
    is_feasible: bool
    reason: str
