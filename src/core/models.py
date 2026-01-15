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
from typing import Optional


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
    # Memory management (for detecting leak/UAF/double-free/mismatch)
    ALLOCATOR = auto()        # Returns newly allocated heap memory
    DEALLOCATOR = auto()      # Frees memory at argument N


class MemoryIssueType(Enum):
    """Types of memory safety bugs that CodeQL detects."""
    MEMORY_LEAK = auto()        # Allocated memory never freed
    USE_AFTER_FREE = auto()     # Accessing freed memory
    DOUBLE_FREE = auto()        # Freeing already freed memory
    ALLOC_DEALLOC_MISMATCH = auto()  # new[]/delete, new/free, malloc/delete mismatch


# Mapping: Which hints help detect which bugs
HINT_TO_BUGS = {
    HintType.ALLOCATOR: [MemoryIssueType.MEMORY_LEAK, MemoryIssueType.USE_AFTER_FREE,
                         MemoryIssueType.DOUBLE_FREE, MemoryIssueType.ALLOC_DEALLOC_MISMATCH],
    HintType.DEALLOCATOR: [MemoryIssueType.MEMORY_LEAK, MemoryIssueType.USE_AFTER_FREE,
                           MemoryIssueType.DOUBLE_FREE, MemoryIssueType.ALLOC_DEALLOC_MISMATCH],
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


@dataclass
class HintSet:
    """Collection of memory safety hints for a codebase."""
    hints: dict[str, list[Hint]] = field(default_factory=dict)

    def add(self, hint: Hint) -> None:
        """Add a hint, avoiding duplicates."""
        func_name = hint.function_name
        if func_name not in self.hints:
            self.hints[func_name] = []
        # Avoid duplicates
        for existing in self.hints[func_name]:
            if existing.hint_type == hint.hint_type and existing.target == hint.target:
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

    def get_allocators(self) -> list[str]:
        """Get all allocator function names."""
        return [fn for fn, _ in self.get_by_type(HintType.ALLOCATOR)]

    def get_deallocators(self) -> list[tuple[str, int]]:
        """Get all deallocator function names with their freed argument index."""
        result = []
        for fn, h in self.get_by_type(HintType.DEALLOCATOR):
            arg_idx = h.arg_index if h.arg_index >= 0 else 0
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
                hint_set.add(Hint(
                    function_name=func_name,
                    hint_type=HintType[h["type"]],
                    target=h.get("target", "return"),
                    arg_index=h.get("arg_index", -1),
                    reason=h.get("reason", ""),
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


# =============================================================================
# Analysis Results (CodeQL generates these, Z3 filters them)
# =============================================================================

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