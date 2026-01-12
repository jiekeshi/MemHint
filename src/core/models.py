"""Core data structures for HINT pipeline - Memory Safety Analysis.

TODO: Add more issue types? But I think these are sufficient for now and a few not used yet.
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class AnnotationType(Enum):
    """Memory-related annotation types for static analysis."""

    # === Allocation ===
    ALLOC_SOURCE = auto()        # Returns newly allocated heap memory
    REALLOC = auto()             # Reallocates memory (like realloc)
    ARRAY_ALLOC = auto()         # Allocates array (like calloc)

    # === Deallocation ===
    FREE_SINK = auto()           # Frees memory at argument N
    FREE_RETURN = auto()         # Frees and returns (like realloc failure path)

    # === Ownership Transfer ===
    OWNERSHIP_TRANSFER = auto()  # Ownership transferred to callee
    OWNERSHIP_RETURN = auto()    # Ownership transferred to caller via return
    OWNERSHIP_ARG_OUT = auto()   # Ownership transferred via output parameter

    # === Null Safety ===
    MUST_CHECK_NULL = auto()     # Return value must be null-checked before use
    NULLABLE_RETURN = auto()     # May return NULL
    NONNULL_RETURN = auto()      # Never returns NULL
    NONNULL_ARG = auto()         # Argument must not be NULL

    # === Memory Properties ===
    NO_ESCAPE = auto()           # Memory doesn't escape function scope
    ESCAPE_RETURN = auto()       # Memory escapes via return value
    ESCAPE_ARG = auto()          # Memory escapes via argument (stored elsewhere)

    # === Negative Annotations (prevent false positives) ===
    NOT_ALLOC = auto()           # Explicitly NOT an allocator
    NOT_FREE = auto()            # Explicitly NOT a deallocator
    STATIC_BUFFER = auto()       # Returns static/global buffer (not heap)
    STACK_BUFFER = auto()        # Returns stack buffer (dangerous)
    BORROWED_REF = auto()        # Returns borrowed reference (don't free)

    # === Bug Detection Annotations (LLM-detected potential bugs) ===
    POTENTIAL_LEAK = auto()      # LLM detected potential memory leak
    USE_AFTER_FREE = auto()      # LLM detected potential use-after-free
    DOUBLE_FREE = auto()         # LLM detected potential double-free
    NULL_DEREF = auto()          # LLM detected potential null dereference


class MemoryIssueType(Enum):
    """Types of memory safety issues that can be detected."""

    # Resource management
    MEMORY_LEAK = auto()           # Allocated memory never freed
    RESOURCE_LEAK = auto()         # File handle, socket, etc. not closed

    # Use after invalidation
    USE_AFTER_FREE = auto()        # Accessing freed memory
    DOUBLE_FREE = auto()           # Freeing already freed memory
    USE_AFTER_RETURN = auto()      # Using pointer to stack after return

    # Null pointer
    NULL_DEREFERENCE = auto()      # Dereferencing NULL pointer
    POTENTIAL_NULL_DEREF = auto()  # May dereference NULL on some paths

    # Uninitialized
    UNINITIALIZED_READ = auto()    # Reading uninitialized memory
    UNINITIALIZED_PTR = auto()     # Using uninitialized pointer

    # Buffer issues
    BUFFER_OVERFLOW = auto()       # Writing beyond buffer bounds
    BUFFER_UNDERFLOW = auto()      # Writing before buffer start
    STACK_OVERFLOW = auto()        # Stack buffer overflow
    HEAP_OVERFLOW = auto()         # Heap buffer overflow


class ConflictType(Enum):
    """Types of conflicts found during validation."""
    FALSE_ALLOC = auto()
    FALSE_FREE = auto()
    MISSING_FREE = auto()
    OWNERSHIP_TRANSFER = auto()
    PATH_INFEASIBLE = auto()
    STATIC_BUFFER = auto()
    NULL_CHECK_EXISTS = auto()
    BORROWED_REFERENCE = auto()


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
    return_expressions: set[str] = field(default_factory=set)
    return_type: str = ""


@dataclass
class Annotation:
    """A memory annotation for a function."""
    function_name: str
    annotation_type: AnnotationType
    target: str  # "return", "arg0", "arg1", variable name, etc.
    arg_index: int = -1  # For multi-arg annotations
    confidence: float = 1.0
    reason: str = ""
    line_number: int = None  # For bug annotations: which line
    condition: str = ""  # For bug annotations: under what condition

    def is_bug_annotation(self) -> bool:
        """Check if this is a bug detection annotation (vs function property)."""
        return self.annotation_type in (
            AnnotationType.POTENTIAL_LEAK,
            AnnotationType.USE_AFTER_FREE,
            AnnotationType.DOUBLE_FREE,
            AnnotationType.NULL_DEREF,
        )

    def to_codeql_kind(self) -> str:
        """Convert to CodeQL model kind."""
        mapping = {
            AnnotationType.ALLOC_SOURCE: "alloc",
            AnnotationType.ARRAY_ALLOC: "alloc",
            AnnotationType.FREE_SINK: "free",
            AnnotationType.REALLOC: "realloc",
        }
        return mapping.get(self.annotation_type, "")


@dataclass
class AnnotationSet:
    """Collection of annotations."""
    annotations: dict[str, list[Annotation]] = field(default_factory=dict)

    def add(self, ann: Annotation) -> None:
        if ann.function_name not in self.annotations:
            self.annotations[ann.function_name] = []
        # Avoid duplicates
        for existing in self.annotations[ann.function_name]:
            if existing.annotation_type == ann.annotation_type and existing.target == ann.target:
                return
        self.annotations[ann.function_name].append(ann)

    def remove(self, func_name: str, ann_type: AnnotationType) -> bool:
        if func_name in self.annotations:
            orig_len = len(self.annotations[func_name])
            self.annotations[func_name] = [
                a for a in self.annotations[func_name] if a.annotation_type != ann_type
            ]
            return len(self.annotations[func_name]) < orig_len
        return False

    def get_by_type(self, ann_type: AnnotationType) -> list[tuple[str, Annotation]]:
        """Get all annotations of a specific type."""
        result = []
        for func_name, anns in self.annotations.items():
            for ann in anns:
                if ann.annotation_type == ann_type:
                    result.append((func_name, ann))
        return result

    def get_alloc_functions(self) -> list[str]:
        """Get all allocation function names."""
        alloc_types = {AnnotationType.ALLOC_SOURCE, AnnotationType.ARRAY_ALLOC, AnnotationType.REALLOC}
        return list({fn for fn, ann in self.annotations.items()
                     for a in ([ann] if isinstance(ann, Annotation) else ann)
                     if a.annotation_type in alloc_types})

    def get_free_functions(self) -> list[str]:
        """Get all deallocation function names."""
        return [fn for fn, _ in self.get_by_type(AnnotationType.FREE_SINK)]

    def to_json(self) -> dict:
        """Export as generic JSON format."""
        result = {
            "functions": {},
            "bug_hints": []
        }

        for func_name, anns in self.annotations.items():
            func_anns = []
            for ann in anns:
                if ann.is_bug_annotation():
                    result["bug_hints"].append({
                        "function": func_name,
                        "type": ann.annotation_type.name,
                        "target": ann.target,
                        "line": ann.line_number,
                        "confidence": ann.confidence,
                        "reason": ann.reason,
                    })
                else:
                    func_anns.append({
                        "type": ann.annotation_type.name,
                        "target": ann.target,
                        "reason": ann.reason,
                    })

            if func_anns:
                result["functions"][func_name] = func_anns

        return result

    def to_codeql_model(self) -> str:
        """Export as CodeQL model extension YAML."""
        lines = ["extensions:"]

        # Allocation functions - using summaryModel for custom allocators
        alloc_funcs = (self.get_by_type(AnnotationType.ALLOC_SOURCE) +
                       self.get_by_type(AnnotationType.ARRAY_ALLOC))
        if alloc_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in alloc_funcs:
                # Format: [namespace, type, subtypes, name, signature, ext, input, output, kind]
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # Also add to sourceModel for allocation tracking
        if alloc_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: sourceModel",
                "    data:",
            ])
            for func_name, _ in alloc_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "ReturnValue", "alloc", "manual"]')

        # Free functions - sinkModel
        free_funcs = self.get_by_type(AnnotationType.FREE_SINK)
        if free_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: sinkModel",
                "    data:",
            ])
            for func_name, ann in free_funcs:
                arg_idx = int(ann.target[3:]) if ann.target.startswith("arg") else 0
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "Argument[{arg_idx}]", "free", "manual"]')

        # Realloc functions
        realloc_funcs = self.get_by_type(AnnotationType.REALLOC)
        if realloc_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: reallocExpr",
                "    data:",
            ])
            for func_name, _ in realloc_funcs:
                lines.append(f'      - ["{func_name}"]')

        return "\n".join(lines)


@dataclass
class Warning:
    """A warning from static analyzer."""
    file_path: str
    line_number: int
    function_name: str
    warning_type: str
    message: str
    issue_type: MemoryIssueType = MemoryIssueType.MEMORY_LEAK
    allocation_site: str = ""
    trace: list[str] = field(default_factory=list)


@dataclass
class CounterExample:
    """Counter-example for spurious warning."""
    warning: Warning
    conflict_type: ConflictType
    reason: str
    blamed_annotation: Annotation = None


@dataclass
class Evidence:
    """Evidence for confirmed bug."""
    warning: Warning
    concrete_trace: list[str]
    leak_point: str
    suggested_fix: str = ""


@dataclass
class AnalysisResult:
    """Final analysis result."""
    confirmed_bugs: list[Evidence]
    final_annotations: AnnotationSet
    iterations: int
    spurious_filtered: int

    def bugs_by_type(self) -> dict[MemoryIssueType, list[Evidence]]:
        """Group bugs by issue type."""
        result = {}
        for bug in self.confirmed_bugs:
            t = bug.warning.issue_type
            if t not in result:
                result[t] = []
            result[t].append(bug)
        return result

    def summary(self) -> str:
        by_type = self.bugs_by_type()
        type_str = "\n".join(f"  - {t.name}: {len(bugs)}" for t, bugs in by_type.items())
        if not type_str:
            type_str = "  None"

        return f"""
=== HINT Analysis Result ===
Total bugs: {len(self.confirmed_bugs)}
By type:
{type_str}
Annotations generated: {len(self.final_annotations.annotations)}
CEGAR iterations: {self.iterations}
Spurious warnings filtered: {self.spurious_filtered}
"""