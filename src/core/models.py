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


# class ConflictType(Enum):
#     """Types of conflicts found during validation."""
#     FALSE_ALLOC = auto()
#     FALSE_FREE = auto()
#     MISSING_FREE = auto()
#     OWNERSHIP_TRANSFER = auto()
#     PATH_INFEASIBLE = auto()
#     STATIC_BUFFER = auto()
#     NULL_CHECK_EXISTS = auto()
#     BORROWED_REFERENCE = auto()


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

    # def to_codeql_kind(self) -> str:
    #     """Convert to CodeQL model kind."""
    #     mapping = {
    #         AnnotationType.ALLOC_SOURCE: "alloc",
    #         AnnotationType.ARRAY_ALLOC: "alloc",
    #         AnnotationType.FREE_SINK: "free",
    #         AnnotationType.REALLOC: "realloc",
    #     }
    #     return mapping.get(self.annotation_type, "")


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

    # def get_free_functions(self) -> list[str]:
    #     """Get all deallocation function names."""
    #     return [fn for fn, _ in self.get_by_type(AnnotationType.FREE_SINK)]

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
        """Export as CodeQL model extension YAML.
        
        Supported AnnotationType values:
        - ALLOC_SOURCE: Allocation functions (added to summaryModel and sourceModel)
        - ARRAY_ALLOC: Array allocation functions (added to summaryModel and sourceModel)
        - REALLOC: Reallocation functions (added to reallocExpr)
        - FREE_SINK: Deallocation functions (added to sinkModel)
        - FREE_RETURN: Free and return functions (added to sinkModel)
        - OWNERSHIP_TRANSFER: Ownership transfer to callee (added to summaryModel)
        - OWNERSHIP_RETURN: Ownership transfer to caller via return (added to summaryModel)
        - OWNERSHIP_ARG_OUT: Ownership transfer via output parameter (added to summaryModel)
        - NULLABLE_RETURN: May return NULL (added to summaryModel)
        - NONNULL_RETURN: Never returns NULL (added to summaryModel)
        - NONNULL_ARG: Argument must not be NULL (added to summaryModel)
        - MUST_CHECK_NULL: Return value must be null-checked (added to summaryModel)
        - NO_ESCAPE: Memory doesn't escape function scope (added to summaryModel)
        - ESCAPE_RETURN: Memory escapes via return value (added to summaryModel)
        - ESCAPE_ARG: Memory escapes via argument (added to summaryModel)
        - STATIC_BUFFER: Returns static/global buffer (added to summaryModel)
        - STACK_BUFFER: Returns stack buffer (added to summaryModel)
        - BORROWED_REF: Returns borrowed reference (added to summaryModel)
        
        Note: Bug detection annotations (POTENTIAL_LEAK, USE_AFTER_FREE, DOUBLE_FREE, NULL_DEREF)
        and negative annotations (NOT_ALLOC, NOT_FREE) are not exported to CodeQL models.
        """
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

        # Free return functions - sinkModel (frees and returns)
        free_return_funcs = self.get_by_type(AnnotationType.FREE_RETURN)
        if free_return_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: sinkModel",
                "    data:",
            ])
            for func_name, ann in free_return_funcs:
                arg_idx = int(ann.target[3:]) if ann.target.startswith("arg") else 0
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "Argument[{arg_idx}]", "free", "manual"]')

        # Ownership transfer functions - summaryModel
        ownership_transfer_funcs = self.get_by_type(AnnotationType.OWNERSHIP_TRANSFER)
        if ownership_transfer_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, ann in ownership_transfer_funcs:
                arg_idx = int(ann.target[3:]) if ann.target.startswith("arg") else 0
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "Argument[{arg_idx}]", "ReturnValue", "taint"]')

        # Ownership return functions - summaryModel
        ownership_return_funcs = self.get_by_type(AnnotationType.OWNERSHIP_RETURN)
        if ownership_return_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in ownership_return_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # Ownership arg out functions - summaryModel
        ownership_arg_out_funcs = self.get_by_type(AnnotationType.OWNERSHIP_ARG_OUT)
        if ownership_arg_out_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, ann in ownership_arg_out_funcs:
                arg_idx = int(ann.target[3:]) if ann.target.startswith("arg") else 0
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "Argument[{arg_idx}]", "Argument[{arg_idx}]", "taint"]')

        # Nullable return functions - summaryModel
        nullable_return_funcs = self.get_by_type(AnnotationType.NULLABLE_RETURN)
        if nullable_return_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in nullable_return_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # Nonnull return functions - summaryModel
        nonnull_return_funcs = self.get_by_type(AnnotationType.NONNULL_RETURN)
        if nonnull_return_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in nonnull_return_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # Nonnull arg functions - summaryModel
        nonnull_arg_funcs = self.get_by_type(AnnotationType.NONNULL_ARG)
        if nonnull_arg_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, ann in nonnull_arg_funcs:
                arg_idx = int(ann.target[3:]) if ann.target.startswith("arg") else 0
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "Argument[{arg_idx}]", "Argument[{arg_idx}]", "taint"]')

        # Must check null functions - summaryModel
        must_check_null_funcs = self.get_by_type(AnnotationType.MUST_CHECK_NULL)
        if must_check_null_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in must_check_null_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # No escape functions - summaryModel
        no_escape_funcs = self.get_by_type(AnnotationType.NO_ESCAPE)
        if no_escape_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in no_escape_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # Escape return functions - summaryModel
        escape_return_funcs = self.get_by_type(AnnotationType.ESCAPE_RETURN)
        if escape_return_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in escape_return_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # Escape arg functions - summaryModel
        escape_arg_funcs = self.get_by_type(AnnotationType.ESCAPE_ARG)
        if escape_arg_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, ann in escape_arg_funcs:
                arg_idx = int(ann.target[3:]) if ann.target.startswith("arg") else 0
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "Argument[{arg_idx}]", "Argument[{arg_idx}]", "taint"]')

        # Static buffer functions - summaryModel (prevent false positives)
        static_buffer_funcs = self.get_by_type(AnnotationType.STATIC_BUFFER)
        if static_buffer_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in static_buffer_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # Stack buffer functions - summaryModel (prevent false positives)
        stack_buffer_funcs = self.get_by_type(AnnotationType.STACK_BUFFER)
        if stack_buffer_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in stack_buffer_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

        # Borrowed ref functions - summaryModel (prevent false positives)
        borrowed_ref_funcs = self.get_by_type(AnnotationType.BORROWED_REF)
        if borrowed_ref_funcs:
            lines.extend([
                "  - addsTo:",
                "      pack: codeql/cpp-all",
                "      extensible: summaryModel",
                "    data:",
            ])
            for func_name, _ in borrowed_ref_funcs:
                lines.append(f'      - ["", "", False, "{func_name}", "", "", "", "ReturnValue", "taint"]')

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