"""Z3-based Validator for Memory Safety Analysis.

This module provides two validation functions:

1. validate_hints(): Validates LLM-generated hints
   - Checks if ALLOCATOR hint is valid (function actually allocates)
   - Checks if DEALLOCATOR hint is valid (function actually frees)
   - Filters out impossible/incorrect hints
   - Supports transitive call chain analysis (wrapper functions)
   - Supports alias analysis and indirect calls

2. validate_warnings(): Filters CodeQL warnings by path feasibility
   - Checks if the bug path is actually reachable
   - Filters out false positives from infeasible paths
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from collections import defaultdict

try:
    from z3 import (
        Solver, Bool, Int, And, Or, Not, Implies,
        sat, unsat, BoolRef, AtMost, BoolVal
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logging.warning("Z3 not available - install with: pip install z3-solver")

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

from src.core.models import (
    FunctionInfo, Hint, HintType, HintSet,
    Warning, MemoryIssueType, ValidationResult, PathFeasibilityResult
)

logger = logging.getLogger(__name__)


# Memory-management functions known before any summary exists: the C
# library's primitives and those of libraries outside the analysed tree, whose
# bodies the LLM never sees. Everything project-specific comes from the
# validated summaries.
STANDARD_ALLOCATORS: frozenset[str] = frozenset({
    # C library
    "malloc", "calloc", "realloc", "reallocarray", "strdup", "strndup",
    "aligned_alloc", "memalign", "valloc", "pvalloc",
    # Linux kernel
    "kmalloc", "kzalloc", "kcalloc", "krealloc", "vmalloc", "vzalloc",
    "kvmalloc", "kvzalloc",
    # GLib
    "g_malloc", "g_malloc0", "g_new", "g_new0",
})
STANDARD_DEALLOCATORS: frozenset[str] = frozenset({
    "free", "cfree",
    "kfree", "vfree", "kvfree", "kfree_sensitive",
    "g_free",
})
# Calls that take ownership of a pointer argument by inserting it into a
# container; Phase 5 treats a pointer passed to one of these as escaped.
OWNERSHIP_SINKS: frozenset[str] = frozenset({
    # BSD queue.h / tree.h
    "TAILQ_INSERT_HEAD", "TAILQ_INSERT_TAIL", "TAILQ_INSERT_AFTER", "TAILQ_INSERT_BEFORE",
    "LIST_INSERT_HEAD", "LIST_INSERT_AFTER", "LIST_INSERT_BEFORE",
    "SLIST_INSERT_HEAD", "SLIST_INSERT_AFTER",
    "STAILQ_INSERT_HEAD", "STAILQ_INSERT_TAIL", "STAILQ_INSERT_AFTER",
    "RB_INSERT", "SPLAY_INSERT",
    # Linux kernel
    "list_add", "list_add_tail", "hlist_add_head",
})


def _param_alias_names(func: FunctionInfo, arg_index: int) -> set[str]:
    """Parameter `arg_index` and the local variables assigned from it
    (v = p, and its transitive closure), by SimpleAliasAnalyzer."""
    names: set[str] = set()
    arg_names = getattr(func, "arg_names", None) or []
    if 0 <= arg_index < len(arg_names) and arg_names[arg_index]:
        names.add(arg_names[arg_index])
    try:
        alias_info = SimpleAliasAnalyzer().analyze(func)
        for var, idx in alias_info.param_aliases.items():
            if idx == arg_index:
                names.add(var)
    except Exception as e:
        logger.debug(f"alias analysis failed for {func.name}: {e}")
    # A pointer-to-pointer parameter: freeing *pp frees the object it refers to.
    arg_types = getattr(func, "arg_types", None) or []
    if 0 <= arg_index < len(arg_names) and arg_index < len(arg_types) and "**" in (arg_types[arg_index] or ""):
        names.add("*" + arg_names[arg_index])
    return {n for n in names if n}


def _frees_one_of(code: str, free_funcs: set[str], variables: set[str]) -> bool:
    """True iff `code` contains `f(v)` for a deallocator f and a variable v
    (the variable itself, not a member or container of it)."""
    if not code or not variables:
        return False
    funcs = set(free_funcs) | set(STANDARD_DEALLOCATORS)
    cast = r"(?:\(\s*[A-Za-z_][\w\s]*\**\s*\)\s*)*"
    for f in funcs:
        for v in variables:
            target = (r"\*\s*" + re.escape(v[1:])) if v.startswith("*") else re.escape(v)
            if re.search(rf"\b{re.escape(f)}\s*\(\s*{cast}\(?\s*{target}\s*\)?\s*\)", code):
                return True
    return False

C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())
_CPP_SUFFIXES = (".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".C", ".CPP")


def _is_cpp(file_path: str) -> bool:
    return file_path.endswith(_CPP_SUFFIXES)

_ASSIGN_RE = re.compile(r"^\s*(?P<lhs>.+?)\s*=\s*(?P<rhs>.+?);?\s*$")
_RETURN_RE = re.compile(r"^\s*return\s+(?P<expr>.+?);?\s*$")
_RETURN_ONLY_VAR_RE = re.compile(
    r"^\s*return\s*\(*\s*(?P<var>[A-Za-z_]\w*)\s*\)*\s*;?\s*$"
)


def _strip_c_comments(code: str) -> str:
    """Remove /* */ and // comments (string literals are not special-cased)."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", code, flags=re.DOTALL)


class NodeType(Enum):
    """Types of CFG nodes."""
    ENTRY = auto()
    EXIT = auto()
    ALLOC = auto()
    FREE = auto()
    RETURN = auto()
    BRANCH = auto()
    ASSIGN = auto()
    DEREF = auto()
    GOTO = auto()
    LABEL = auto()


@dataclass
class CFGNode:
    id: int
    node_type: NodeType
    line: int
    code: str
    variable: str = ""
    obj: str = ""
    condition: str = ""
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)


@dataclass
class CallSite:
    """Represents a function call site."""
    caller: str
    callee: str
    line: int
    arg_mapping: dict[int, int] = field(default_factory=dict)
    arg_expressions: dict[int, str] = field(default_factory=dict)
    is_indirect: bool = False


@dataclass
class CallGraph:
    """Call graph for interprocedural analysis."""
    callsites: dict[str, list[CallSite]] = field(default_factory=lambda: defaultdict(list))
    callers: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    callees: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))


class CallGraphBuilder:
    """Build call graph from source code."""

    def __init__(self):
        self.parser = Parser(C_LANGUAGE)
        self.cpp_parser = Parser(CPP_LANGUAGE)

    def build(self, functions) -> CallGraph:
        """Build call graph from all functions."""
        cg = CallGraph()

        for func_name, func_info in functions.items():
            if func_info is None:
                continue
            self._extract_calls(func_name, func_info, cg)

        return cg


    def _extract_calls(self, func_name: str, func_info, cg: CallGraph):
        """Extract all call sites from a function."""
        if isinstance(func_info, dict):
            code = func_info.get("code", "") or ""
        else:
            code = getattr(func_info, "code", "") or ""

        if not isinstance(code, str):
            code = str(code)

        src_bytes = code.encode("utf-8", errors="ignore")
        parser = self.cpp_parser if _is_cpp(getattr(func_info, "file_path", "")) else self.parser
        tree = parser.parse(src_bytes)

        for node in self._walk(tree.root_node):
            if node.type == "call_expression":
                callsite = self._process_call(func_name, func_info, node, src_bytes)
                if callsite:
                    cg.callsites[func_name].append(callsite)
                    if not callsite.is_indirect:
                        cg.callers[callsite.callee].add(func_name)
                        cg.callees[func_name].add(callsite.callee)


    def _walk(self, node):
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _process_call(
        self,
        caller: str,
        caller_info,
        node,
        source: bytes
    ) -> Optional[CallSite]:
        func_node = node.child_by_field_name("function")
        if not func_node:
            return None

        callee = source[func_node.start_byte:func_node.end_byte].decode(errors="ignore").strip()
        
        base = 0
        if isinstance(caller_info, dict):
            base = int(caller_info.get("start_line", 0) or 0)
        else:
            base = int(getattr(caller_info, "start_line", 0) or 0)

        line = (node.start_point[0] + 1) + (base - 1 if base > 0 else 0)

        is_indirect = func_node.type != "identifier"

        if isinstance(caller_info, dict):
            arg_names = caller_info.get("arg_names", []) or []
        else:
            arg_names = getattr(caller_info, "arg_names", []) or []

        arg_mapping: dict[int, int] = {}
        arg_expressions: dict[int, str] = {}

        args_node = node.child_by_field_name("arguments")
        if args_node:
            arg_idx = 0
            for child in args_node.children:
                if child.type in (
                    "identifier", "pointer_expression", "cast_expression",
                    "number_literal", "string_literal", "call_expression",
                    "subscript_expression", "field_expression", "parenthesized_expression"
                ):
                    expr = source[child.start_byte:child.end_byte].decode(errors="ignore").strip()
                    arg_expressions[arg_idx] = expr

                    if child.type == "identifier" and expr in arg_names:
                        arg_mapping[arg_idx] = arg_names.index(expr)
                    else:
                        arg_mapping[arg_idx] = -1

                    arg_idx += 1

        return CallSite(
            caller=caller,
            callee=callee,
            line=line,
            arg_mapping=arg_mapping,
            arg_expressions=arg_expressions,
            is_indirect=is_indirect
        )


@dataclass
class AliasInfo:
    """Track variable aliases within a function."""
    # Maps variable -> set of variables it may alias
    aliases: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Maps variable -> parameter index if it aliases a parameter
    param_aliases: dict[str, int] = field(default_factory=dict)


class SimpleAliasAnalyzer:
    """Simple intraprocedural alias analysis."""

    def __init__(self):
        self.parser = Parser(C_LANGUAGE)
        self.cpp_parser = Parser(CPP_LANGUAGE)

    def analyze(self, func_info: FunctionInfo) -> AliasInfo:
        """Analyze aliases in a function."""
        parser = self.cpp_parser if _is_cpp(func_info.file_path) else self.parser
        tree = parser.parse(func_info.code.encode())
        source = func_info.code.encode()

        alias_info = AliasInfo()

        for idx, param in enumerate(func_info.arg_names):
            alias_info.aliases[param].add(param)
            alias_info.param_aliases[param] = idx

        for node in self._walk(tree.root_node):
            if node.type == "assignment_expression":
                self._process_assignment(node, source, func_info, alias_info)
            elif node.type == "init_declarator":
                self._process_init(node, source, func_info, alias_info)

        self._compute_transitive_closure(alias_info)

        return alias_info

    def _walk(self, node):
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _process_assignment(
        self,
        node,
        source: bytes,
        func_info: FunctionInfo,
        alias_info: AliasInfo
    ):
        """Process assignment: lhs = rhs"""
        lhs = node.child_by_field_name("left")
        rhs = node.child_by_field_name("right")

        if not lhs or not rhs:
            return

        rhs_name = self._same_object(rhs, source, func_info)
        if lhs.type == "identifier" and rhs_name:
            lhs_name = source[lhs.start_byte:lhs.end_byte].decode()

            alias_info.aliases[lhs_name].add(rhs_name)
            alias_info.aliases[lhs_name].update(alias_info.aliases.get(rhs_name, set()))

            if rhs_name in alias_info.param_aliases:
                alias_info.param_aliases[lhs_name] = alias_info.param_aliases[rhs_name]

    def _process_init(
        self,
        node,
        source: bytes,
        func_info: FunctionInfo,
        alias_info: AliasInfo
    ):
        """Process initialization: type var = expr (v = p, where p is a name)"""
        declarator = node.child_by_field_name("declarator")
        value = node.child_by_field_name("value")
        if declarator is None or value is None:
            return
        val_name = self._same_object(value, source, func_info)
        if not val_name:
            return
        # Descend through pointer/parenthesized declarators to the name.
        while declarator is not None and declarator.type != "identifier":
            inner = declarator.child_by_field_name("declarator")
            if inner is None:
                inner = next((c for c in declarator.children if c.type in ("identifier", "pointer_declarator", "parenthesized_declarator")), None)
            declarator = inner
        if declarator is None:
            return

        var_name = source[declarator.start_byte:declarator.end_byte].decode()

        alias_info.aliases[var_name].add(val_name)
        alias_info.aliases[var_name].update(alias_info.aliases.get(val_name, set()))

        if val_name in alias_info.param_aliases:
            alias_info.param_aliases[var_name] = alias_info.param_aliases[val_name]

    @staticmethod
    def _same_object(node, source: bytes, func_info: FunctionInfo) -> str:
        """The name whose object `node` denotes: `p`, `(T *)p`, `(p)`, or `*pp`
        for a pointer-to-pointer parameter pp (the object pp refers to)."""
        while node is not None and node.type in ("cast_expression", "parenthesized_expression"):
            inner = node.child_by_field_name("value")
            if inner is None:
                named = [c for c in node.children if c.is_named and c.type != "type_descriptor"]
                inner = named[-1] if named else None
            node = inner
        if node is None:
            return ""
        if node.type == "identifier":
            return source[node.start_byte:node.end_byte].decode()
        if node.type == "pointer_expression" and source[node.start_byte:node.start_byte + 1] == b"*":
            arg = node.child_by_field_name("argument")
            if arg is not None and arg.type == "identifier":
                name = source[arg.start_byte:arg.end_byte].decode()
                names = list(getattr(func_info, "arg_names", None) or [])
                types = list(getattr(func_info, "arg_types", None) or [])
                if name in names and "**" in (types[names.index(name)] if names.index(name) < len(types) else ""):
                    return name
        return ""

    def _compute_transitive_closure(self, alias_info: AliasInfo):
        """Compute transitive closure of alias relation."""
        changed = True
        while changed:
            changed = False
            for var, aliases in list(alias_info.aliases.items()):
                for alias in list(aliases):
                    if alias in alias_info.aliases:
                        new_aliases = alias_info.aliases[alias] - aliases
                        if new_aliases:
                            alias_info.aliases[var].update(new_aliases)
                            changed = True


class TransitiveAnalyzer:
    """Analyze transitive allocator/deallocator relationships."""

    # Standard primitives; project wrappers come from summaries.
    BASE_ALLOCATORS = STANDARD_ALLOCATORS
    BASE_DEALLOCATORS = STANDARD_DEALLOCATORS

    def __init__(self, call_graph: CallGraph, functions: dict[str, FunctionInfo]):
        self.cg = call_graph
        self.functions = functions
        self.parser = Parser(C_LANGUAGE)
        self.alias_analyzer = SimpleAliasAnalyzer()

        self._is_allocator_cache: dict[str, tuple[bool, list[str]]] = {}
        self._is_deallocator_cache: dict[str, tuple[bool, int, list[str]]] = {}
        self._alias_cache: dict[str, AliasInfo] = {}

        self._known_deallocators: set[str] = set(self.BASE_DEALLOCATORS)
        self._precompute_known_deallocators()

    def _precompute_known_deallocators(self):
        """Pre-scan functions to find obvious deallocators."""
        for func_name, func_info in self.functions.items():
            code = func_info.code
            for dealloc in self.BASE_DEALLOCATORS:
                if f"{dealloc}(" in code:
                    self._known_deallocators.add(func_name)
                    break

    def _get_alias_info(self, func_name: str) -> Optional[AliasInfo]:
        """Get cached alias info for function."""
        if func_name not in self._alias_cache:
            if func_name in self.functions:
                self._alias_cache[func_name] = self.alias_analyzer.analyze(
                    self.functions[func_name]
                )
            else:
                return None
        return self._alias_cache[func_name]

    def is_transitive_allocator(
        self,
        func_name: str,
        max_depth: int = 10
    ) -> tuple[bool, list[str]]:
        """Check if function is an allocator (directly or transitively)."""
        if func_name in self._is_allocator_cache:
            return self._is_allocator_cache[func_name]

        result = self._check_allocator_recursive(func_name, [], set(), max_depth)
        self._is_allocator_cache[func_name] = result
        return result

    def _check_allocator_recursive(
        self,
        func_name: str,
        chain: list[str],
        visited: set[str],
        depth: int
    ) -> tuple[bool, list[str]]:
        """Recursively check if function allocates memory."""
        if depth <= 0 or func_name in visited:
            return (False, [])

        visited.add(func_name)
        current_chain = chain + [func_name]

        if func_name in self.BASE_ALLOCATORS:
            return (True, current_chain)

        if func_name not in self.functions:
            return (False, [])

        func_info = self.functions[func_name]

        if not func_info.return_type or '*' not in func_info.return_type:
            return (False, [])

        for callsite in self.cg.callsites.get(func_name, []):
            if callsite.is_indirect:
                continue

            callee = callsite.callee
            is_alloc, result_chain = self._check_allocator_recursive(
                callee, current_chain, visited.copy(), depth - 1
            )

            if is_alloc:
                if self._returns_call_result(func_info, callsite.callee):
                    return (True, result_chain)

        return (False, [])

    def _returns_call_result(self, func_info: FunctionInfo, callee: str) -> bool:
        """Check if function returns the result of calling callee."""
        code = func_info.code

        if re.search(rf'return\s+{re.escape(callee)}\s*\(', code):
            return True

        match = re.search(rf'(\w+)\s*=\s*{re.escape(callee)}\s*\(', code)
        if match:
            var = match.group(1)
            if re.search(rf'return\s+{re.escape(var)}\s*;', code):
                return True

        return False

    def is_transitive_deallocator(
        self,
        func_name: str,
        max_depth: int = 10
    ) -> tuple[bool, int, list[str]]:
        """Check if function is a deallocator."""
        if func_name in self._is_deallocator_cache:
            return self._is_deallocator_cache[func_name]

        result = self._check_deallocator_enhanced(func_name, max_depth)
        self._is_deallocator_cache[func_name] = result
        return result

    def _check_deallocator_enhanced(
        self,
        func_name: str,
        max_depth: int
    ) -> tuple[bool, int, list[str]]:
        """Enhanced deallocator check with alias support."""
        if func_name in self.BASE_DEALLOCATORS:
            return (True, 0, [func_name])

        if func_name not in self.functions:
            return (False, -1, [])

        func_info = self.functions[func_name]
        alias_info = self._get_alias_info(func_name)

        result = self._check_deallocator_recursive(func_name, [], set(), max_depth)
        if result[0]:
            return result

        result = self._check_deallocator_via_alias(func_name, func_info, alias_info, max_depth)
        if result[0]:
            return result

        result = self._check_deallocator_pattern(func_name, func_info)
        if result[0]:
            return result

        return (False, -1, [])

    def _check_deallocator_recursive(
        self,
        func_name: str,
        chain: list[str],
        visited: set[str],
        depth: int
    ) -> tuple[bool, int, list[str]]:
        """Recursively check if function frees memory via call graph."""
        if depth <= 0 or func_name in visited:
            return (False, -1, [])

        visited.add(func_name)
        current_chain = chain + [func_name]

        if func_name in self.BASE_DEALLOCATORS:
            return (True, 0, current_chain)

        if func_name not in self.functions:
            return (False, -1, [])

        for callsite in self.cg.callsites.get(func_name, []):
            if callsite.is_indirect:
                continue

            callee = callsite.callee

            is_dealloc, callee_freed_arg, result_chain = self._check_deallocator_recursive(
                callee, current_chain, visited.copy(), depth - 1
            )

            if is_dealloc:
                if callee_freed_arg in callsite.arg_mapping:
                    our_arg_idx = callsite.arg_mapping[callee_freed_arg]
                    if our_arg_idx >= 0:
                        return (True, our_arg_idx, result_chain)

                continue

        return (False, -1, [])

    def _check_deallocator_via_alias(
        self,
        func_name: str,
        func_info: FunctionInfo,
        alias_info: Optional[AliasInfo],
        max_depth: int
    ) -> tuple[bool, int, list[str]]:
        """Check if function frees a parameter through an alias."""
        if not alias_info:
            return (False, -1, [])

        code = func_info.code

        for callsite in self.cg.callsites.get(func_name, []):
            callee = callsite.callee

            if callee not in self._known_deallocators:
                continue

            for arg_idx, arg_expr in callsite.arg_expressions.items():
                if arg_expr in alias_info.param_aliases:
                    param_idx = alias_info.param_aliases[arg_expr]
                    return (True, param_idx, [func_name, callee])

                for var, aliases in alias_info.aliases.items():
                    if arg_expr in aliases or var == arg_expr:
                        if var in alias_info.param_aliases:
                            param_idx = alias_info.param_aliases[var]
                            return (True, param_idx, [func_name, callee])

        return (False, -1, [])

    def _check_deallocator_pattern(
        self,
        func_name: str,
        func_info: FunctionInfo
    ) -> tuple[bool, int, list[str]]:
        """Pattern-based detection for deallocators."""
        code = func_info.code

        # Visit the deallocation calls in source order so that the result does
        # not depend on the iteration order of the deallocator set.
        found = []
        for dealloc in self._known_deallocators:
            pattern = rf'{re.escape(dealloc)}\s*\(\s*(?:\(\s*[A-Za-z_][\w\s]*\**\s*\)\s*)*(\*?)\s*(\w+)\s*\)'
            for match in re.finditer(pattern, code):
                found.append((match.start(), dealloc, match))
        found.sort(key=lambda t: (t[0], t[1]))

        for _, dealloc, match in found:
            star, freed_var = match.group(1), match.group(2)

            if freed_var in func_info.arg_names:
                param_idx = func_info.arg_names.index(freed_var)
                types = func_info.arg_types or []
                if star and not (param_idx < len(types) and "**" in (types[param_idx] or "")):
                    continue
                return (True, param_idx, [func_name, dealloc])
            if star:
                continue

            derived_patterns = [
                rf'(\w+)\s*\*?\s*{re.escape(freed_var)}\s*=\s*(\w+)',
                rf'{re.escape(freed_var)}\s*=\s*(\w+)->',
                rf'{re.escape(freed_var)}\s*=\s*(\w+)\s*;',
            ]

            for dp in derived_patterns:
                dm = re.search(dp, code)
                if dm:
                    source_var = dm.group(2) if dm.lastindex >= 2 else dm.group(1)
                    if source_var in func_info.arg_names:
                        param_idx = func_info.arg_names.index(source_var)
                        return (True, param_idx, [func_name, dealloc])

        return (False, -1, [])


class CFGBuilder:
    """Build Control Flow Graph from C code."""

    def __init__(
        self,
        alloc_funcs: set[str] = None,
        free_funcs: set[str] = None,
        alloc_out_params: dict[str, int] = None,
        file_path: str = "",
    ):
        self.parser = Parser(CPP_LANGUAGE if _is_cpp(file_path) else C_LANGUAGE)
        self.alloc_funcs = set(alloc_funcs) if alloc_funcs else set(STANDARD_ALLOCATORS)
        self.free_funcs = set(free_funcs) if free_funcs else set(STANDARD_DEALLOCATORS)
        self.alloc_out_params = alloc_out_params or {}


    def build(self, code: str, line_offset: int = 0) -> dict[int, CFGNode]:
        """Build CFG from function code."""
        tree = self.parser.parse(code.encode())
        source = code.encode()

        nodes: dict[int, CFGNode] = {}
        node_id = 0

        def new_node(
            nt: NodeType,
            line: int,
            code_txt: str,
            variable: str = "",
            obj: str = "",
            condition: str = ""
        ) -> int:
            nonlocal node_id
            nid = node_id
            nodes[nid] = CFGNode(
                id=nid,
                node_type=nt,
                line=(line + line_offset) if line > 0 else line,
                code=code_txt,
                variable=variable,
                obj=obj,
                condition=condition,
                successors=[],
                predecessors=[],
            )
            node_id += 1
            return nid


        def add_edge(a: Optional[int], b: Optional[int]):
            # A statement after a return or goto has no predecessor: it is
            # unreachable and simply gets no incoming edge.
            if a is None or b is None:
                return
            if b not in nodes[a].successors:
                nodes[a].successors.append(b)
            if a not in nodes[b].predecessors:
                nodes[b].predecessors.append(a)

        entry_id = new_node(NodeType.ENTRY, 0, "entry")

        body = None
        for ch in tree.root_node.children:
            if ch.type == "function_definition":
                for c2 in ch.children:
                    if c2.type == "compound_statement":
                        body = c2
                        break
                break
        if body is None:
            body = tree.root_node
            
        label_nodes: dict[str, int] = {}
        pending_gotos: list[tuple[int, str]] = []


        exit_id = new_node(NodeType.EXIT, 0, "exit")

        def build_stmt_list(stmt_list_node, in_id: int) -> int:
            cur: Optional[int] = in_id
            last_reachable: int = in_id

            for st in stmt_list_node.children:
                if st.type in ("{", "}"):
                    continue

                tail = build_stmt(st, cur)

                if cur is not None:
                    last_reachable = tail
                    if nodes[tail].node_type in (NodeType.RETURN, NodeType.GOTO):
                        cur = None
                    else:
                        cur = tail

            return last_reachable


        def build_stmt(st_node,in_id: Optional[int]) -> int:
            """Build CFG for a statement; returns the tail node id."""
            if st_node.type == "if_statement":
                return build_if(st_node, in_id)

            if st_node.type == "switch_statement":
                return build_switch(st_node, in_id)
            if st_node.type == "case_statement":
                return build_case(st_node, in_id)

            if st_node.type == "return_statement":
                rid = new_node(
                    NodeType.RETURN,
                    st_node.start_point[0] + 1,
                    source[st_node.start_byte:st_node.end_byte].decode(errors="ignore"),
                )
                if in_id is not None:
                    add_edge(in_id, rid)
                add_edge(rid, exit_id)
                return rid

            if st_node.type == "labeled_statement":
                label_id_node = st_node.child(0)
                label_name = source[label_id_node.start_byte:label_id_node.end_byte].decode(errors="ignore").strip()

                lid = new_node(NodeType.LABEL, st_node.start_point[0] + 1, f"{label_name}:")
                label_nodes[label_name] = lid
                if in_id is not None:
                    add_edge(in_id, lid)


                body_stmt = st_node.child_by_field_name("statement")
                if body_stmt is None:
                    body_stmt = None
                    for ch in reversed(st_node.children):
                        if ch.type not in (":", "identifier"):
                            body_stmt = ch
                            break

                if body_stmt is not None:
                    return build_stmt(body_stmt, lid)
                return lid


            if st_node.type == "goto_statement":
                target_node = None
                for ch in st_node.children:
                    if ch.type in ("identifier", "statement_identifier"):
                        target_node = ch
                        break
                label = source[target_node.start_byte:target_node.end_byte].decode(errors="ignore").strip() if target_node else ""

                gid = new_node(NodeType.GOTO, st_node.start_point[0] + 1,
                            source[st_node.start_byte:st_node.end_byte].decode(errors="ignore").strip())
                if in_id is not None:
                    add_edge(in_id, gid)
                pending_gotos.append((gid, label))

                return gid

            if st_node.type == "compound_statement":
                return build_stmt_list(st_node, in_id)
            if st_node.type in ("for_statement", "while_statement", "do_statement", "for_range_loop"):
                return build_loop(st_node, in_id)

            tail = in_id
            created_any = False

            for n in st_node.children:
                cfg_n = self._process_node(n, source, node_id)
                if cfg_n:
                    nid = new_node(
                        cfg_n.node_type, cfg_n.line, cfg_n.code,
                        cfg_n.variable, cfg_n.obj, cfg_n.condition
                    )
                    add_edge(tail, nid)
                    tail = nid
                    created_any = True

            if created_any:
                return tail

            return in_id
        def is_terminal(nid: int) -> bool:
            # A goto leaves the branch: `if (!p) goto err;` does not fall through.
            return nodes[nid].node_type in (NodeType.RETURN, NodeType.EXIT, NodeType.GOTO)

        def build_loop(lp_node, in_id: int) -> int:
            """Acyclic loop: the body runs zero times or once (no back edge)."""
            cur = in_id
            body = lp_node.child_by_field_name("body")
            ini = lp_node.child_by_field_name("initializer")
            if ini is not None:
                for sub in self._walk(ini):
                    if sub.type in ("assignment_expression", "init_declarator", "call_expression"):
                        cfg_n = self._process_node(sub, source, node_id)
                        if cfg_n is not None:
                            nid = new_node(cfg_n.node_type, cfg_n.line, cfg_n.code,
                                           cfg_n.variable, cfg_n.obj, cfg_n.condition)
                            add_edge(cur, nid)
                            cur = nid
                        break
            cond = lp_node.child_by_field_name("condition")
            cond_txt = source[cond.start_byte:cond.end_byte].decode(errors="ignore") if cond is not None else ""
            join_id = new_node(NodeType.ASSIGN, lp_node.end_point[0] + 1, "loop_join")
            if lp_node.type == "do_statement":
                tail = build_stmt(body, cur) if body is not None else cur
                if tail is not None and not is_terminal(tail):
                    add_edge(tail, join_id)
                return join_id
            bid = new_node(NodeType.BRANCH, lp_node.start_point[0] + 1,
                           source[lp_node.start_byte:lp_node.end_byte].decode(errors="ignore")[:80],
                           condition=cond_txt)
            add_edge(cur, bid)
            tail = build_stmt(body, bid) if body is not None else bid
            if tail is not None and not is_terminal(tail):
                add_edge(tail, join_id)
            add_edge(bid, join_id)
            return join_id

        def build_if(if_node, in_id: int) -> int:
            """Build CFG for if/else with proper join."""
            cond = if_node.child_by_field_name("condition")
            then_part = if_node.child_by_field_name("consequence")
            else_part = if_node.child_by_field_name("alternative")

            cond_txt = ""
            if cond is not None:
                cond_txt = source[cond.start_byte:cond.end_byte].decode(errors="ignore")
                # `if ((p = alloc(n)) == NULL)`: the assignment in the condition
                # is an allocation that happens before the branch is taken.
                for sub in self._walk(cond):
                    if sub.type == "call_expression":
                        ov, oo = self._out_param_target(sub, source)
                        if ov or oo:
                            aid = new_node(NodeType.ALLOC, sub.start_point[0] + 1,
                                           source[sub.start_byte:sub.end_byte].decode(errors="ignore"), ov, oo)
                            add_edge(in_id, aid)
                            in_id = aid
                        continue
                    if sub.type != "assignment_expression":
                        continue
                    cfg_n = self._process_assign(sub, source, node_id)
                    if cfg_n is not None and cfg_n.node_type == NodeType.ALLOC:
                        aid = new_node(cfg_n.node_type, cfg_n.line, cfg_n.code,
                                       cfg_n.variable, cfg_n.obj, cfg_n.condition)
                        add_edge(in_id, aid)
                        in_id = aid

            bid = new_node(
                NodeType.BRANCH,
                if_node.start_point[0] + 1,
                source[if_node.start_byte:if_node.end_byte].decode(errors="ignore")[:80],
                condition=cond_txt,
            )
            add_edge(in_id, bid)

            join_id = new_node(NodeType.ASSIGN, if_node.end_point[0] + 1, "join")

            then_tail = bid
            if then_part is not None:
                if then_part.type == "compound_statement":
                    then_tail = build_stmt_list(then_part, bid)
                else:
                    then_tail = build_stmt(then_part, bid)
            if not is_terminal(then_tail):
                add_edge(then_tail, join_id)

            if else_part is not None and else_part.type == "else_clause":
                inner = None
                for ch in else_part.children:
                    if ch.is_named:
                        inner = ch
                else_part = inner

            else_tail = bid
            if else_part is not None:
                if else_part.type == "compound_statement":
                    else_tail = build_stmt_list(else_part, bid)
                else:
                    else_tail = build_stmt(else_part, bid)
            if not is_terminal(else_tail):
                add_edge(else_tail, join_id)

            return join_id

        def build_switch(sw_node, in_id: int) -> int:
            """Each case body is a parallel successor of the branch node (fallthrough not modeled)."""
            cond = sw_node.child_by_field_name("condition")
            cond_txt = ""
            if cond is not None:
                cond_txt = source[cond.start_byte:cond.end_byte].decode(errors="ignore")
                # An out-parameter allocation in the switch expression happens
                # before the branch, as in an if condition.
                for sub in self._walk(cond):
                    if sub.type == "call_expression":
                        ov, oo = self._out_param_target(sub, source)
                        if ov or oo:
                            aid = new_node(NodeType.ALLOC, sub.start_point[0] + 1,
                                           source[sub.start_byte:sub.end_byte].decode(errors="ignore"), ov, oo)
                            add_edge(in_id, aid)
                            in_id = aid

            bid = new_node(
                NodeType.BRANCH,
                sw_node.start_point[0] + 1,
                source[sw_node.start_byte:sw_node.end_byte].decode(errors="ignore")[:80],
                condition=cond_txt,
            )
            add_edge(in_id, bid)

            join_id = new_node(NodeType.ASSIGN, sw_node.end_point[0] + 1, "switch_join")

            body = sw_node.child_by_field_name("body")
            has_default = False
            if body is not None and body.type == "compound_statement":
                for ch in body.children:
                    if ch.type == "case_statement":
                        if any(c.type == "default" for c in ch.children):
                            has_default = True
                        case_tail = build_case(ch, bid)
                        if case_tail is not None and not is_terminal(case_tail):
                            add_edge(case_tail, join_id)

            # Without a default label the switch may select no case at all.
            if not has_default:
                add_edge(bid, join_id)
            return join_id

        def build_case(case_node, in_id: int) -> int:
            cur = in_id
            for ch in case_node.children:
                if not ch.is_named:
                    continue
                if ch.type in ("number_literal", "char_literal", "string_literal", "concatenated_string"):
                    continue
                cur = build_stmt(ch, cur)
                if cur is not None and is_terminal(cur):
                    break
            return cur

        tail = entry_id
        if body.type == "compound_statement":
            tail = build_stmt_list(body, entry_id)
        else:
            tail = build_stmt(body, entry_id)

        if nodes[tail].node_type not in (NodeType.RETURN, NodeType.GOTO, NodeType.EXIT):
            add_edge(tail, exit_id)

            
        for gid, label in pending_gotos:
            tgt = label_nodes.get(label)
            logger.info(f"goto {label} to {tgt}")

            for s in list(nodes[gid].successors):
                nodes[gid].successors.remove(s)
                if gid in nodes[s].predecessors:
                    nodes[s].predecessors.remove(gid)

            if tgt is not None:
                add_edge(gid, tgt)
            else:
                add_edge(gid, exit_id)


        return nodes

    def _out_param_target(self, call_node, source: bytes) -> tuple[str, str]:
        """For a call to an out-parameter allocator, (variable, obj) of the
        object it allocates: `f(&x)` allocates x; `f(out)` allocates *out."""
        while call_node is not None and call_node.type in ("cast_expression", "parenthesized_expression"):
            inner = call_node.child_by_field_name("value")
            if inner is None:
                named = [c for c in call_node.children if c.is_named]
                inner = named[-1] if named else None
            call_node = inner
        if call_node is None or call_node.type != "call_expression":
            return "", ""
        fn = call_node.child_by_field_name("function")
        name = source[fn.start_byte:fn.end_byte].decode(errors="ignore").strip() if fn else ""
        if name not in self.alloc_out_params:
            return "", ""
        args = call_node.child_by_field_name("arguments")
        named = [c for c in args.children if c.is_named] if args else []
        idx = self.alloc_out_params[name]
        if not (0 <= idx < len(named)):
            return "", ""
        txt = source[named[idx].start_byte:named[idx].end_byte].decode(errors="ignore").strip()
        m = re.fullmatch(r"&\s*([A-Za-z_]\w*)", txt)
        if m:
            return m.group(1), m.group(1)
        if re.fullmatch(r"[A-Za-z_]\w*", txt):
            return txt, "*" + txt
        return "", txt

    def _rhs_allocator_call(self, rhs_node, source: bytes) -> str:
        """Allocator name if rhs is a call to a known alloc func, unwrapping
        casts, parentheses, and chained assignments (a = b = malloc(n))."""
        while rhs_node is not None and rhs_node.type in (
            "cast_expression", "parenthesized_expression", "assignment_expression"
        ):
            if rhs_node.type == "assignment_expression":
                inner = rhs_node.child_by_field_name("right")
            else:
                inner = rhs_node.child_by_field_name("value")
                if inner is None:
                    named = [c for c in rhs_node.children if c.is_named]
                    inner = named[-1] if named else None
            rhs_node = inner
        if rhs_node is not None and rhs_node.type == "new_expression":
            return "new"
        if rhs_node is None or rhs_node.type != "call_expression":
            return ""
        fn = rhs_node.child_by_field_name("function")
        if not fn:
            return ""
        fname = source[fn.start_byte:fn.end_byte].decode(errors="ignore").strip()
        return fname if fname in self.alloc_funcs else ""

    def _walk(self, node):
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _process_node(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        if node.type == "call_expression":
            return self._process_call(node, source, node_id)
        elif node.type == "delete_expression":
            return self._process_delete(node, source, node_id)
        elif node.type == "if_statement":
            return self._process_if(node, source, node_id)
        elif node.type == "return_statement":
            return self._process_return(node, source, node_id)
        elif node.type in ("pointer_expression", "subscript_expression", "field_expression"):
            return self._process_deref(node, source, node_id)
        elif node.type == "goto_statement":
            return self._process_goto(node, source, node_id)

        elif node.type == "assignment_expression":
            return self._process_assign(node, source, node_id)
        elif node.type == "init_declarator":
            return self._process_init_declarator(node, source, node_id)

        return None

    def _process_goto(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        line = node.start_point[0] + 1
        code = source[node.start_byte:node.end_byte].decode(errors="ignore").strip()

        label = ""
        for ch in node.children:
            if ch.type == "identifier":
                label = source[ch.start_byte:ch.end_byte].decode(errors="ignore").strip()
                break

        if not label:
            return None

        return CFGNode(
            id=node_id,
            node_type=NodeType.GOTO,
            line=line,
            code=code,
            variable=label,
            obj=label,
        )

        
    def _node_text(self, node, source: bytes) -> str:
        return source[node.start_byte:node.end_byte].decode(errors="ignore")

    def _process_assign(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """Emit ASSIGN for `lhs = rhs` so the leak checker can see escape
        sinks."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return None

        lhs_txt = self._node_text(left, source).strip()
        rhs_txt = self._node_text(right, source).strip()
        
        out_var, out_obj = self._out_param_target(right, source)
        if out_var or out_obj:
            return CFGNode(id=node_id, node_type=NodeType.ALLOC, line=node.start_point[0] + 1,
                           code=f"{lhs_txt} = {rhs_txt}", variable=out_var, obj=out_obj)
        alloc_name = self._rhs_allocator_call(right, source)
        if alloc_name:
            var = lhs_txt if re.fullmatch(r"[A-Za-z_]\w*", lhs_txt) else ""
            return CFGNode(
                id=node_id,
                node_type=NodeType.ALLOC,
                line=node.start_point[0] + 1,
                code=f"{lhs_txt} = {rhs_txt}",
                variable=var,
                obj=lhs_txt,
            )


        code = f"{lhs_txt} = {rhs_txt}"

        return CFGNode(
            id=node_id,
            node_type=NodeType.ASSIGN,
            line=node.start_point[0] + 1,
            code=code,
            variable=lhs_txt,
            obj=rhs_txt,
        )

    def _process_init_declarator(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """Emit ASSIGN for `T *p = alloc(...)` so the leak checker can see the
        escape."""
        value = node.child_by_field_name("value")
        if value is None:
            return None

        declarator = node.child_by_field_name("declarator")
        lhs_name = ""

        if declarator is not None:
            stack = [declarator]
            while stack:
                cur = stack.pop()
                if cur.type == "identifier":
                    lhs_name = self._node_text(cur, source).strip()
                    break
                stack.extend(list(cur.children))

        if not lhs_name:
            for ch in node.children:
                if ch.type == "identifier":
                    lhs_name = self._node_text(ch, source).strip()
                    break

        rhs_txt = self._node_text(value, source).strip()
        if not rhs_txt:
            return None

        code = f"{lhs_name} = {rhs_txt}" if lhs_name else rhs_txt

        out_var, out_obj = self._out_param_target(value, source)
        if out_var or out_obj:
            return CFGNode(id=node_id, node_type=NodeType.ALLOC, line=node.start_point[0] + 1,
                           code=code, variable=out_var, obj=out_obj)
        alloc_name = self._rhs_allocator_call(value, source)
        if alloc_name:
            var = lhs_name if re.fullmatch(r"[A-Za-z_]\w*", lhs_name or "") else ""
            return CFGNode(
                id=node_id,
                node_type=NodeType.ALLOC,
                line=node.start_point[0] + 1,
                code=code,
                variable=var,
                obj=lhs_name,
            )

        return CFGNode(
            id=node_id,
            node_type=NodeType.ASSIGN,
            line=node.start_point[0] + 1,
            code=code,
            variable=lhs_name,
            obj=rhs_txt,
        )


    def _process_deref(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """Create a DEREF node when we see obvious pointer/array/member
        dereference."""
        line = node.start_point[0] + 1
        code = source[node.start_byte:node.end_byte].decode(errors="ignore")

        var = ""

        if node.type == "pointer_expression":
            arg = node.child_by_field_name("argument")
            if arg and arg.type == "identifier":
                var = source[arg.start_byte:arg.end_byte].decode(errors="ignore")

        elif node.type == "subscript_expression":
            arg = node.child_by_field_name("argument")
            if arg and arg.type == "identifier":
                var = source[arg.start_byte:arg.end_byte].decode(errors="ignore")

        elif node.type == "field_expression":
            arg = node.child_by_field_name("argument")
            if arg:
                if arg.type == "identifier":
                    var = source[arg.start_byte:arg.end_byte].decode(errors="ignore")
                elif arg.type == "parenthesized_expression":
                    for ch in arg.children:
                        if ch.type == "identifier":
                            var = source[ch.start_byte:ch.end_byte].decode(errors="ignore")
                            break

        
        if not var:
            return None

        return CFGNode(
            id=node_id,
            node_type=NodeType.DEREF,
            line=line,
            code=code,
            variable=var,
            obj=code.strip(),
        )


    
    def _extract_base_identifier(self, node, source: bytes) -> str:
        """Best-effort: extract the base identifier used as a pointer."""
        cur = node

        while cur is not None and cur.type in ("parenthesized_expression", "cast_expression"):
            inner = cur.child_by_field_name("value") or cur.child_by_field_name("expression")
            if inner is None and cur.children:
                inner = cur.children[-1]
            cur = inner

        if cur is None:
            return ""

        if cur.type == "pointer_expression":
            arg = cur.child_by_field_name("argument")
            if arg is None and cur.children:
                arg = cur.children[-1]
            return self._extract_base_identifier(arg, source)

        if cur.type == "field_expression":
            obj = cur.child_by_field_name("argument")
            if obj is None and cur.children:
                obj = cur.children[0]
            return self._extract_base_identifier(obj, source)

        if cur.type == "subscript_expression":
            arr = cur.child_by_field_name("argument") or cur.child_by_field_name("array")
            if arr is None and cur.children:
                arr = cur.children[0]
            return self._extract_base_identifier(arr, source)

        if cur.type == "identifier":
            return source[cur.start_byte:cur.end_byte].decode(errors="ignore")

        if cur.type == "call_expression":
            fn = cur.child_by_field_name("function")
            if fn is not None:
                return self._extract_base_identifier(fn, source)

        return ""


    def _process_delete(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        operand = node.children[-1]
        obj = source[operand.start_byte:operand.end_byte].decode(errors="ignore").strip()
        return CFGNode(
            id=node_id,
            node_type=NodeType.FREE,
            line=node.start_point[0] + 1,
            code=source[node.start_byte:node.end_byte].decode(errors="ignore"),
            variable=obj if re.fullmatch(r"[A-Za-z_]\w*", obj) else "",
            obj=obj,
        )

    def _process_call(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        func_node = node.child_by_field_name("function")
        if not func_node:
            return None

        func_name = source[func_node.start_byte:func_node.end_byte].decode(errors="ignore")
        line = node.start_point[0] + 1
        code = source[node.start_byte:node.end_byte].decode(errors="ignore")

        if func_name in self.alloc_out_params:
            out_var, out_obj = self._out_param_target(node, source)
            return CFGNode(
                id=node_id,
                node_type=NodeType.ALLOC,
                line=line,
                code=code,
                variable=out_var,
                obj=out_obj,
            )


        if func_name in self.alloc_funcs:
            parent = node.parent
            var = ""
            if parent and parent.type in ("assignment_expression", "init_declarator"):
                for child in parent.children:
                    if child.type == "identifier":
                        var = source[child.start_byte:child.end_byte].decode(errors="ignore")
                        break
            alloc_obj = var
            return CFGNode(
                id=node_id,
                node_type=NodeType.ALLOC,
                line=line,
                code=code,
                variable=var,
                obj=alloc_obj,
            )


        if func_name in self.free_funcs:
            args = node.child_by_field_name("arguments")
            obj = ""
            var = ""
            # The summary records which argument is freed (X_free(ctx, p) frees p).
            fidx = getattr(self, "free_arg_idx", {}).get(func_name)
            if args and fidx is not None:
                named = [c for c in args.children if c.is_named]
                if 0 <= fidx < len(named):
                    obj = source[named[fidx].start_byte:named[fidx].end_byte].decode(errors="ignore").strip()
                    args = None
            if args:
                for child in args.children:
                    if child.type in (
                        "identifier",
                        "subscript_expression",
                        "field_expression",
                        "pointer_expression",
                        "parenthesized_expression",
                        "cast_expression",
                        "call_expression",
                    ):
                        obj = source[child.start_byte:child.end_byte].decode(errors="ignore").strip()
                        break
            if obj:
                # free((void *)p), free((p)), free(&*p): the object is p
                prev = None
                while prev != obj:
                    prev = obj
                    stripped = re.sub(r"^\(\s*[A-Za-z_][\w\s]*\**\s*\)\s*", "", obj).strip()
                    if stripped:
                        obj = stripped
                    if obj.startswith("(") and obj.endswith(")"):
                        obj = obj[1:-1].strip()
                # wrapper(&p) on a pointer-to-pointer deallocator frees p
                if obj.startswith("&") and re.fullmatch(r"&\s*[A-Za-z_]\w*", obj):
                    obj = obj[1:].strip()
            if obj:
                if re.fullmatch(r"[A-Za-z_]\w*", obj):
                    var = obj

            return CFGNode(
                id=node_id,
                node_type=NodeType.FREE,
                line=line,
                code=code,
                variable=var,
                obj=obj,
            )


        args = node.child_by_field_name("arguments")
        if args:
            for child in args.children:
                if child.type in (
                    "identifier",
                    "subscript_expression",
                    "field_expression",
                    "pointer_expression",
                    "parenthesized_expression",
                    "cast_expression",
                ):
                    obj = source[child.start_byte:child.end_byte].decode(errors="ignore").strip()
                    if not obj:
                        continue
                    var = obj if re.fullmatch(r"[A-Za-z_]\w*", obj) else ""
                    return CFGNode(
                        id=node_id,
                        node_type=NodeType.DEREF,
                        line=line,
                        code=code,
                        variable=var,
                        obj=obj,
                    )


        return None


    def _process_if(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        cond = node.child_by_field_name("condition")
        if not cond:
            return None

        condition = source[cond.start_byte:cond.end_byte].decode()
        return CFGNode(
            id=node_id, node_type=NodeType.BRANCH,
            line=node.start_point[0] + 1,
            code=source[node.start_byte:node.end_byte].decode()[:50],
            condition=condition
        )

    def _process_return(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        return CFGNode(
            id=node_id, node_type=NodeType.RETURN,
            line=node.start_point[0] + 1,
            code=source[node.start_byte:node.end_byte].decode()
        )


class OwnershipAnalysis:
    """Identifier-level ownership facts shared by Phase 3 and Phase 5: which
    names a function declares, which branch conditions coincide, and whether
    an allocator keeps (stores into a global or a container) the object it
    returns on every path, in which case the caller does not own it.
    Users set _functions_ref, _alloc_set, _free_set and _retain_memo."""

    ownership_sinks: frozenset = OWNERSHIP_SINKS

    _COND_KEYWORDS = {"NULL", "nullptr", "sizeof", "true", "false", "TRUE", "FALSE"}

    def _cond_key(self, cond: str):
        """Normalise a branch condition to (key, negated, identifiers), or
        None when it may have side effects or differ between evaluations."""
        def strip_parens(c):
            c = c.strip()
            while c.startswith("(") and c.endswith(")"):
                depth = 0
                for i, ch in enumerate(c):
                    depth += ch == "("
                    depth -= ch == ")"
                    if depth == 0 and i < len(c) - 1:
                        return c
                c = c[1:-1].strip()
            return c
        c = strip_parens(cond or "")
        neg = False
        while c.startswith("!") and not c.startswith("!="):
            c = strip_parens(c[1:])
            neg = not neg
        m = (re.fullmatch(r"(.+?)\s*(==|!=)\s*(NULL|0|nullptr)", c)
             or re.fullmatch(r"(NULL|0|nullptr)\s*(==|!=)\s*(.+)", c))
        if m:
            x = m.group(1) if m.group(3) in ("NULL", "0", "nullptr") and m.group(1) not in ("NULL", "0", "nullptr") else m.group(3)
            c = strip_parens(x)
            if m.group(2) == "==":
                neg = not neg
        if not c or re.search(r"[A-Za-z_]\w*\s*\(", c):     # a call
            return None
        if re.search(r"(?<![=!<>])=(?!=)|\+\+|--", c):       # an assignment
            return None
        idents = set(re.findall(r"[A-Za-z_]\w*", c)) - self._COND_KEYWORDS
        if not idents:
            return None
        return re.sub(r"\s+", "", c), neg, idents

    @staticmethod
    def _call_args(code: str, open_idx: int) -> list[str]:
        """Top-level arguments of the call whose '(' is at open_idx."""
        args, depth, cur = [], 0, []
        for ch in code[open_idx + 1:]:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            if ch == "," and depth == 0:
                args.append("".join(cur).strip()); cur = []
                continue
            cur.append(ch)
        last = "".join(cur).strip()
        if last or args:
            args.append(last)
        return args

    @staticmethod
    def _plain(expr: str) -> str:
        """The identifier an argument/rhs denotes, ignoring casts and parens."""
        e = (expr or "").strip()
        prev = None
        while prev != e:
            prev = e
            e = re.sub(r"^\(\s*[A-Za-z_][\w\s]*\**\s*\)\s*(?=\S)", "", e).strip()
            if e.startswith("(") and e.endswith(")"):
                e = e[1:-1].strip()
        return e if re.fullmatch(r"[A-Za-z_]\w*", e) else ""

    def _calls(self, code: str):
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", code or ""):
            yield m.group(1), self._call_args(code, m.end() - 1)

    def _stores_var(self, node, var: str, local: set, params: set) -> bool:
        """Does this CFG node hand `var` to memory that outlives the call?"""
        code = (node.code or "").strip()
        if node.node_type == NodeType.BRANCH:
            return False
        m = re.match(r"\s*([^=]+?)\s*(?<![=!<>+\-*/%&|^])=(?!=)\s*(.+?)\s*;?\s*$", code, re.S)
        rhs_last = re.split(r"(?<![=!<>+\-*/%&|^])=(?!=)", m.group(2))[-1] if m else ""
        if m and self._plain(rhs_last.lstrip("&")) == var:
            lhs = m.group(1).strip()
            mb = re.match(r"[\s*(]*([A-Za-z_]\w*)", lhs)
            if mb and mb.group(1) != var:
                base = mb.group(1)
                if base not in local and base not in params and lhs != base:
                    return True                        # into a global table: g[i], g.f, g->f
        for callee, args in self._calls(code):
            for k, a in enumerate(args):
                if self._plain(a) != var:
                    continue
                if callee in self.ownership_sinks:
                    return True
        return False

    def _cfg_of(self, fn):
        b = CFGBuilder(getattr(self, "_alloc_set", set()), getattr(self, "_free_set", set()),
                       file_path=getattr(fn, "file_path", "") or "")
        return b.build(fn.code, line_offset=max((getattr(fn, "start_line", 1) or 1) - 1, 0))

    @staticmethod
    def _reaches_avoiding(cfg, starts, targets, blockers, blocked_edges=frozenset()) -> bool:
        seen, stack = set(), [s for s in starts if s not in blockers]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in targets:
                return True
            for s in cfg[n].successors:
                if s not in blockers and s not in seen and (n, s) not in blocked_edges:
                    stack.append(s)
        return False

    def _oom_edges(self, cfg, only: Optional[set] = None) -> set[tuple[int, int]]:
        """Entry nodes of allocation-failure branches: `if (!x)` / `if (x == NULL)`
        right after x was allocated. Ownership summaries are about the paths on
        which the callee succeeds."""
        allocated = {(n.variable or n.obj or "").strip() for n in cfg.values() if n.node_type == NodeType.ALLOC}
        allocated.discard("")
        out = set()
        for n in cfg.values():
            if n.node_type != NodeType.BRANCH or len(n.successors) != 2:
                continue
            cond = n.condition or ""
            c = cond.strip()
            while c.startswith("(") and c.endswith(")") and c.count("(") == c.count(")"):
                inner = c[1:-1].strip()
                depth = 0; ok = True
                for ch in inner:
                    depth += (ch == "(") - (ch == ")")
                    if depth < 0:
                        ok = False; break
                if not ok:
                    break
                c = inner
            ma = re.fullmatch(r"\(\s*([A-Za-z_]\w*)\s*=[^=].*\)\s*(==|!=)\s*(NULL|0|nullptr)", c, re.S)
            mb = re.fullmatch(r"(!?)\s*\(\s*([A-Za-z_]\w*)\s*=[^=].*\)", c, re.S)
            if ma:
                k = (ma.group(1), ma.group(2) == "==", {ma.group(1)})
            elif mb:
                k = (mb.group(2), mb.group(1) == "!", {mb.group(2)})
            else:
                k = self._cond_key(cond)
            if k is None or k[0] not in allocated or (only is not None and k[0] not in only):
                continue
            # Only a test that directly follows the allocation of that pointer
            # (straight-line code in between) checks for allocation failure;
            # `if (p == NULL) p = alloc();` is lazy initialisation, not one.
            cur, steps, follows = n, 0, False
            while len(cur.predecessors) == 1 and steps < 12:
                cur = cfg[cur.predecessors[0]]
                steps += 1
                if cur.node_type == NodeType.ALLOC and (cur.variable or cur.obj or "").strip() == k[0]:
                    follows = True
                    break
                if cur.node_type in (NodeType.BRANCH, NodeType.ENTRY, NodeType.LABEL):
                    break
            if not follows:
                continue
            # neg: condition is !x / x == NULL, so the true edge is the failure.
            # The edge, not its target: with no else branch the failure edge
            # leads straight to the join, which the success path also reaches.
            out.add((n.id, n.successors[0] if k[1] else n.successors[1]))
        return out

    def _blockers_for(self, cfg, var, local, params) -> set[int]:
        """Nodes that store var out, directly or through a local carrier
        (o->f = var; then o is stored out)."""
        direct = {n.id for n in cfg.values() if self._stores_var(n, var, local, params)}
        carriers = set()
        for n in cfg.values():
            m = re.match(r"\s*([A-Za-z_]\w*)\s*(?:->|\.)[^=]*?(?<![=!<>])=(?!=)\s*(.+?)\s*;?\s*$", n.code or "", re.S)
            if m and m.group(1) in local and m.group(1) != var and self._plain(m.group(2)) == var:
                carriers.add(m.group(1))
        for o in carriers:
            direct |= {n.id for n in cfg.values() if self._stores_var(n, o, local, params)}
        return direct

    def _allocator_retains(self, name: str, depth: int) -> bool:
        """Allocator `name` stores the object it allocates and returns into a
        global or an ownership sink on every path from the allocation to the
        return, so the caller does not acquire ownership."""
        memo = self._retain_memo
        if name in memo:
            return memo[name]
        memo[name] = False
        fn = self._functions_ref.get(name)
        if fn is None or not fn.code or depth > 10:
            return False
        try:
            cfg = self._cfg_of(fn)
        except Exception:
            return False
        local = self._declared_names(fn)
        params = set(fn.arg_names or [])
        alloc_paths = 0
        for r in cfg.values():
            if r.node_type != NodeType.RETURN:
                continue
            mret = re.match(r"\s*return\s*(.*?)\s*;?\s*$", r.code or "", re.S)
            expr = mret.group(1) if mret else ""
            rv = self._plain(expr)
            if not rv:
                mc = re.match(r"\(?\s*([A-Za-z_]\w*)\s*\(", expr)
                if mc and mc.group(1) in self._functions_ref and mc.group(1) in self._alloc_set:
                    alloc_paths += 1
                    if not self._allocator_retains(mc.group(1), depth + 1):
                        memo[name] = False
                        return False
                continue
            allocs = [a for a in cfg.values() if a.node_type == NodeType.ALLOC
                      and (a.variable or a.obj or "").strip() == rv]
            if not allocs:
                continue                      # not an allocation path (lookup)
            stores = self._blockers_for(cfg, rv, local, params)
            failed = self._oom_edges(cfg)
            for a in allocs:
                alloc_paths += 1
                mcal = re.search(r"\b([A-Za-z_]\w*)\s*\(", a.code or "")
                callee = mcal.group(1) if mcal else ""
                if callee in self._functions_ref and callee != name and self._allocator_retains(callee, depth + 1):
                    continue                  # retained by the callee itself
                # Retention needs positive evidence: the allocation must reach
                # this return in the CFG at all, and every such path must pass
                # a store. A path the CFG cannot see (e.g. code under #ifdef
                # that the parser could not structure) is no evidence.
                if not self._reaches_avoiding(cfg, [a.id], {r.id}, set(), failed):
                    alloc_paths -= 1
                    continue
                if self._reaches_avoiding(cfg, [a.id], {r.id}, stores - {a.id}, failed):
                    memo[name] = False
                    return False
        memo[name] = alloc_paths > 0
        return memo[name]

    def _declared_names(self, func) -> set[str]:
        """Parameters and locals declared in the function; any other plain
        identifier assigned to is a global (or file-scope static)."""
        lang = CPP_LANGUAGE if _is_cpp(getattr(func, "file_path", "") or "") else C_LANGUAGE
        src = func.code.encode(errors="ignore")
        root = Parser(lang).parse(src).root_node
        names = set(getattr(func, "arg_names", None) or [])
        stack = [root]
        while stack:
            n = stack.pop()
            stack.extend(n.children)
            if n.type not in ("declaration", "parameter_declaration"):
                continue
            if any(c.type == "storage_class_specifier"
                   and src[c.start_byte:c.end_byte] in (b"static", b"extern") for c in n.children):
                continue          # a static/extern local outlives the call, like a global
            for d in n.children_by_field_name("declarator"):
                while d is not None and d.type != "identifier":
                    d = d.child_by_field_name("declarator")
                if d is not None:
                    names.add(src[d.start_byte:d.end_byte].decode(errors="ignore"))
        return names

    def _return_is_exact_var(self, code: str, var: str) -> bool:
        """True iff the statement is exactly 'return var;' (optionally parenthesized)."""
        if not code or not var:
            return False
        m = _RETURN_ONLY_VAR_RE.match(code.strip())
        return bool(m and m.group("var") == var)


    def _assignment_lines(self, func) -> dict[str, list[int]]:
        """Line numbers at which each identifier may be written."""
        out: dict[str, list[int]] = defaultdict(list)
        base = (getattr(func, "start_line", 1) or 1)
        pat = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)?(?:[-+*/%&|^]|<<|>>)?=(?!=)"
                         r"|&\s*([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*(?:\+\+|--)|(?:\+\+|--)\s*([A-Za-z_]\w*)")
        for i, line in enumerate((func.code or "").split("\n")):
            for m in pat.finditer(line):
                name = next(g for g in m.groups() if g)
                out[name].append(base + i)
        return out

    def _add_branch_constraints(self, cfg, edge, solver):
        """Branch conditions as uninterpreted Booleans: the true edge needs
        b, the false edge not b. Syntactically identical conditions over
        operands not written in between share b; c and !c are each other's
        negation. This rules out paths through mutually exclusive branches."""
        fn = getattr(self, "_cur_func", None)
        if fn is None:
            return
        writes = self._assignment_lines(fn)
        local = self._declared_names(fn)
        code = fn.code or ""
        groups: dict[str, list] = defaultdict(list)
        for n in cfg.values():
            if n.node_type != NodeType.BRANCH or len(n.successors) != 2:
                continue
            if (n.code or "").lstrip().startswith("switch"):
                continue
            k = self._cond_key(n.condition)
            if k is not None:
                groups[k[0]].append((n, k[1], k[2]))
        for i, (key, members) in enumerate(groups.items()):
            if len(members) < 2:
                continue
            lo = min(n.line for n, _, _ in members)
            hi = max(n.line for n, _, _ in members)
            idents = set().union(*(ids for _, _, ids in members))
            if any(lo <= l <= hi for v in idents for l in writes.get(v, ())):
                continue
            # Only plain locals/parameters whose address is never taken: a
            # call or a pointer write cannot change them unseen. Fields,
            # dereferences and globals may be changed by any call.
            if re.search(r"->|\.|\*|\[", key):
                continue
            if any(v not in local or re.search(rf"&\s*{re.escape(v)}\b", code) for v in idents):
                continue
            b = Bool(f"cond_{i}")
            for n, neg, _ in members:
                lit = Not(b) if neg else b
                t, f = n.successors[0], n.successors[1]
                if (n.id, t) in edge:
                    solver.add(Implies(edge[(n.id, t)], lit))
                if (n.id, f) in edge:
                    solver.add(Implies(edge[(n.id, f)], Not(lit)))


class HintValidator(OwnershipAnalysis):
    """Validate LLM-generated summaries against the function CFG with Z3.

    An allocator is kept only if some path carries an allocation to the return;
    a deallocator only if some path frees the target argument (or an alias).
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            logger.warning("Z3 not available, validation will be skipped")
        self._transitive_analyzer: Optional[TransitiveAnalyzer] = None

    def _init_transitive_analyzer(self, functions: dict[str, FunctionInfo]):
        """Initialize transitive analyzer lazily."""
        if self._transitive_analyzer is None:
            cg_builder = CallGraphBuilder()
            call_graph = cg_builder.build(functions)
            self._transitive_analyzer = TransitiveAnalyzer(call_graph, functions)

    def validate_hints(
        self,
        hints: HintSet,
        functions: dict[str, FunctionInfo],
    ) -> tuple[HintSet, list[str]]:
        """Validate all hints, returning validated hints and conflict messages."""
        if not Z3_AVAILABLE:
            return hints, []

        self._init_transitive_analyzer(functions)

        # An allocator summary means the caller owns the returned object; one
        # that stores the object into a global or a container on every path
        # keeps it, so the caller never owns it.
        own = OwnershipAnalysis()
        own._functions_ref = functions
        own._alloc_set = (set(TransitiveAnalyzer.BASE_ALLOCATORS)
                          | {n for n, hs in hints.hints.items()
                             for h in hs if h.hint_type == HintType.ALLOCATOR})
        own._free_set = (set(TransitiveAnalyzer.BASE_DEALLOCATORS)
                         | {n for n, hs in hints.hints.items()
                            for h in hs if h.hint_type == HintType.DEALLOCATOR})
        own._retain_memo = {}
        self._ownership = own

        validated = HintSet()
        conflicts = []

        for func_name, func_hints in hints.hints.items():
            if func_name not in functions:
                continue

            func = functions[func_name]

            for hint in func_hints:
                try:
                    result = self._validate_hint(hint, func, functions)
                except Exception as e:
                    logger.warning(f"Phase 3: validating {func_name} failed ({e}); kept")
                    result = ValidationResult(is_valid=True, reason=f"validation error; kept: {e}")
                if result.is_valid:
                    validated.add(hint)
                else:
                    conflicts.append(
                        f"REMOVED {func_name}.{hint.hint_type.name}: {result.reason}"
                    )

        return validated, conflicts

    def _validate_hint(
        self,
        hint: Hint,
        func: FunctionInfo,
        functions: dict[str, FunctionInfo]
    ) -> ValidationResult:
        """Validate a single hint against function code."""
        if hint.hint_type == HintType.ALLOCATOR:
            return self._validate_allocator(hint, func, functions)
        elif hint.hint_type == HintType.DEALLOCATOR:
            return self._validate_deallocator(hint, func, functions)
        else:
            return ValidationResult(is_valid=True, reason="accepted")

    def _detect_allocator(self, hint: Hint, func: FunctionInfo,
                          alloc_funcs: Optional[set[str]] = None) -> ValidationResult:
        """Heuristic detection via direct allocation primitives or transitive chains."""
        code = func.code
        func_name = hint.function_name
        alloc_funcs = set(alloc_funcs or ()) | set(STANDARD_ALLOCATORS)

        if hint.arg_index >= 0:
            if hint.arg_index < len(func.arg_types) and '**' not in func.arg_types[hint.arg_index]:
                return ValidationResult(is_valid=False,
                                        reason=f"Argument {hint.arg_index} is not a pointer-to-pointer type")
            if any(f"{a}(" in code for a in alloc_funcs):
                return ValidationResult(is_valid=True, reason="out-parameter allocation found")
            if self._transitive_analyzer:
                is_alloc, chain = self._transitive_analyzer.is_transitive_allocator(func_name)
                if is_alloc:
                    return ValidationResult(is_valid=True, reason=f"out-parameter, chain: {' -> '.join(chain)}")
            return ValidationResult(is_valid=False, reason="No allocation found for out-parameter")

        if not func.return_type or '*' not in func.return_type:
            return ValidationResult(is_valid=False, reason="Does not return pointer type")
        if any(f"{a}(" in code for a in alloc_funcs):
            return ValidationResult(is_valid=True, reason="direct allocation found")
        if self._transitive_analyzer:
            is_alloc, chain = self._transitive_analyzer.is_transitive_allocator(func_name)
            if is_alloc:
                return ValidationResult(is_valid=True, reason=f"chain: {' -> '.join(chain)}")
        if "return" not in code.lower():
            return ValidationResult(is_valid=False, reason="No allocation call and no return")
        return ValidationResult(is_valid=True, reason="validated (weak)")

    def _validate_allocator(
        self,
        hint: Hint,
        func: FunctionInfo,
        functions: dict[str, FunctionInfo]
    ) -> ValidationResult:
        """Detection gated by a Z3 CFG feasibility filter: discard only if Z3
        proves the allocation never reaches the return; keep when there is
        no CFG evidence."""
        alloc_funcs, free_funcs = self._wrapper_funcsets(func, functions)
        det = self._detect_allocator(hint, func, alloc_funcs)
        if not det.is_valid:
            return det

        out_param_idx = hint.arg_index if hint.arg_index >= 0 else None
        keep, reason = self._z3_allocator_feasible(func, alloc_funcs, free_funcs, out_param_idx)
        own = getattr(self, "_ownership", None)
        if keep and out_param_idx is None and own is not None:
            try:
                retained = own._allocator_retains(hint.function_name, 0)
            except Exception as e:
                logger.debug(f"retention check failed for {hint.function_name}: {e}")
                retained = False
            if retained:
                return ValidationResult(
                    is_valid=False,
                    reason="object stored into a global or a container on every path "
                           "to the return; the caller does not own it")
        if keep:
            return ValidationResult(is_valid=True, reason=f"{det.reason}; Z3 ok: {reason}")
        return ValidationResult(is_valid=False, reason=f"Z3 discarded (infeasible): {reason}")

    def _validate_deallocator(
        self,
        hint: Hint,
        func: FunctionInfo,
        functions: dict[str, FunctionInfo]
    ) -> ValidationResult:
        """Validate DEALLOCATOR hint with enhanced analysis."""
        func_name = hint.function_name

        if not self._transitive_analyzer:
            return self._validate_deallocator_simple(hint, func)

        is_dealloc, freed_arg, chain = self._transitive_analyzer.is_transitive_deallocator(func_name)
        if is_dealloc:
            _, free_funcs = self._wrapper_funcsets(func, functions)
            if hint.arg_index >= 0:
                if not self._frees_argument_itself(func, hint.arg_index, free_funcs):
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Does not free arg{hint.arg_index} itself (only internal members / containers)"
                    )

            if hint.arg_index >= 0 and freed_arg >= 0 and hint.arg_index != freed_arg:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Hint says arg {hint.arg_index} freed, but analysis shows arg {freed_arg}"
                )

            arg_for_z3 = hint.arg_index if hint.arg_index >= 0 else (freed_arg if freed_arg >= 0 else 0)
            keep, reason = self._z3_deallocator_feasible(func, free_funcs, arg_for_z3)
            chain_str = " -> ".join(chain) if chain else func_name
            if keep:
                return ValidationResult(is_valid=True, reason=f"Z3: {reason}; chain: {chain_str}")
            return ValidationResult(is_valid=False, reason=f"Z3 discarded: {reason}")

        if self._has_indirect_free_call(func):
            return ValidationResult(
                is_valid=True,
                reason="indirect call detected (conservative)"
            )

        return ValidationResult(
            is_valid=False,
            reason="No deallocation call found (direct or transitive)"
        )

    def _validate_deallocator_simple(
        self,
        hint: Hint,
        func: FunctionInfo
    ) -> ValidationResult:
        """Simple deallocator validation without transitive analysis."""
        code = func.code
        has_direct_free = any(f"{f}(" in code for f in STANDARD_DEALLOCATORS)

        if has_direct_free:
            return ValidationResult(is_valid=True, reason="direct free found")

        return ValidationResult(
            is_valid=False,
            reason="No deallocation call found"
        )

    def _has_indirect_free_call(self, func: FunctionInfo) -> bool:
        """Check if function has indirect (function pointer) calls that might
        free."""
        code = func.code

        for param in func.arg_names:
            if re.search(rf'\b{re.escape(param)}\s*\(', code):
                return True

        return False

    def _frees_argument_itself(self, func: FunctionInfo, arg_index: int,
                               free_funcs: Optional[set[str]] = None) -> bool:
        """True iff the code deallocates parameter `arg_index` itself, or a
        local alias of it (v = p), rather than only a member or container."""
        return _frees_one_of(func.code, set(free_funcs or ()), _param_alias_names(func, arg_index))

    # Z3 CFG path-feasibility checks. Branch conditions are uninterpreted, so a
    # "feasible" path is a single structural Entry->...->target path.

    def _wrapper_funcsets(
        self,
        func: FunctionInfo,
        functions: dict[str, FunctionInfo],
    ) -> tuple[set[str], set[str]]:
        """Allocator/deallocator name sets for CFG labeling: base primitives
        plus any direct callee that the transitive analyzer identifies as a
        wrapper."""
        alloc_funcs = set(TransitiveAnalyzer.BASE_ALLOCATORS)
        free_funcs = set(TransitiveAnalyzer.BASE_DEALLOCATORS)
        if self._transitive_analyzer:
            for callee in (getattr(func, "callees", None) or set()):
                if callee == func.name:
                    continue
                try:
                    is_alloc, _ = self._transitive_analyzer.is_transitive_allocator(callee)
                    if is_alloc:
                        alloc_funcs.add(callee)
                except Exception:
                    pass
                try:
                    is_dealloc, _, _ = self._transitive_analyzer.is_transitive_deallocator(callee)
                    if is_dealloc:
                        free_funcs.add(callee)
                except Exception:
                    pass
        return alloc_funcs, free_funcs

    def _build_cfg(
        self,
        func: FunctionInfo,
        alloc_funcs: set[str],
        free_funcs: set[str],
        out_params: Optional[dict[str, int]] = None,
    ) -> Optional[dict[int, CFGNode]]:
        self._cur_func = func   # for the branch-condition constraints
        try:
            line_offset = (getattr(func, "start_line", 1) or 1) - 1
            if line_offset < 0:
                line_offset = 0
            builder = CFGBuilder(alloc_funcs, free_funcs, alloc_out_params=out_params or {},
                                 file_path=func.file_path)
            return builder.build(func.code, line_offset=line_offset)
        except Exception as e:
            logger.warning(f"Phase 3 CFG build failed for {func.name}: {e}")
            return None

    def _single_path_solver(self, cfg: dict[int, CFGNode]):
        """Single-path encoding: at most one incoming edge per node; reach iff
        some incoming edge is chosen."""
        solver = Solver()
        reach = {n.id: Bool(f"reach_{n.id}") for n in cfg.values()}
        entry = next(n for n in cfg.values() if n.node_type == NodeType.ENTRY)
        solver.add(reach[entry.id] == True)

        edge: dict[tuple[int, int], BoolRef] = {}
        for n in cfg.values():
            for p in n.predecessors:
                edge[(p, n.id)] = Bool(f"edge_{p}_{n.id}")

        for node in cfg.values():
            if node.id == entry.id:
                continue
            preds = node.predecessors
            if not preds:
                solver.add(reach[node.id] == False)
                continue
            incoming = [edge[(p, node.id)] for p in preds]
            solver.add(AtMost(*incoming, 1))
            for p in preds:
                solver.add(Implies(edge[(p, node.id)], reach[p]))
            solver.add(reach[node.id] == Or(*incoming))

        # Path condition: the true edge of a branch requires its condition,
        # the false edge its negation (identical conditions share a variable).
        self._add_branch_constraints(cfg, edge, solver)

        return solver, reach, edge, entry

    def _reachable_any(self, cfg: dict[int, CFGNode], node_ids: list[int]) -> bool:
        """True iff at least one of node_ids is reachable on some single path."""
        if not node_ids:
            return False
        solver, reach, _, _ = self._single_path_solver(cfg)
        solver.add(Or(*[reach[i] for i in node_ids]))
        return solver.check() == sat

    def _return_is_exact_var(self, code: str, var: str) -> bool:
        m = _RETURN_ONLY_VAR_RE.match(code or "")
        return bool(m and m.group("var") == var)

    def _return_may_carry(self, code: str) -> bool:
        """False iff the return statement yields nothing or a constant (NULL,
        0, a numeric/char/string literal), which cannot carry an allocation."""
        m = re.match(r"\s*return\b\s*(.*?);?\s*$", code or "", re.DOTALL)
        if not m:
            return True
        expr = m.group(1).strip()
        while True:
            prev = expr
            cast = re.match(r"\(\s*(?:const\s+)?[A-Za-z_][\w\s]*\**\s*\)\s*(?=\S)", expr)
            if cast:
                expr = expr[cast.end():].strip()
            elif expr.startswith("(") and expr.endswith(")"):
                expr = expr[1:-1].strip()
            if expr == prev:
                break
        if not expr:
            return False
        if expr in ("NULL", "nullptr", "false", "true"):
            return False
        return not re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*|'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"", expr)

    def _return_calls_alloc(self, code: str, alloc_funcs: set[str]) -> bool:
        """Match `return <alloc>(...)`, tolerating casts and wrapping parens."""
        m = re.match(r"\s*return\b\s*(.*?);?\s*$", code or "", re.DOTALL)
        if not m:
            return False
        expr = m.group(1).strip()
        expr = re.sub(r"^\(\s*[A-Za-z_][\w\s]*\*+\s*\)\s*", "", expr)
        expr = expr.lstrip("(").lstrip()
        m2 = re.match(r"([A-Za-z_]\w*)\s*\(", expr)
        return bool(m2 and m2.group(1) in alloc_funcs)

    def _feasible_alloc_to_return(
        self,
        cfg: dict[int, CFGNode],
        var: str,
        return_nodes: list[CFGNode],
    ) -> bool:
        """Z3: a feasible path on which `var` is allocated then returned, unfreed."""
        solver, reach, edge, entry = self._single_path_solver(cfg)
        allocd = {n.id: Bool(f"allocd_{n.id}") for n in cfg.values()}
        freed = {n.id: Bool(f"freed_{n.id}") for n in cfg.values()}
        solver.add(allocd[entry.id] == False)
        solver.add(freed[entry.id] == False)

        for node in cfg.values():
            if node.id == entry.id:
                continue
            preds = node.predecessors
            if not preds:
                solver.add(allocd[node.id] == False)
                solver.add(freed[node.id] == False)
                continue
            alloc_from = Or(*[And(edge[(p, node.id)], allocd[p]) for p in preds])
            freed_from = Or(*[And(edge[(p, node.id)], freed[p]) for p in preds])

            is_alloc = node.node_type == NodeType.ALLOC and (node.variable or "").strip() == var
            # `var = <expr not mentioning var>` overwrites the allocated value.
            is_overwrite = (
                node.node_type == NodeType.ASSIGN
                and (node.variable or "").strip() == var
                and not re.search(rf"\b{re.escape(var)}\b", node.obj or "")
            )
            if is_alloc:
                solver.add(allocd[node.id] == Or(alloc_from, reach[node.id]))
            elif is_overwrite:
                solver.add(allocd[node.id] == False)
            else:
                solver.add(allocd[node.id] == alloc_from)

            is_free = node.node_type == NodeType.FREE and (
                (node.variable or "").strip() == var or (node.obj or "").strip() == var
            )
            if is_free:
                solver.add(freed[node.id] == Or(freed_from, And(reach[node.id], allocd[node.id])))
            else:
                solver.add(freed[node.id] == freed_from)

        solver.add(Or(*[
            And(reach[r.id], allocd[r.id], Not(freed[r.id])) for r in return_nodes
        ]))
        return solver.check() == sat

    def _z3_allocator_feasible(
        self,
        func: FunctionInfo,
        alloc_funcs: set[str],
        free_funcs: set[str],
        out_param_idx: Optional[int],
    ) -> tuple[bool, str]:
        """(keep, reason): keep if a feasible path carries an allocation to the
        return (or out-parameter), or if there is no CFG evidence to decide."""
        cfg = self._build_cfg(func, alloc_funcs, free_funcs)
        if not cfg:
            return True, "CFG unavailable; kept (conservative)"

        alloc_nodes = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]
        return_nodes = [n for n in cfg.values() if n.node_type == NodeType.RETURN]

        if out_param_idx is not None and out_param_idx >= 0:
            if not alloc_nodes:
                return True, "no allocation evidence in CFG; kept (conservative)"
            arg_names = getattr(func, "arg_names", None) or []
            pname = arg_names[out_param_idx] if out_param_idx < len(arg_names) else ""
            targets = {f"*{pname}", pname} if pname else set()
            out_allocs = [
                a for a in alloc_nodes
                if (a.obj or "").strip() in targets or (a.variable or "").strip() == pname
            ]
            if not out_allocs and pname:
                alloc_locals = {(a.variable or "").strip() for a in alloc_nodes if (a.variable or "").strip()}
                if any(
                    n.node_type == NodeType.ASSIGN
                    and (n.variable or "").strip() in targets
                    and (n.obj or "").strip() in alloc_locals
                    for n in cfg.values()
                ):
                    out_allocs = alloc_nodes
            if not out_allocs:
                out_allocs = alloc_nodes
            if self._reachable_any(cfg, [a.id for a in out_allocs]):
                return True, "out-parameter allocation reachable on a feasible path"
            return False, "no feasible path where an allocation reaches the out-parameter"

        # Scan the source, not the CFG: returns inside loop bodies are not CFG nodes.
        src_returns = re.findall(r"\breturn\b[^;{}]*", _strip_c_comments(func.code or ""))
        if src_returns and not any(self._return_may_carry(r) for r in src_returns):
            return False, "every return yields a constant (e.g., NULL); no allocation reaches the caller"

        direct_returns = [r for r in return_nodes if self._return_calls_alloc(r.code, alloc_funcs)]

        if not alloc_nodes and not direct_returns:
            return True, "no allocation evidence in CFG; kept (conservative)"

        if direct_returns and self._reachable_any(cfg, [r.id for r in direct_returns]):
            return True, "allocation reaches return on a feasible path"

        untracked = []
        for a in alloc_nodes:
            v = (a.variable or "").strip()
            rets = [r for r in return_nodes if v and self._return_is_exact_var(r.code, v)]
            if not rets:
                untracked.append(a)
                continue
            if self._feasible_alloc_to_return(cfg, v, rets):
                return True, f"allocated '{v}' reaches return on a feasible path"

        # Allocations whose flow to the return is not tracked (no `return v;`)
        # are kept conservatively; tracked ones Z3 has just refuted.
        if untracked and self._reachable_any(cfg, [a.id for a in untracked]):
            return True, "reachable allocation present (flow to return not precisely tracked)"

        return False, "no feasible path where an allocation reaches the return"

    def _param_alias_names(self, func: FunctionInfo, arg_index: int) -> set[str]:
        return _param_alias_names(func, arg_index)

    def _z3_deallocator_feasible(
        self,
        func: FunctionInfo,
        free_funcs: set[str],
        arg_index: int,
    ) -> tuple[bool, str]:
        """Decide whether a deallocator summary survives, as (keep, reason)."""
        cfg = self._build_cfg(func, set(TransitiveAnalyzer.BASE_ALLOCATORS), free_funcs)
        if not cfg:
            return True, "CFG unavailable; kept (conservative)"

        free_nodes = [n for n in cfg.values() if n.node_type == NodeType.FREE]
        if not free_nodes:
            return True, "no free node in CFG; kept (conservative)"

        alias_names = self._param_alias_names(func, arg_index)
        if not alias_names:
            return True, f"arg{arg_index} name unresolved; kept (conservative)"

        target_frees = [
            f for f in free_nodes
            if (f.variable or "").strip() in alias_names or (f.obj or "").strip() in alias_names
        ]
        if not target_frees:
            return False, f"no free of arg{arg_index} (or its aliases) in CFG"

        if self._reachable_any(cfg, [f.id for f in target_frees]):
            return True, f"frees arg{arg_index} on a feasible path"
        return False, f"free of arg{arg_index} not reachable on any feasible path"


class WarningValidator(OwnershipAnalysis):
    """Validate CodeQL warnings using Z3 path feasibility analysis."""

    def __init__(
        self,
        alloc_funcs: set[str] = None,
        free_funcs: set[str] = None,
        ownership_sinks: Optional[set[str]] = None,
    ):
        self.base_alloc_funcs = set(alloc_funcs) if alloc_funcs else set(STANDARD_ALLOCATORS)
        self.base_free_funcs = set(free_funcs) if free_funcs else set(STANDARD_DEALLOCATORS)
        self._transitive_analyzer: Optional[TransitiveAnalyzer] = None
        # Calls that take ownership of a pointer argument (container insertion).
        self.ownership_sinks: set[str] = set(ownership_sinks) if ownership_sinks is not None else set(OWNERSHIP_SINKS)


    

    def _dump_cfg(self, cfg: dict[int, CFGNode], func_name: str, alloc_line: int = -1, exit_line: int = -1):
        logger.debug("=== CFG dump for %s (nodes=%d) ===", func_name, len(cfg))

        for n in sorted(cfg.values(), key=lambda x: (x.line, x.id)):
            mark = ""
            if alloc_line > 0 and n.line == alloc_line:
                mark += " [ALLOC_LINE]"
            if exit_line > 0 and n.line == exit_line and n.node_type in (NodeType.RETURN, NodeType.EXIT):
                mark += " [EXIT_LINE]"
            logger.debug(
                "id=%d line=%d type=%s preds=%s succs=%s%s | %s",
                n.id, n.line, n.node_type.name,
                n.predecessors, n.successors, mark,
                (n.code or "").strip().replace("\n", " ")[:120],
            )


    def validate_warnings(
        self,
        warnings: list[Warning],
        functions: dict[str, FunctionInfo],
        hints: Optional[HintSet] = None,
    ) -> tuple[list[Warning], list[Warning]]:
        """Validate warnings, returning (confirmed, filtered)."""
        if not Z3_AVAILABLE:
            return warnings, []

        # Allocation and deallocation operations: the standard primitives and
        # the validated summaries (Phase 3 already followed the call chains).
        alloc_funcs = set(self.base_alloc_funcs) | set(TransitiveAnalyzer.BASE_ALLOCATORS)
        free_funcs = set(self.base_free_funcs) | set(TransitiveAnalyzer.BASE_DEALLOCATORS)

        if hints:
            for func_name, arg_idx in hints.get_allocators():
                alloc_funcs.add(func_name)

            # Phase 3 has validated these deallocators; their call sites are
            # Free nodes without re-analysing the wrapper's body.
            for func_name, arg_idx in hints.get_deallocators():
                if func_name in functions and arg_idx is not None and arg_idx >= 0:
                    free_funcs.add(func_name)

        self._free_arg_idx = {}
        if hints:
            for func_name, arg_idx in hints.get_deallocators():
                if arg_idx is not None and arg_idx >= 0:
                    self._free_arg_idx.setdefault(func_name, arg_idx)

        confirmed = []
        filtered = []
        unconfirmed=[]

        for warning in warnings:
            func = functions.get(warning.function_name)
            if not func:
                logger.warning("Validator: function not found for warning: %s (available=%d). Example keys: %s",
                            warning.function_name, len(functions), list(functions.keys())[:20])
                unconfirmed.append(warning)
                continue

            alloc_out_params: dict[str, int] = {}
            if hints:
                for func_name, arg_idx in hints.get_allocators():
                    if arg_idx is not None and arg_idx >= 0:
                        alloc_out_params[func_name] = arg_idx

            result = self._check_feasibility(warning, func, alloc_funcs, free_funcs, alloc_out_params)
            # Recorded so that the dropped-warning files say why.
            warning.z3_reason = result.reason

            if result.is_feasible:
                confirmed.append(warning)
            else:
                # Includes "analysis failed" (the CFG or Z3 encoding raised):
                # a warning that cannot be encoded is discarded.
                filtered.append(warning)
                logger.debug(
                    f"Filtered {warning.function_name}:{warning.line_number}: {result.reason}"
                )

        return confirmed, filtered, unconfirmed
    
    def _check_feasibility(
        self,
        warning: Warning,
        func: FunctionInfo,
        alloc_funcs: set[str],
        free_funcs: set[str],
        alloc_out_params: Optional[dict[str, int]] = None,
    ) -> PathFeasibilityResult:
        """Check if warning's bug path is feasible."""
        
        logger.info(f"check_feasibility: {func.name}")
        try:
            # The same CFG construction as Phase 3 (same node classification).
            cfg_builder = CFGBuilder(alloc_funcs, free_funcs,
                                     alloc_out_params=alloc_out_params or {},
                                     file_path=getattr(func, "file_path", "") or "")
            cfg_builder.free_arg_idx = getattr(self, "_free_arg_idx", {})
            line_offset = (getattr(func, "start_line", 1) or 1) - 1
            if line_offset < 0:
                line_offset = 0

            cfg = cfg_builder.build(func.code, line_offset=line_offset)


            
            alloc_line = self._trace_alloc_line(warning)
            exit_line  = self._trace_end_line(warning)
            self._dump_cfg(cfg, warning.function_name, alloc_line, exit_line)

            if warning.issue_type in (MemoryIssueType.MEMORY_LEAK, MemoryIssueType.MEMORY_LEAK_FILTERED):
                self._local_names = self._declared_names(func)
                self._cur_func = func
                self._param_names = set(getattr(func, "arg_names", None) or [])
                return self._check_leak_feasibility_published(cfg, warning)
            else:
                return PathFeasibilityResult(is_feasible=False, reason="unsupported bug type in validator")

        except Exception as e:
            logger.error(f"Feasibility check failed: {e}")
            return PathFeasibilityResult(is_feasible=False, reason=f"analysis failed: {e}")


    def _mentions_var(self, expr: str, var: str) -> bool:
        if not var:
            return False
        return re.search(rf"\b{re.escape(var)}\b", expr) is not None


    def _is_ownership_transfer_call(self, code: str, alloc_var: str) -> bool:
        """Heuristic: treat some calls as ownership transfer if they mention alloc_var."""
        if not code or not alloc_var:
            return False
        if not self._mentions_var(code, alloc_var):
            return False
        for fn in self.ownership_sinks:
            if re.search(rf"\b{re.escape(fn)}\s*\(", code):
                return True
        return False

    def _collect_aliases_from_trace(
        self,
        cfg: dict[int, CFGNode],
        trace_lines: list[int],
        alloc_var: str,
    ) -> set[str]:
        """Collect trivial aliases from trace lines only (cheap, reduces FPs)."""
        aliases: set[str] = set()
        if not alloc_var:
            return aliases

        trace_set = set(trace_lines or [])
        for n in cfg.values():
            if n.node_type != NodeType.ASSIGN:
                continue
            if n.line not in trace_set:
                continue

            m = _ASSIGN_RE.match((n.code or "").strip())
            if not m:
                continue

            lhs = m.group("lhs").strip()
            rhs = m.group("rhs").strip()

            if not re.fullmatch(r"[A-Za-z_]\w*", lhs):
                continue

            if self._mentions_var(rhs, alloc_var):
                aliases.add(lhs)

        return aliases


    def _is_escape_event(self, code: str, alloc_var: str) -> bool:
        """The object itself (not a value read from it) is handed out: returned,
        stored into a global (g, g.f, g[i], g->f), or written through a
        pointer parameter (*out = p). Stores into a local's or a parameter's
        fields are not an escape."""
        code = code.strip()

        m = _RETURN_RE.match(code)
        if m and self._plain(m.group("expr")) == alloc_var:
            return True

        m = _ASSIGN_RE.match(code)
        if not m:
            return False
        lhs, rhs = m.group("lhs").strip(), m.group("rhs")
        rhs_last = re.split(r"(?<![=!<>+\-*/%&|^])=(?!=)", rhs)[-1]
        if self._plain(rhs_last.lstrip("&")) != alloc_var:
            return False
        mb = re.match(r"([\s*(]*)([A-Za-z_]\w*)", lhs)
        if not mb:
            return False
        stars, base = mb.group(1), mb.group(2)
        locals_ = getattr(self, "_local_names", None)
        params = set(getattr(self, "_param_names", ()) or ())
        if locals_ is None or base == alloc_var:
            return False
        if base not in locals_ and base not in params:
            return True                                # global store
        if "*" in stars and base in params and re.fullmatch(r"[\s*(]*[A-Za-z_]\w*\)?", lhs):
            return True                                # *out = p
        return False

    def _choose_alloc_var(self, cfg: dict[int, CFGNode], warning: Warning) -> str:
        target_line = self._trace_alloc_line(warning)

        alloc_nodes = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]

        if target_line > 0:
            exact = [n for n in alloc_nodes if n.line == target_line]
            for n in exact:
                ident = (n.variable or n.obj or "").strip()
                if ident:
                    return ident

        if target_line > 0:
            before = [n for n in alloc_nodes if n.line <= target_line and (n.variable or n.obj)]
            if before:
                n = max(before, key=lambda x: x.line)
                return (n.variable or n.obj).strip()

        for n in alloc_nodes:
            ident = (n.variable or n.obj or "").strip()
            if ident:
                return ident

        return ""

    def _parse_trace_lines(self, warning: Warning) -> list[int]:
        """warning.trace entries look like "layout-custom.c:176"."""
        lines = []
        for t in (warning.trace or []):
            m = re.search(r":(\d+)\b", str(t))
            if m:
                lines.append(int(m.group(1)))
        return lines

    def _trace_end_line(self, warning: Warning) -> int:
        lines = self._parse_trace_lines(warning)
        return lines[-1] if lines else -1

    def _trace_alloc_line(self, warning: Warning) -> int:
        alloc_site = (warning.allocation_site or "").strip()
        m = re.search(r":(\d+)\b", alloc_site)
        if m:
            return int(m.group(1))
        lines = self._parse_trace_lines(warning)
        return lines[0] if lines else -1


    def _check_leak_feasibility_published(
        self,
        cfg: dict[int, CFGNode],
        warning: Warning,
    ) -> PathFeasibilityResult:
        solver = Solver()
        solver.set(unsat_core=True)

        alloc_nodes = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]
        # Function exits: every return statement, plus the implicit exit a
        # function without a trailing return falls off into.
        exit_nodes = [n for n in cfg.values()
                      if n.node_type == NodeType.RETURN
                      or (n.node_type == NodeType.EXIT and n.predecessors)]
        if not alloc_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no allocation found")

        alloc_var = self._choose_alloc_var(cfg, warning)
        if not alloc_var:
            return PathFeasibilityResult(is_feasible=False, reason="unknown allocated variable (conservative)")

        reach   = {n.id: Bool(f"reach_{n.id}")   for n in cfg.values()}
        allocd  = {n.id: Bool(f"allocd_{n.id}")  for n in cfg.values()}
        freed   = {n.id: Bool(f"freed_{n.id}")   for n in cfg.values()}
        escaped = {n.id: Bool(f"escaped_{n.id}") for n in cfg.values()}

        entry = next(n for n in cfg.values() if n.node_type == NodeType.ENTRY)
        solver.add(reach[entry.id] == True)
        solver.add(allocd[entry.id] == False)
        solver.add(freed[entry.id] == False)
        solver.add(escaped[entry.id] == False)

        alloc_line = self._trace_alloc_line(warning)

        if alloc_line > 0:
            matching_nodes = [n for n in cfg.values() if n.line == alloc_line]
            if matching_nodes:
                solver.add(Or(*[reach[n.id] for n in matching_nodes]))
            else:
                logger.warning(f"No CFG nodes at allocation line {alloc_line} for {warning.function_name}")

        # The question is whether any feasible path from this allocation reaches
        # an exit with the object unfreed and unescaped, so the analyzer's own
        # trace is used only to locate the allocation, not to pin the path.
        trace_lines = self._parse_trace_lines(warning)
        alias_vars = self._collect_aliases_from_trace(cfg, trace_lines, alloc_var)
        alias_vars.add(alloc_var)
        fn = getattr(self, "_cur_func", None)
        if fn is not None and re.fullmatch(r"[A-Za-z_]\w*", alloc_var):
            try:
                ai = SimpleAliasAnalyzer().analyze(fn)
                alias_vars |= {v for v, src in ai.aliases.items() if alloc_var in src}
            except Exception as e:
                logger.debug(f"alias analysis failed for {fn.name}: {e}")

        def alloc_matches_var(node: CFGNode) -> bool:
            if node.node_type != NodeType.ALLOC:
                return False
            ident = (node.variable or node.obj or "").strip()
            return ident == alloc_var

        def free_matches_alloc(node: CFGNode) -> bool:
            if node.node_type != NodeType.FREE:
                return False

            if node.variable and node.variable in alias_vars:
                return True

            o = (node.obj or "").strip()
            if o and (o == alloc_var or o in alias_vars):
                return True

            return False

        # On the failure edge of a NULL test of the allocated pointer
        # (`if (!p)`, `if (p == NULL)`), nothing was allocated.
        failed_alloc = self._oom_edges(cfg, only=alias_vars)
        # A pointer parameter that the function re-points (`out = &local;`)
        # no longer reaches the caller's memory.
        fn_ = getattr(self, "_cur_func", None)
        param_rebound = self._assignment_lines(fn_) if fn_ is not None else {}

        owner = ""
        m_own = re.match(r"\s*([A-Za-z_]\w*)\s*(->|\.)", alloc_var)
        if m_own and m_own.group(1) in getattr(self, "_local_names", set()):
            owner = m_own.group(1)

        edge: dict[tuple[int, int], BoolRef] = {}
        for n in cfg.values():
            for p in n.predecessors:
                edge[(p, n.id)] = Bool(f"edge_{p}_{n.id}")

        for node in cfg.values():
            if node.id == entry.id:
                continue

            preds = node.predecessors
            if not preds:
                solver.add(reach[node.id] == False)
                solver.add(allocd[node.id] == False)
                solver.add(freed[node.id] == False)
                solver.add(escaped[node.id] == False)
                continue

            incoming = [edge[(p, node.id)] for p in preds]

            solver.add(AtMost(*incoming, 1))

            for p in preds:
                solver.add(Implies(edge[(p, node.id)], reach[p]))

            solver.add(reach[node.id] == Or(*incoming))

            # nothing allocated flows along the failure edge of a NULL test
            alloc_from = Or(*([And(edge[(p, node.id)], allocd[p]) for p in preds
                               if (p, node.id) not in failed_alloc] or [BoolVal(False)]))
            freed_from = Or(*[And(edge[(p, node.id)], freed[p]) for p in preds])
            esc_from   = Or(*[And(edge[(p, node.id)], escaped[p]) for p in preds])

            if alloc_matches_var(node):
                solver.add(allocd[node.id] == Or(alloc_from, reach[node.id]))
            else:
                solver.add(allocd[node.id] == alloc_from)

            if free_matches_alloc(node):
                solver.add(freed[node.id] == Or(freed_from, And(reach[node.id], allocd[node.id])))
            else:
                solver.add(freed[node.id] == freed_from)

            escape_now = False

            if (node.node_type == NodeType.ALLOC and alloc_matches_var(node)
                    and (node.obj or "").startswith("*")
                    and (node.variable or "") in set(getattr(self, "_param_names", ()) or ())
                    and not param_rebound.get(node.variable)):
                escape_now = True          # f(out) allocates *out for the caller

            if alloc_matches_var(node):
                # Allocated straight into a global (g = alloc(), g.f = alloc(),
                # g[i] = alloc()): global store. A parameter's fields stay the
                # caller's business, and a local's fields are not a store.
                m_base = re.match(r"[\s*(&]*([A-Za-z_]\w*)", alloc_var)
                base = m_base.group(1) if m_base else ""
                locals_ = getattr(self, "_local_names", None)
                params = set(getattr(self, "_param_names", ()) or ())
                if base and locals_ is not None and base not in locals_ and base not in params:
                    escape_now = True

            for v in alias_vars:
                if node.node_type == NodeType.RETURN and self._return_is_exact_var(node.code, v):
                    escape_now = True
                if node.node_type == NodeType.ASSIGN and self._is_escape_event(node.code, v):
                    escape_now = True
                if node.node_type == NodeType.DEREF and self._is_ownership_transfer_call(node.code, v):
                    escape_now = True

            # The object was stored in a field of a local object (o->f = p,
            # o->f = alloc()): it escapes when its container does.
            if owner:
                if node.node_type == NodeType.RETURN and self._return_is_exact_var(node.code, owner):
                    escape_now = True
                if node.node_type == NodeType.ASSIGN and self._is_escape_event(node.code, owner):
                    escape_now = True
                if node.node_type == NodeType.DEREF and self._is_ownership_transfer_call(node.code, owner):
                    escape_now = True

            if escape_now:
                solver.add(escaped[node.id] == Or(esc_from, And(reach[node.id], allocd[node.id])))
            else:
                solver.add(escaped[node.id] == esc_from)

        if not exit_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no exit/return nodes in CFG")

        self._add_branch_constraints(cfg, edge, solver)

        leak_conditions = [
            And(reach[x.id], allocd[x.id], Not(freed[x.id]), Not(escaped[x.id]))
            for x in exit_nodes
        ]
        solver.add(Or(*leak_conditions))

        result = solver.check()
        if result == sat:
            return PathFeasibilityResult(is_feasible=True, reason=f"leak path exists for {alloc_var}")
        return PathFeasibilityResult(is_feasible=False, reason=f"all paths free/escape or never allocate {alloc_var}")


