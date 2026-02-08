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
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from collections import defaultdict

try:
    from z3 import (
        Solver, Bool, Int, And, Or, Not, Implies,
        sat, unsat, BoolRef
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logging.warning("Z3 not available - install with: pip install z3-solver")

import tree_sitter_c as tsc
from tree_sitter import Language, Parser

from src.core.models import (
    FunctionInfo, Hint, HintType, HintSet,
    Warning, MemoryIssueType, ValidationResult, PathFeasibilityResult
)

logger = logging.getLogger(__name__)

C_LANGUAGE = Language(tsc.language())

_ASSIGN_RE = re.compile(r"^\s*(?P<lhs>.+?)\s*=\s*(?P<rhs>.+?);?\s*$")
_RETURN_RE = re.compile(r"^\s*return\s+(?P<expr>.+?);?\s*$")


# =============================================================================
# CFG Data Structures
# =============================================================================

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


@dataclass
class CFGNode:
    id: int
    node_type: NodeType
    line: int
    code: str
    variable: str = ""   # keep for legacy / leak checks 
    obj: str = ""        # NEW: full expression identity for DF/UAF (e.g., acl_args[i], p->x)
    condition: str = ""
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)


# =============================================================================
# Call Graph for Transitive Analysis
# =============================================================================

@dataclass
class CallSite:
    """Represents a function call site."""
    caller: str
    callee: str
    line: int
    arg_mapping: dict[int, int] = field(default_factory=dict)
    arg_expressions: dict[int, str] = field(default_factory=dict)
    is_indirect: bool = False  # Function pointer call


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

    def build(self, functions) -> CallGraph:
        """Build call graph from all functions.

        Robust: functions[name] can be FunctionInfo-like OR dict-like.
        """
        cg = CallGraph()

        for func_name, func_info in functions.items():
            # Skip obviously wrong entries early
            if func_info is None:
                continue
            self._extract_calls(func_name, func_info, cg)

        return cg




    def _extract_calls(self, func_name: str, func_info, cg: CallGraph):
        """Extract all call sites from a function.

        Accepts:
        - FunctionInfo-like objects with .code / .arg_names
        - dict-like objects with ["code"] / ["arg_names"]
        """
        # --- robust read code ---
        if isinstance(func_info, dict):
            code = func_info.get("code", "") or ""
        else:
            code = getattr(func_info, "code", "") or ""

        if not isinstance(code, str):
            code = str(code)

        src_bytes = code.encode("utf-8", errors="ignore")
        tree = self.parser.parse(src_bytes)

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
        line = node.start_point[0] + 1

        is_indirect = func_node.type != "identifier"

        # robust arg_names
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


# =============================================================================
# Alias Analysis (Simple)
# =============================================================================

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

    def analyze(self, func_info: FunctionInfo) -> AliasInfo:
        """Analyze aliases in a function."""
        tree = self.parser.parse(func_info.code.encode())
        source = func_info.code.encode()

        alias_info = AliasInfo()

        # Initialize parameters as aliasing themselves
        for idx, param in enumerate(func_info.arg_names):
            alias_info.aliases[param].add(param)
            alias_info.param_aliases[param] = idx

        # Find assignments
        for node in self._walk(tree.root_node):
            if node.type == "assignment_expression":
                self._process_assignment(node, source, func_info, alias_info)
            elif node.type == "init_declarator":
                self._process_init(node, source, func_info, alias_info)

        # Compute transitive closure
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

        if lhs.type == "identifier" and rhs.type == "identifier":
            lhs_name = source[lhs.start_byte:lhs.end_byte].decode()
            rhs_name = source[rhs.start_byte:rhs.end_byte].decode()

            # lhs now aliases everything rhs aliases
            alias_info.aliases[lhs_name].add(rhs_name)
            alias_info.aliases[lhs_name].update(alias_info.aliases.get(rhs_name, set()))

            # If rhs is a parameter alias, so is lhs
            if rhs_name in alias_info.param_aliases:
                alias_info.param_aliases[lhs_name] = alias_info.param_aliases[rhs_name]

    def _process_init(
        self,
        node,
        source: bytes,
        func_info: FunctionInfo,
        alias_info: AliasInfo
    ):
        """Process initialization: type var = expr"""
        declarator = None
        value = None

        for child in node.children:
            if child.type in ("identifier", "pointer_declarator"):
                if child.type == "pointer_declarator":
                    # Find identifier inside pointer_declarator
                    for sub in self._walk(child):
                        if sub.type == "identifier":
                            declarator = sub
                            break
                else:
                    declarator = child
            elif child.type == "identifier" and declarator is not None:
                value = child

        if declarator and value:
            var_name = source[declarator.start_byte:declarator.end_byte].decode()
            val_name = source[value.start_byte:value.end_byte].decode()

            alias_info.aliases[var_name].add(val_name)
            alias_info.aliases[var_name].update(alias_info.aliases.get(val_name, set()))

            if val_name in alias_info.param_aliases:
                alias_info.param_aliases[var_name] = alias_info.param_aliases[val_name]

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


# =============================================================================
# Transitive Allocator/Deallocator Analyzer
# =============================================================================

class TransitiveAnalyzer:
    """Analyze transitive allocator/deallocator relationships."""

    BASE_ALLOCATORS = {
        "malloc", "calloc", "realloc", "strdup", "strndup",
        "aligned_alloc", "memalign", "pvalloc", "valloc",
        "g_malloc", "g_malloc0", "g_new", "g_new0",
        "kmalloc", "kzalloc", "vmalloc", "kcalloc",
        "xmalloc", "xcalloc", "xrealloc",
    }

    BASE_DEALLOCATORS = {
        "free", "cfree",
        "g_free",
        "kfree", "vfree", "kfree_sensitive",
        "xfree",
        "sdsfree", "zfree", "decrRefCount",
    }

    def __init__(self, call_graph: CallGraph, functions: dict[str, FunctionInfo]):
        self.cg = call_graph
        self.functions = functions
        self.parser = Parser(C_LANGUAGE)
        self.alias_analyzer = SimpleAliasAnalyzer()

        self._is_allocator_cache: dict[str, tuple[bool, list[str]]] = {}
        self._is_deallocator_cache: dict[str, tuple[bool, int, list[str]]] = {}
        self._alias_cache: dict[str, AliasInfo] = {}

        # Add known deallocators from functions
        self._known_deallocators: set[str] = set(self.BASE_DEALLOCATORS)
        self._precompute_known_deallocators()

    def _precompute_known_deallocators(self):
        """Pre-scan functions to find obvious deallocators."""
        for func_name, func_info in self.functions.items():
            code = func_info.code
            # Quick check: does it call a known deallocator?
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
        """Check if function is a deallocator.

        Handles:
        1. Direct calls: freeClient(c) where freeClient calls free
        2. Alias calls: target = c; freeClient(target)
        3. Indirect calls via known deallocator patterns

        Returns:
            (is_deallocator, freed_arg_index, call_chain)
        """
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

        # Method 1: Check direct calls to deallocators (including transitive)
        result = self._check_deallocator_recursive(func_name, [], set(), max_depth)
        if result[0]:
            return result

        # Method 2: Check calls through aliases
        result = self._check_deallocator_via_alias(func_name, func_info, alias_info, max_depth)
        if result[0]:
            return result

        # Method 3: Pattern matching for common deallocator patterns
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

                # Don't guess. Many functions free internal fields / owned containers.
                # If we can't map the freed argument precisely, treat as unknown.
                continue

        return (False, -1, [])

    def _check_deallocator_via_alias(
        self,
        func_name: str,
        func_info: FunctionInfo,
        alias_info: Optional[AliasInfo],
        max_depth: int
    ) -> tuple[bool, int, list[str]]:
        """Check if function frees a parameter through an alias.

        Example:
            void handleClientWithAlias(client *c) {
                client *target = c;
                freeClient(target);  // frees c through alias
            }
        """
        if not alias_info:
            return (False, -1, [])

        code = func_info.code

        # Find all calls to known deallocators
        for callsite in self.cg.callsites.get(func_name, []):
            callee = callsite.callee

            # Check if callee is a known deallocator
            if callee not in self._known_deallocators:
                continue

            # Check arguments - do any alias a parameter?
            for arg_idx, arg_expr in callsite.arg_expressions.items():
                # Check if this argument aliases any parameter
                if arg_expr in alias_info.param_aliases:
                    param_idx = alias_info.param_aliases[arg_expr]
                    return (True, param_idx, [func_name, callee])

                # Check transitive aliases
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
        """Pattern-based detection for deallocators.

        Handles cases like:
            void processEvent(event *ev) {
                client *c = ev->client;
                switch(...) {
                    case 0: freeClient(c); break;
                }
            }

        Also handles:
            void func(client *c) {
                if (cond) freeClient(c);  // Conditional free
            }
        """
        code = func_info.code

        # Find calls to known deallocators
        for dealloc in self._known_deallocators:
            # Pattern: dealloc(var) where var is derived from a parameter
            pattern = rf'{re.escape(dealloc)}\s*\(\s*(\w+)\s*\)'
            matches = re.finditer(pattern, code)

            for match in matches:
                freed_var = match.group(1)

                # Check if freed_var is a parameter
                if freed_var in func_info.arg_names:
                    param_idx = func_info.arg_names.index(freed_var)
                    return (True, param_idx, [func_name, dealloc])

                # Check if freed_var is derived from a parameter
                # Pattern: type *var = param->field or var = param
                derived_patterns = [
                    rf'(\w+)\s*\*?\s*{re.escape(freed_var)}\s*=\s*(\w+)',  # var = param
                    rf'{re.escape(freed_var)}\s*=\s*(\w+)->',  # var = param->field
                    rf'{re.escape(freed_var)}\s*=\s*(\w+)\s*;',  # var = param;
                ]

                for dp in derived_patterns:
                    dm = re.search(dp, code)
                    if dm:
                        # Get the source variable
                        source_var = dm.group(2) if dm.lastindex >= 2 else dm.group(1)
                        if source_var in func_info.arg_names:
                            param_idx = func_info.arg_names.index(source_var)
                            return (True, param_idx, [func_name, dealloc])

        return (False, -1, [])


# =============================================================================
# CFG Builder
# =============================================================================

class CFGBuilder:
    """Build Control Flow Graph from C code."""

    def __init__(
        self,
        alloc_funcs: set[str] = None,
        free_funcs: set[str] = None,
        alloc_out_params: dict[str, int] = None,
    ):
        self.parser = Parser(C_LANGUAGE)
        self.alloc_funcs = alloc_funcs or {"malloc", "calloc", "realloc", "strdup"}
        self.free_funcs = free_funcs or {"free"}
        # func_name -> out param index (e.g., posix_memalign style)
        self.alloc_out_params = alloc_out_params or {}


    def build(self, code: str) -> dict[int, CFGNode]:
        """Build CFG from function code.

        Fixes:
        - Real if/else control-flow edges (not a linear chain)
        - Emits RETURN nodes as exits
        - Keeps predecessors/successors consistent
        """
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
                line=line,
                code=code_txt,
                variable=variable,
                obj=obj,
                condition=condition,
                successors=[],
                predecessors=[],
            )
            node_id += 1
            return nid


        def add_edge(a: int, b: int):
            if b not in nodes[a].successors:
                nodes[a].successors.append(b)
            if a not in nodes[b].predecessors:
                nodes[b].predecessors.append(a)

        entry_id = new_node(NodeType.ENTRY, 0, "entry")

        # Build CFG only from the function body (compound_statement) to avoid noise.
        body = None
        for ch in tree.root_node.children:
            if ch.type == "function_definition":
                for c2 in ch.children:
                    if c2.type == "compound_statement":
                        body = c2
                        break
                break
        if body is None:
            # Fallback: walk whole tree if we couldn't find a body
            body = tree.root_node

        exit_id = new_node(NodeType.EXIT, 0, "exit")

        def build_stmt_list(stmt_list_node, in_id: int) -> int:
            """Build sequential CFG for a list of statements.
            Returns the 'tail' node id after the list.
            """
            cur = in_id
            for st in stmt_list_node.children:
                if st.type in ("{", "}"):
                    continue
                cur = build_stmt(st, cur)
            return cur

        def build_stmt(st_node, in_id: int) -> int:
            """Build CFG for a statement; returns the tail node id."""
            # if-statement: create BRANCH and split edges
            if st_node.type == "if_statement":
                return build_if(st_node, in_id)

            # return-statement: create RETURN node and edge to EXIT
            if st_node.type == "return_statement":
                rid = new_node(
                    NodeType.RETURN,
                    st_node.start_point[0] + 1,
                    source[st_node.start_byte:st_node.end_byte].decode(errors="ignore"),
                )
                add_edge(in_id, rid)
                add_edge(rid, exit_id)
                return rid

            # expression_statement: may contain call_expression / deref / etc.
            # We'll scan inside it and emit nodes for interesting operations
            tail = in_id
            created_any = False
            for n in st_node.children:
                # or only walk one level down, or filter
                cfg_n = self._process_node(n, source, node_id)
                if cfg_n:
                    # materialize the node into `nodes` with a fresh id
                    nid = new_node(cfg_n.node_type, cfg_n.line, cfg_n.code, cfg_n.variable, cfg_n.obj, cfg_n.condition)
                    add_edge(tail, nid)
                    tail = nid
                    created_any = True

            if created_any:
                return tail

            # default: ignore, keep flow
            return in_id

        def build_if(if_node, in_id: int) -> int:
            """Build CFG for if/else with proper join."""
            cond = if_node.child_by_field_name("condition")
            then_part = if_node.child_by_field_name("consequence")
            else_part = if_node.child_by_field_name("alternative")

            cond_txt = ""
            if cond is not None:
                cond_txt = source[cond.start_byte:cond.end_byte].decode(errors="ignore")

            bid = new_node(
                NodeType.BRANCH,
                if_node.start_point[0] + 1,
                source[if_node.start_byte:if_node.end_byte].decode(errors="ignore")[:80],
                condition=cond_txt,
            )
            add_edge(in_id, bid)

            join_id = new_node(NodeType.ASSIGN, if_node.end_point[0] + 1, "join")  # generic join node

            # THEN branch
            then_tail = bid
            if then_part is not None:
                if then_part.type == "compound_statement":
                    then_tail = build_stmt_list(then_part, bid)
                else:
                    then_tail = build_stmt(then_part, bid)
            add_edge(then_tail, join_id)

            # ELSE branch
            else_tail = bid
            if else_part is not None:
                if else_part.type == "compound_statement":
                    else_tail = build_stmt_list(else_part, bid)
                else:
                    else_tail = build_stmt(else_part, bid)
            add_edge(else_tail, join_id)

            return join_id

        # Build the body statements starting from entry
        tail = entry_id
        if body.type == "compound_statement":
            tail = build_stmt_list(body, entry_id)
        else:
            tail = build_stmt(body, entry_id)

        # If the last statement didn't return, connect to EXIT
        if exit_id not in nodes[tail].successors:
            add_edge(tail, exit_id)

        return nodes


    def _walk(self, node):
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _process_node(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        if node.type == "call_expression":
            return self._process_call(node, source, node_id)
        elif node.type == "if_statement":
            return self._process_if(node, source, node_id)
        elif node.type == "return_statement":
            return self._process_return(node, source, node_id)
        elif node.type in ("pointer_expression", "subscript_expression", "field_expression"):
            return self._process_deref(node, source, node_id)

        # NEW: capture escapes like Users = raxNew(), UsersToLoad = listCreate(), etc.
        elif node.type == "assignment_expression":
            return self._process_assign(node, source, node_id)
        elif node.type == "init_declarator":
            return self._process_init_declarator(node, source, node_id)

        return None

    
    def _process_assign(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """
        Emit ASSIGN node so leak checker can detect escape events like:
        *out = p
        global = p
        obj->field = p
        """
        line = node.start_point[0] + 1
        code = source[node.start_byte:node.end_byte].decode(errors="ignore").strip()
        if not code:
            return None
        return CFGNode(
            id=node_id,
            node_type=NodeType.ASSIGN,
            line=line,
            code=code,
        )
        
    def _node_text(self, node, source: bytes) -> str:
        return source[node.start_byte:node.end_byte].decode(errors="ignore")

    def _process_assign(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """
        assignment_expression: <left> = <right>
        We emit ASSIGN so leak checker can detect escape sinks:
        - global_var = ...
        - obj->field = ...
        - *out = ...
        """
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return None

        lhs_txt = self._node_text(left, source).strip()
        rhs_txt = self._node_text(right, source).strip()

        # Keep full assignment text (without requiring semicolon)
        code = f"{lhs_txt} = {rhs_txt}"

        return CFGNode(
            id=node_id,
            node_type=NodeType.ASSIGN,
            line=node.start_point[0] + 1,
            code=code,
            variable=lhs_txt,  # store LHS (may be global, field, *out, local)
            obj=rhs_txt,       # store RHS (useful later if you want)
        )

    def _process_init_declarator(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """
        init_declarator: declarator '=' value
        Examples:
        user *u = ACLCreateUser(...)
        sds *acl_args = ACLMergeSelectorArguments(...)
        Emit ASSIGN so leak checker can see escape when initializer is assigned to global/field.
        (Even for locals, this helps build a consistent model.)
        """
        value = node.child_by_field_name("value")
        if value is None:
            return None

        # Find the declared identifier on the LHS
        declarator = node.child_by_field_name("declarator")
        lhs_name = ""

        if declarator is not None:
            # walk to find identifier inside pointer_declarator etc.
            stack = [declarator]
            while stack:
                cur = stack.pop()
                if cur.type == "identifier":
                    lhs_name = self._node_text(cur, source).strip()
                    break
                stack.extend(list(cur.children))

        if not lhs_name:
            # fallback: best effort by scanning children
            for ch in node.children:
                if ch.type == "identifier":
                    lhs_name = self._node_text(ch, source).strip()
                    break

        rhs_txt = self._node_text(value, source).strip()
        if not rhs_txt:
            return None

        code = f"{lhs_name} = {rhs_txt}" if lhs_name else rhs_txt

        return CFGNode(
            id=node_id,
            node_type=NodeType.ASSIGN,
            line=node.start_point[0] + 1,
            code=code,
            variable=lhs_name,
            obj=rhs_txt,
        )



    def _process_deref(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """Create a DEREF node when we see obvious pointer/array/member dereference.

        We only handle easy cases:
        *p
        p[i]
        p->field / p.field  (field_expression)
        and extract the base identifier as node.variable.
        """
        line = node.start_point[0] + 1
        code = source[node.start_byte:node.end_byte].decode(errors="ignore")

        var = ""

        # pointer_expression: '*' <argument>
        if node.type == "pointer_expression":
            arg = node.child_by_field_name("argument")
            if arg and arg.type == "identifier":
                var = source[arg.start_byte:arg.end_byte].decode(errors="ignore")

        # subscript_expression: <argument> '[' <index> ']'
        elif node.type == "subscript_expression":
            arg = node.child_by_field_name("argument")
            if arg and arg.type == "identifier":
                var = source[arg.start_byte:arg.end_byte].decode(errors="ignore")

        # field_expression: <argument> ('.'|'->') <field>
        elif node.type == "field_expression":
            arg = node.child_by_field_name("argument")
            if arg:
                # common easy case: identifier
                if arg.type == "identifier":
                    var = source[arg.start_byte:arg.end_byte].decode(errors="ignore")
                # or parenthesized: (p)->x
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
            obj=code.strip(),  # full deref expr like p->x, p[i], *p
        )


    
    def _extract_base_identifier(self, node, source: bytes) -> str:
        """Best-effort: extract the base identifier used as a pointer.

        Examples:
        *p            -> p
        p->x          -> p
        p.x           -> p
        p[i]          -> p
        (*p)->x       -> p
        (p)->x        -> p
        """
        cur = node

        # Unwrap parentheses / casts
        while cur is not None and cur.type in ("parenthesized_expression", "cast_expression"):
            inner = cur.child_by_field_name("value") or cur.child_by_field_name("expression")
            if inner is None and cur.children:
                inner = cur.children[-1]
            cur = inner

        if cur is None:
            return ""

        # pointer_expression: *expr
        if cur.type == "pointer_expression":
            arg = cur.child_by_field_name("argument")
            if arg is None and cur.children:
                arg = cur.children[-1]
            return self._extract_base_identifier(arg, source)

        # field_expression: obj.field or obj->field
        if cur.type == "field_expression":
            obj = cur.child_by_field_name("argument")
            if obj is None and cur.children:
                obj = cur.children[0]
            return self._extract_base_identifier(obj, source)

        # subscript_expression: arr[idx]
        if cur.type == "subscript_expression":
            arr = cur.child_by_field_name("argument") or cur.child_by_field_name("array")
            if arr is None and cur.children:
                arr = cur.children[0]
            return self._extract_base_identifier(arr, source)

        # identifier: base case
        if cur.type == "identifier":
            return source[cur.start_byte:cur.end_byte].decode(errors="ignore")

        # For member call like p->f() the "function" is a field_expression; base id is inside it.
        if cur.type == "call_expression":
            fn = cur.child_by_field_name("function")
            if fn is not None:
                return self._extract_base_identifier(fn, source)

        return ""



    def _process_call(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        func_node = node.child_by_field_name("function")
        if not func_node:
            return None

        func_name = source[func_node.start_byte:func_node.end_byte].decode(errors="ignore")
        line = node.start_point[0] + 1
        code = source[node.start_byte:node.end_byte].decode(errors="ignore")

        # Out-parameter style allocator: func(..., &out, ...) allocates into argument
        if func_name in self.alloc_out_params:
            out_idx = self.alloc_out_params[func_name]
            return CFGNode(
                id=node_id,
                node_type=NodeType.ALLOC,
                line=line,
                code=code,
                variable=f"arg{out_idx}",
                obj=f"arg{out_idx}",
            )


        # Regular allocators (return-value style)
        if func_name in self.alloc_funcs:
            parent = node.parent
            var = ""
            if parent and parent.type in ("assignment_expression", "init_declarator"):
                for child in parent.children:
                    if child.type == "identifier":
                        var = source[child.start_byte:child.end_byte].decode(errors="ignore")
                        break
            alloc_obj = var  # best we can do for return-value style alloc
            return CFGNode(
                id=node_id,
                node_type=NodeType.ALLOC,
                line=line,
                code=code,
                variable=var,
                obj=alloc_obj,
            )


        # Frees
        if func_name in self.free_funcs:
            args = node.child_by_field_name("arguments")
            obj = ""
            var = ""  # keep identifier-only if you still want it
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
                # if it's a plain identifier, also populate var
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


        # Generic "use": if this call passes an identifier argument, treat it as a use site.
        # This enables UAF detection even if the deref happens in the callee.
        args = node.child_by_field_name("arguments")
        # Generic "use": treat passing a pointer-ish expression as a use site (helps UAF)
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


# =============================================================================
# Hint Validator
# =============================================================================

class HintValidator:
    """Validate LLM-generated hints using Z3 constraint solving.

    Supports:
    - Direct allocation/deallocation calls
    - Transitive call chain analysis (wrapper functions)
    - Alias analysis (target = c; free(target))
    - Pattern-based detection (switch-case, conditionals)
    - Indirect calls via function pointers (conservative)
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

        validated = HintSet()
        conflicts = []

        for func_name, func_hints in hints.hints.items():
            if func_name not in functions:
                continue

            func = functions[func_name]

            for hint in func_hints:
                result = self._validate_hint(hint, func, functions)
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

    def _validate_allocator(
        self,
        hint: Hint,
        func: FunctionInfo,
        functions: dict[str, FunctionInfo]
    ) -> ValidationResult:
        """Validate ALLOCATOR hint."""
        code = func.code
        func_name = hint.function_name

        # Check if it's an out-parameter allocator (e.g., posix_memalign)
        if hint.arg_index >= 0:
            # Out-parameter style: int func(void **out, ...)
            # Check if the specified argument is a pointer-to-pointer
            if hint.arg_index < len(func.arg_types):
                arg_type = func.arg_types[hint.arg_index]
                if '**' not in arg_type:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Argument {hint.arg_index} is not a pointer-to-pointer type"
                    )

            # Check if allocation result is written to this argument
            alloc_funcs = ["malloc", "calloc", "realloc", "strdup", "strndup",
                        "g_malloc", "g_new", "kmalloc"]
            has_alloc = any(f"{a}(" in code for a in alloc_funcs)

            if has_alloc:
                return ValidationResult(is_valid=True, reason="out-parameter allocation found")

            # Check transitive
            if self._transitive_analyzer:
                is_alloc, chain = self._transitive_analyzer.is_transitive_allocator(func_name)
                if is_alloc:
                    return ValidationResult(
                        is_valid=True,
                        reason=f"out-parameter, chain: {' -> '.join(chain)}"
                    )

            return ValidationResult(
                is_valid=False,
                reason="No allocation found for out-parameter"
            )

        # Return-value style allocator (arg_index == -1)
        if not func.return_type or '*' not in func.return_type:
            return ValidationResult(
                is_valid=False,
                reason="Does not return pointer type"
            )

        alloc_funcs = ["malloc", "calloc", "realloc", "strdup", "strndup",
                    "g_malloc", "g_new", "kmalloc"]
        has_direct_alloc = any(f"{a}(" in code for a in alloc_funcs)

        if has_direct_alloc:
            return ValidationResult(is_valid=True, reason="direct allocation found")

        if self._transitive_analyzer:
            is_alloc, chain = self._transitive_analyzer.is_transitive_allocator(func_name)
            if is_alloc:
                return ValidationResult(
                    is_valid=True,
                    reason=f"chain: {' -> '.join(chain)}"
                )

        if "return" not in code.lower():
            return ValidationResult(
                is_valid=False,
                reason="No allocation call and no return"
            )

        return ValidationResult(is_valid=True, reason="validated (weak)")

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
            # If a hint claims "argK is deallocated", require evidence that the function (or its chain)
            # frees that argument *itself* (free(argK) / sdsfree(argK) / decrRefCount(argK) ...),
            # NOT merely frees fields reachable from it (argK->field, listEmpty(argK->...)).
            if hint.arg_index >= 0:
                if not self._frees_argument_itself(func, hint.arg_index):
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Does not free arg{hint.arg_index} itself (only internal members / containers)"
                    )

            # Verify argument index if specified
            if hint.arg_index >= 0 and freed_arg >= 0 and hint.arg_index != freed_arg:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Hint says arg {hint.arg_index} freed, but analysis shows arg {freed_arg}"
                )

            chain_str = " -> ".join(chain) if chain else func_name
            return ValidationResult(
                is_valid=True,
                reason=f"chain: {chain_str}, frees arg {freed_arg}"
            )

        # Check for indirect calls (function pointers) - be conservative
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
        free_funcs = ["free", "g_free", "kfree", "vfree"]

        has_direct_free = any(f"{f}(" in code for f in free_funcs)

        if has_direct_free:
            return ValidationResult(is_valid=True, reason="direct free found")

        return ValidationResult(
            is_valid=False,
            reason="No deallocation call found"
        )

    def _has_indirect_free_call(self, func: FunctionInfo) -> bool:
        """Check if function has indirect (function pointer) calls that might free.

        Example: cb(c, privdata) where cb is a function pointer
        """
        # Look for function pointer call patterns
        code = func.code

        # Pattern: identifier that's a parameter being called
        for param in func.arg_names:
            # Check if parameter is called as function
            if re.search(rf'\b{re.escape(param)}\s*\(', code):
                return True

        return False

    def _frees_argument_itself(self, func: FunctionInfo, arg_index: int) -> bool:
        """Return True iff code contains a clear deallocation of the argument variable itself.

        Accepts patterns like:
        free(p)
        sdsfree(p)
        decrRefCount(p)
        zfree(p)
        Rejects patterns like:
        free(p->field)
        decrRefCount(p->field)
        listEmpty(p->list)
        """
        if arg_index < 0 or arg_index >= len(func.arg_names):
            return False

        var = func.arg_names[arg_index]
        if not var:
            return False

        code = func.code

        # Common free-like calls (extend as needed; still generic)
        free_like = [
            "free", "cfree",
            "sdsfree",
            "decrRefCount",
            "zfree",
            "g_free",
            "kfree", "vfree", "xfree",
        ]

        for f in free_like:
            # Must be exactly f(var) with optional whitespace.
            # Crucially disallow f(var->...) or f(var....) by requiring ')' after var.
            if re.search(rf'\b{re.escape(f)}\s*\(\s*{re.escape(var)}\s*\)', code):
                return True

        return False


    def discover_memory_functions(
        self,
        functions: dict[str, FunctionInfo]
    ) -> tuple[set[str], dict[str, int]]:
        """Discover allocator and deallocator functions via call chain analysis."""
        self._init_transitive_analyzer(functions)

        if not self._transitive_analyzer:
            return set(), {}

        allocators = set()
        deallocators = {}

        for func_name in functions:
            is_alloc, chain = self._transitive_analyzer.is_transitive_allocator(func_name)
            if is_alloc and len(chain) > 1:
                allocators.add(func_name)

            is_dealloc, arg_idx, chain = self._transitive_analyzer.is_transitive_deallocator(func_name)
            if is_dealloc and len(chain) > 1:
                deallocators[func_name] = arg_idx

        return allocators, deallocators


# =============================================================================
# Warning Validator (Path Feasibility)
# =============================================================================

class WarningValidator:
    """Validate CodeQL warnings using Z3 path feasibility analysis."""

    def __init__(
        self,
        alloc_funcs: set[str] = None,
        free_funcs: set[str] = None
    ):
        self.base_alloc_funcs = alloc_funcs or {"malloc", "calloc", "realloc", "strdup"}
        self.base_free_funcs = free_funcs or {"free"}
        self._transitive_analyzer: Optional[TransitiveAnalyzer] = None

    def _init_transitive_analyzer(self, functions: dict[str, FunctionInfo]):
        if self._transitive_analyzer is None and functions:
            cg_builder = CallGraphBuilder()
            call_graph = cg_builder.build(functions)
            self._transitive_analyzer = TransitiveAnalyzer(call_graph, functions)

    def _get_all_alloc_funcs(self, functions: dict[str, FunctionInfo]) -> set[str]:
        alloc_funcs = set(self.base_alloc_funcs)
        alloc_funcs.update(TransitiveAnalyzer.BASE_ALLOCATORS)

        if self._transitive_analyzer:
            for func_name in functions:
                is_alloc, _ = self._transitive_analyzer.is_transitive_allocator(func_name)
                if is_alloc:
                    alloc_funcs.add(func_name)

        return alloc_funcs

    def _get_all_free_funcs(self, functions: dict[str, FunctionInfo]) -> set[str]:
        # Only include "strong" frees: ones that free the argument pointer itself.
        free_funcs = set(self.base_free_funcs)
        free_funcs.update(TransitiveAnalyzer.BASE_DEALLOCATORS)

        if self._transitive_analyzer:
            for func_name, finfo in functions.items():
                is_dealloc, freed_arg, _ = self._transitive_analyzer.is_transitive_deallocator(func_name)
                if not is_dealloc or freed_arg < 0:
                    continue

                # Strong criterion: function must directly free its freed_arg (free(arg), decrRefCount(arg), ...)
                if self._directly_frees_param_pointer(finfo, freed_arg):
                    free_funcs.add(func_name)

        return free_funcs
    
    def _directly_frees_param_pointer(self, func: FunctionInfo, arg_index: int) -> bool:
        """True iff this function directly deallocates argN itself, not argN->field / container members."""
        if arg_index < 0 or arg_index >= len(func.arg_names):
            return False
        var = func.arg_names[arg_index]
        if not var:
            return False

        code = func.code

        # Conservative, generic free-like primitives. Safe across projects.
        free_like = [
            "free", "cfree",
            "sdsfree",
            "decrRefCount",
            "zfree",
            "g_free",
            "kfree", "vfree", "xfree",
        ]

        # Match: f(var) exactly; reject f(var->x), f(&var), f(*var), etc.
        for f in free_like:
            if re.search(rf"\b{re.escape(f)}\s*\(\s*{re.escape(var)}\s*\)", code):
                return True

        return False



    def validate_warnings(
        self,
        warnings: list[Warning],
        functions: dict[str, FunctionInfo],
        hints: Optional[HintSet] = None,
    ) -> tuple[list[Warning], list[Warning]]:
        """Validate warnings, returning (confirmed, filtered)."""
        if not Z3_AVAILABLE:
            return warnings, []

        self._init_transitive_analyzer(functions)

        alloc_funcs = self._get_all_alloc_funcs(functions)
        free_funcs = self._get_all_free_funcs(functions)
        
        if hints:
            for func_name, arg_idx in hints.get_allocators():
                alloc_funcs.add(func_name)

            for func_name, arg_idx in hints.get_deallocators():
                finfo = functions.get(func_name)
                if not finfo:
                    continue
                # Only treat as FREE if it frees the argument pointer itself.
                free_funcs.add(func_name)

        confirmed = []
        filtered = []

        for warning in warnings:
            func = functions.get(warning.function_name)
            if not func:
                logger.warning("Validator: function not found for warning: %s (available=%d). Example keys: %s",
                            warning.function_name, len(functions), list(functions.keys())[:20])
                confirmed.append(warning)
                continue



            alloc_out_params: dict[str, int] = {}
            if hints:
                for func_name, arg_idx in hints.get_allocators():
                    if arg_idx is not None and arg_idx >= 0:
                        alloc_out_params[func_name] = arg_idx

            result = self._check_feasibility(warning, func, alloc_funcs, free_funcs, alloc_out_params)

            if result.is_feasible:
                confirmed.append(warning)
            else:
                filtered.append(warning)
                logger.debug(
                    f"Filtered {warning.function_name}:{warning.line_number}: {result.reason}"
                )

        return confirmed, filtered

    def _check_feasibility(
        self,
        warning: Warning,
        func: FunctionInfo,
        alloc_funcs: set[str],
        free_funcs: set[str],
        alloc_out_params: Optional[dict[str, int]] = None,
    ) -> PathFeasibilityResult:
        """Check if warning's bug path is feasible."""
        try:
            cfg_builder = CFGBuilder(alloc_funcs, free_funcs, alloc_out_params=alloc_out_params or {})
            cfg = cfg_builder.build(func.code)

            if warning.issue_type == MemoryIssueType.MEMORY_LEAK:
                return self._check_leak_feasibility(cfg, warning)
            elif warning.issue_type == MemoryIssueType.DOUBLE_FREE:
                return self._check_double_free_feasibility(cfg, warning)
            elif warning.issue_type == MemoryIssueType.USE_AFTER_FREE:
                return self._check_uaf_feasibility(cfg, warning)
            else:
                return PathFeasibilityResult(is_feasible=False, reason="unsupported bug type in validator")

        except Exception as e:
            logger.debug(f"Feasibility check failed: {e}")
            return PathFeasibilityResult(is_feasible=True, reason=f"analysis failed: {e}")




    def _mentions_var(self, expr: str, var: str) -> bool:
        if not var:
            return False
        return re.search(rf"\b{re.escape(var)}\b", expr) is not None

    def _lhs_is_escape_sink(self, lhs: str) -> bool:
        lhs = lhs.strip()
        # out-parameter store: *out = ...
        if re.match(r"^\*+\s*\w+", lhs):
            return True
        # struct/global-ish store: obj->field = ... or obj.field = ...
        if re.match(r"^\w+\s*(->|\.)\s*\w+", lhs):
            return True
        return False

    def _is_escape_event(self, code: str, alloc_var: str) -> bool:
        code = code.strip()

        # return alloc_var;
        m = _RETURN_RE.match(code)
        if m and self._mentions_var(m.group("expr"), alloc_var):
            return True

        # sink = alloc_var;
        m = _ASSIGN_RE.match(code)
        if m:
            lhs, rhs = m.group("lhs"), m.group("rhs")
            if self._lhs_is_escape_sink(lhs) and self._mentions_var(rhs, alloc_var):
                return True

        return False

    def _choose_alloc_var(self, cfg: dict[int, CFGNode], warning: Warning) -> str:
        # 0) If CodeQL stored an allocation_site string like "...:123" or "line 123"
        alloc_site_line = -1
        alloc_site = getattr(warning, "allocation_site", "") or ""
        if alloc_site:
            m = re.search(r"\b(\d+)\b", alloc_site)
            if m:
                alloc_site_line = int(m.group(1))

        # 1) Prefer CodeQL alloc-site line if present, otherwise fall back to warning line
        target_line = alloc_site_line if alloc_site_line > 0 else (getattr(warning, "line_number", -1) or -1)

        # 2) Collect allocation nodes with an identity
        allocs = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]
        candidates = []
        for n in allocs:
            ident = (n.variable or (n.obj or "")).strip()
            if ident:
                candidates.append((n.line, ident))

        if not candidates:
            return ""

        # 3) If we have a target line, pick the allocation whose line is closest (prefer <= target_line)
        if target_line > 0:
            # Prefer allocations at/above the target line (closest backward), else closest overall
            before = [(ln, ident) for (ln, ident) in candidates if ln <= target_line]
            if before:
                # choose nearest previous alloc
                ln, ident = max(before, key=lambda x: x[0])
                return ident

            # otherwise choose nearest alloc by absolute distance
            ln, ident = min(candidates, key=lambda x: abs(x[0] - target_line))
            return ident

        # 4) Fallback: first alloc identity
        return candidates[0][1]


    
    def _sanitize(self, s: str) -> str:
        """Sanitize an arbitrary expression to a Z3-safe symbol suffix."""
        if not s:
            return "empty"
        s = s.strip()
        s = re.sub(r"[^A-Za-z0-9_]", "_", s)
        # avoid super long names
        return s[:120]


    def _check_leak_feasibility(
        self,
        cfg: dict[int, CFGNode],
        warning: Warning,
    ) -> PathFeasibilityResult:
        solver = Solver()

        alloc_nodes = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]
        exit_nodes = [n for n in cfg.values() if n.node_type in (NodeType.EXIT, NodeType.RETURN)]
        if not alloc_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no allocation found")

        alloc_var = self._choose_alloc_var(cfg, warning)
        if not alloc_var:
            return PathFeasibilityResult(is_feasible=True, reason="unknown allocated variable (conservative)")

        reach   = {n.id: Bool(f"reach_{n.id}")   for n in cfg.values()}
        allocd  = {n.id: Bool(f"allocd_{n.id}")  for n in cfg.values()}   # NEW
        freed   = {n.id: Bool(f"freed_{n.id}")   for n in cfg.values()}
        escaped = {n.id: Bool(f"escaped_{n.id}") for n in cfg.values()}

        entry = next(n for n in cfg.values() if n.node_type == NodeType.ENTRY)
        solver.add(reach[entry.id] == True)
        solver.add(allocd[entry.id] == False)    # NEW
        solver.add(freed[entry.id] == False)
        solver.add(escaped[entry.id] == False)

        def alloc_matches_var(node: CFGNode) -> bool:
            # Return-value alloc: variable is LHS identifier extracted by CFGBuilder
            return node.node_type == NodeType.ALLOC and (node.variable or "") == alloc_var

        def free_matches_alloc(node: CFGNode) -> bool:
            if node.node_type != NodeType.FREE:
                return False

            # Strong match only: free(alloc_var)
            if node.variable and node.variable == alloc_var:
                return True

            o = (node.obj or "").strip()
            if o and o == alloc_var:
                return True

            # IMPORTANT: do NOT treat "mentions alloc_var" as freeing it.
            # free(p->x) does NOT free p.
            return False

        for node in cfg.values():
            if node.node_type == NodeType.ENTRY:
                continue

            preds = node.predecessors
            if preds:
                solver.add(reach[node.id] == Or(*[reach[p] for p in preds]))

                # allocated propagates existentially ("there exists a path where allocated happened")
                alloc_from_preds = Or(*[allocd[p] for p in preds])
                if alloc_matches_var(node):
                    solver.add(allocd[node.id] == Or(alloc_from_preds, reach[node.id]))
                else:
                    solver.add(allocd[node.id] == alloc_from_preds)

                # freed propagates, but only becomes true if we are already allocated on that path
                freed_from_preds = Or(*[freed[p] for p in preds])
                if free_matches_alloc(node):
                    solver.add(freed[node.id] == Or(freed_from_preds, And(reach[node.id], allocd[node.id])))
                else:
                    solver.add(freed[node.id] == freed_from_preds)

                # escaped propagates
                solver.add(escaped[node.id] == Or(*[escaped[p] for p in preds]))
            else:
                solver.add(reach[node.id] == False)
                solver.add(allocd[node.id] == False)
                solver.add(freed[node.id] == False)
                solver.add(escaped[node.id] == False)

            # Escape: out-parameter allocation (only if this is the tracked object)
            if node.node_type == NodeType.ALLOC and (node.variable or "").startswith("arg"):
                # This is not alloc_var usually; keep if you intentionally treat argX as escaped
                solver.add(escaped[node.id] == True)

            # Escape: return alloc_var (only makes sense after allocation)
            if node.node_type == NodeType.RETURN:
                if self._mentions_var(node.code, alloc_var):
                    solver.add(escaped[node.id] == And(reach[node.id], allocd[node.id]))

            # Escape: store alloc_var into sink (only makes sense if you actually emitted ASSIGN nodes)
            if node.node_type == NodeType.ASSIGN:
                if self._is_escape_event(node.code, alloc_var):
                    escaped[node.id] == True


        # Leak exists if reachable exit where:
        # - alloc happened on that path
        # - not freed
        # - not escaped
        leak_conditions = [
            And(reach[x.id], allocd[x.id], Not(freed[x.id]), Not(escaped[x.id]))
            for x in exit_nodes
        ]
        solver.add(Or(*leak_conditions))

        result = solver.check()
        if result == sat:
            return PathFeasibilityResult(is_feasible=True, reason=f"leak path exists for {alloc_var}")
        return PathFeasibilityResult(is_feasible=False, reason=f"all paths free/escape or never allocate {alloc_var}")




    def _check_double_free_feasibility(
        self,
            cfg: dict[int, CFGNode],
            warning: Warning,
        ) -> PathFeasibilityResult:
        """Check if double-free is feasible (per-object expression).

        Double-free is feasible iff there exists a reachable FREE(obj) such that
        obj has already been freed earlier on that reachable path.

        We use CFGNode.obj (full freed expression, e.g., acl_args[i], p->x, *p)
        instead of CFGNode.variable (base identifier), to avoid collapsing distinct
        objects into the same name.
        """
        solver = Solver()

        # Use full expression identity
        free_nodes = [n for n in cfg.values() if n.node_type == NodeType.FREE and (n.obj or "").strip()]
        if len(free_nodes) < 2:
            return PathFeasibilityResult(is_feasible=False, reason="less than 2 free calls")

        objs_freed = sorted({(n.obj or "").strip() for n in free_nodes if (n.obj or "").strip()})
        if not objs_freed:
            return PathFeasibilityResult(is_feasible=False, reason="no free object extracted")

        # Reachability boolean per node
        reach = {n.id: Bool(f"reach_{n.id}") for n in cfg.values()}
        entry = next(n for n in cfg.values() if n.node_type == NodeType.ENTRY)
        solver.add(reach[entry.id] == True)

        # Reachability propagation
        for node in cfg.values():
            if node.node_type == NodeType.ENTRY:
                continue
            preds = node.predecessors
            if preds:
                solver.add(reach[node.id] == Or(*[reach[p] for p in preds]))
            else:
                solver.add(reach[node.id] == False)

        # Per-object "freed already before this node" and "freed up to this node"
        freed_in: dict[str, dict[int, BoolRef]] = {}
        freed: dict[str, dict[int, BoolRef]] = {}

        for o in objs_freed:
            okey = self._sanitize(o)
            freed_in[o] = {n.id: Bool(f"freedIn_{okey}_{n.id}") for n in cfg.values()}
            freed[o] = {n.id: Bool(f"freed_{okey}_{n.id}") for n in cfg.values()}
            solver.add(freed_in[o][entry.id] == False)
            solver.add(freed[o][entry.id] == False)

        # Propagate + event updates
        for node in cfg.values():
            if node.node_type == NodeType.ENTRY:
                continue

            preds = node.predecessors

            for o in objs_freed:
                # freed_in[o][node] = OR freed[o][pred]
                if preds:
                    solver.add(freed_in[o][node.id] == Or(*[freed[o][p] for p in preds]))
                else:
                    solver.add(freed_in[o][node.id] == False)

                # If this node frees o on a reachable path, then freed[o] becomes true
                if node.node_type == NodeType.FREE and (node.obj or "").strip() == o:
                    # freed[o][node] = freed_in[o][node] OR reach[node]
                    solver.add(freed[o][node.id] == Or(freed_in[o][node.id], reach[node.id]))
                else:
                    solver.add(freed[o][node.id] == freed_in[o][node.id])

        # Double-free exists if there is a FREE(o) node where freed_in[o] is already true
        df_conditions = []
        for node in free_nodes:
            o = (node.obj or "").strip()
            df_conditions.append(And(reach[node.id], freed_in[o][node.id]))

        if not df_conditions:
            return PathFeasibilityResult(is_feasible=False, reason="no candidate double-free nodes")

        solver.add(Or(*df_conditions))

        result = solver.check()
        if result == sat:
            return PathFeasibilityResult(is_feasible=True, reason="double-free path exists")
        return PathFeasibilityResult(is_feasible=False, reason="no reachable double-free")
    
    def _check_uaf_feasibility(
        self,
        cfg: dict[int, CFGNode],
        warning: Warning,
    ) -> PathFeasibilityResult:
        """Check if use-after-free is feasible (per-object expression).

        UAF is feasible iff there exists a reachable DEREF(obj) such that
        obj has already been freed earlier on that reachable path.

        We use CFGNode.obj (full used/freed expression) rather than CFGNode.variable.
        """
        solver = Solver()

        free_nodes = [n for n in cfg.values() if n.node_type == NodeType.FREE and (n.obj or "").strip()]
        deref_nodes = [n for n in cfg.values() if n.node_type == NodeType.DEREF and (n.obj or "").strip()]

        if not free_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no free found")
        if not deref_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no deref/use found (CFGBuilder missing DEREF)")

        objs_of_interest = sorted(
            {(n.obj or "").strip() for n in free_nodes if (n.obj or "").strip()} |
            {(n.obj or "").strip() for n in deref_nodes if (n.obj or "").strip()}
        )
        if not objs_of_interest:
            return PathFeasibilityResult(is_feasible=False, reason="no objects extracted")

        reach = {n.id: Bool(f"reach_{n.id}") for n in cfg.values()}
        entry = next(n for n in cfg.values() if n.node_type == NodeType.ENTRY)
        solver.add(reach[entry.id] == True)

        # Reachability propagation
        for node in cfg.values():
            if node.node_type == NodeType.ENTRY:
                continue
            preds = node.predecessors
            if preds:
                solver.add(reach[node.id] == Or(*[reach[p] for p in preds]))
            else:
                solver.add(reach[node.id] == False)

        # Track freed status per object
        freed_in: dict[str, dict[int, BoolRef]] = {}
        freed: dict[str, dict[int, BoolRef]] = {}

        for o in objs_of_interest:
            okey = self._sanitize(o)
            freed_in[o] = {n.id: Bool(f"freedIn_{okey}_{n.id}") for n in cfg.values()}
            freed[o] = {n.id: Bool(f"freed_{okey}_{n.id}") for n in cfg.values()}
            solver.add(freed_in[o][entry.id] == False)
            solver.add(freed[o][entry.id] == False)

        for node in cfg.values():
            if node.node_type == NodeType.ENTRY:
                continue

            preds = node.predecessors

            for o in objs_of_interest:
                # freed_in[o][node] = OR freed[o][pred]
                if preds:
                    solver.add(freed_in[o][node.id] == Or(*[freed[o][p] for p in preds]))
                else:
                    solver.add(freed_in[o][node.id] == False)

                # FREE(o) makes freed[o] true
                if node.node_type == NodeType.FREE and (node.obj or "").strip() == o:
                    solver.add(freed[o][node.id] == Or(freed_in[o][node.id], reach[node.id]))
                else:
                    solver.add(freed[o][node.id] == freed_in[o][node.id])

        # UAF exists if DEREF(o) reachable AND o was freed before
        uaf_conditions = []
        for node in deref_nodes:
            o = (node.obj or "").strip()
            uaf_conditions.append(And(reach[node.id], freed_in[o][node.id]))

        solver.add(Or(*uaf_conditions))

        result = solver.check()
        if result == sat:
            return PathFeasibilityResult(is_feasible=True, reason="uaf path exists")
        return PathFeasibilityResult(is_feasible=False, reason="no reachable use-after-free")

