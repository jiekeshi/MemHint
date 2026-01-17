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
    """Control Flow Graph node."""
    id: int
    node_type: NodeType
    line: int
    code: str
    variable: str = ""
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

    def build(self, functions: dict[str, FunctionInfo]) -> CallGraph:
        """Build call graph from all functions."""
        cg = CallGraph()

        for func_name, func_info in functions.items():
            self._extract_calls(func_name, func_info, cg)

        return cg

    def _extract_calls(self, func_name: str, func_info: FunctionInfo, cg: CallGraph):
        """Extract all call sites from a function."""
        tree = self.parser.parse(func_info.code.encode())
        source = func_info.code.encode()

        for node in self._walk(tree.root_node):
            if node.type == "call_expression":
                callsite = self._process_call(func_name, func_info, node, source)
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
        caller_info: FunctionInfo,
        node,
        source: bytes
    ) -> Optional[CallSite]:
        """Process a call expression into a CallSite."""
        func_node = node.child_by_field_name("function")
        if not func_node:
            return None

        callee = source[func_node.start_byte:func_node.end_byte].decode()
        line = node.start_point[0] + 1

        # Check if indirect call (function pointer)
        is_indirect = func_node.type != "identifier"

        arg_mapping = {}
        arg_expressions = {}

        args_node = node.child_by_field_name("arguments")
        if args_node:
            arg_idx = 0
            for child in args_node.children:
                if child.type in ("identifier", "pointer_expression", "cast_expression",
                                  "number_literal", "string_literal", "call_expression"):
                    expr = source[child.start_byte:child.end_byte].decode()
                    arg_expressions[arg_idx] = expr

                    if child.type == "identifier":
                        param_name = expr
                        if param_name in caller_info.arg_names:
                            caller_arg_idx = caller_info.arg_names.index(param_name)
                            arg_mapping[arg_idx] = caller_arg_idx
                        else:
                            arg_mapping[arg_idx] = -1
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

                for callee_arg, our_arg in callsite.arg_mapping.items():
                    if our_arg >= 0:
                        return (True, our_arg, result_chain)

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
    ):
        self.parser = Parser(C_LANGUAGE)
        self.alloc_funcs = alloc_funcs or {"malloc", "calloc", "realloc", "strdup"}
        self.free_funcs = free_funcs or {"free"}

    def build(self, code: str) -> dict[int, CFGNode]:
        """Build CFG from function code."""
        tree = self.parser.parse(code.encode())
        source = code.encode()

        nodes: dict[int, CFGNode] = {}
        node_id = 0

        nodes[node_id] = CFGNode(
            id=node_id, node_type=NodeType.ENTRY, line=0, code="entry"
        )
        prev_id = node_id
        node_id += 1

        for node in self._walk(tree.root_node):
            cfg_node = self._process_node(node, source, node_id)
            if cfg_node:
                nodes[node_id] = cfg_node
                nodes[prev_id].successors.append(node_id)
                cfg_node.predecessors.append(prev_id)
                prev_id = node_id
                node_id += 1

        nodes[node_id] = CFGNode(
            id=node_id, node_type=NodeType.EXIT, line=0, code="exit"
        )
        nodes[prev_id].successors.append(node_id)
        nodes[node_id].predecessors.append(prev_id)

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
        return None

    def _process_call(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        func_node = node.child_by_field_name("function")
        if not func_node:
            return None

        func_name = source[func_node.start_byte:func_node.end_byte].decode()
        line = node.start_point[0] + 1
        code = source[node.start_byte:node.end_byte].decode()

        if func_name in self.alloc_funcs:
            parent = node.parent
            var = ""
            if parent and parent.type in ("assignment_expression", "init_declarator"):
                for child in parent.children:
                    if child.type == "identifier":
                        var = source[child.start_byte:child.end_byte].decode()
                        break

            return CFGNode(
                id=node_id, node_type=NodeType.ALLOC,
                line=line, code=code, variable=var
            )

        if func_name in self.free_funcs:
            args = node.child_by_field_name("arguments")
            var = ""
            if args:
                for child in args.children:
                    if child.type == "identifier":
                        var = source[child.start_byte:child.end_byte].decode()
                        break

            return CFGNode(
                id=node_id, node_type=NodeType.FREE,
                line=line, code=code, variable=var
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

        # Use enhanced transitive analyzer
        is_dealloc, freed_arg, chain = self._transitive_analyzer.is_transitive_deallocator(func_name)

        if is_dealloc:
            # Verify argument index if specified
            if hint.arg_index >= 0 and freed_arg >= 0 and hint.arg_index != freed_arg:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Hint says arg {hint.arg_index} freed, but analysis shows arg {freed_arg}"
                )

            chain_str = ' -> '.join(chain) if chain else func_name
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
        free_funcs = set(self.base_free_funcs)
        free_funcs.update(TransitiveAnalyzer.BASE_DEALLOCATORS)

        if self._transitive_analyzer:
            for func_name in functions:
                is_dealloc, _, _ = self._transitive_analyzer.is_transitive_deallocator(func_name)
                if is_dealloc:
                    free_funcs.add(func_name)

        return free_funcs

    def validate_warnings(
        self,
        warnings: list[Warning],
        functions: dict[str, FunctionInfo],
    ) -> tuple[list[Warning], list[Warning]]:
        """Validate warnings, returning (confirmed, filtered)."""
        if not Z3_AVAILABLE:
            return warnings, []

        self._init_transitive_analyzer(functions)

        alloc_funcs = self._get_all_alloc_funcs(functions)
        free_funcs = self._get_all_free_funcs(functions)

        confirmed = []
        filtered = []

        for warning in warnings:
            func = functions.get(warning.function_name)
            if not func:
                confirmed.append(warning)
                continue

            result = self._check_feasibility(warning, func, alloc_funcs, free_funcs)
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
    ) -> PathFeasibilityResult:
        """Check if warning's bug path is feasible."""
        try:
            cfg_builder = CFGBuilder(alloc_funcs, free_funcs)
            cfg = cfg_builder.build(func.code)

            if warning.issue_type == MemoryIssueType.MEMORY_LEAK:
                return self._check_leak_feasibility(cfg, warning)
            elif warning.issue_type == MemoryIssueType.DOUBLE_FREE:
                return self._check_double_free_feasibility(cfg, warning)
            elif warning.issue_type == MemoryIssueType.USE_AFTER_FREE:
                return self._check_uaf_feasibility(cfg, warning)
            else:
                return PathFeasibilityResult(is_feasible=True, reason="unhandled bug type")

        except Exception as e:
            logger.debug(f"Feasibility check failed: {e}")
            return PathFeasibilityResult(is_feasible=True, reason=f"analysis failed: {e}")

    def _check_leak_feasibility(
        self,
        cfg: dict[int, CFGNode],
        warning: Warning,
    ) -> PathFeasibilityResult:
        """Check if memory leak path is feasible."""
        solver = Solver()

        alloc_nodes = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]
        free_nodes = [n for n in cfg.values() if n.node_type == NodeType.FREE]
        exit_nodes = [n for n in cfg.values() if n.node_type in (NodeType.EXIT, NodeType.RETURN)]

        if not alloc_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no allocation found")

        reach = {n.id: Bool(f"reach_{n.id}") for n in cfg.values()}
        freed = {n.id: Bool(f"freed_{n.id}") for n in cfg.values()}

        entry = next(n for n in cfg.values() if n.node_type == NodeType.ENTRY)
        solver.add(reach[entry.id] == True)
        solver.add(freed[entry.id] == False)

        for node in cfg.values():
            if node.node_type == NodeType.ENTRY:
                continue

            preds = node.predecessors
            if preds:
                reach_conds = [reach[p] for p in preds]
                solver.add(reach[node.id] == Or(*reach_conds))

                if node.node_type == NodeType.FREE:
                    solver.add(freed[node.id] == Or(reach[node.id], *[freed[p] for p in preds]))
                else:
                    freed_conds = [freed[p] for p in preds]
                    solver.add(freed[node.id] == Or(*freed_conds))

        leak_conditions = []
        for exit_node in exit_nodes:
            leak_conditions.append(And(reach[exit_node.id], Not(freed[exit_node.id])))

        if leak_conditions:
            solver.add(Or(*leak_conditions))

        result = solver.check()

        if result == sat:
            return PathFeasibilityResult(is_feasible=True, reason="leak path exists")
        else:
            return PathFeasibilityResult(is_feasible=False, reason="all paths free memory")

    def _check_double_free_feasibility(
        self,
        cfg: dict[int, CFGNode],
        warning: Warning,
    ) -> PathFeasibilityResult:
        """Check if double-free path is feasible."""
        free_nodes = [n for n in cfg.values() if n.node_type == NodeType.FREE]

        if len(free_nodes) < 2:
            return PathFeasibilityResult(is_feasible=False, reason="less than 2 free calls")

        vars_freed = [n.variable for n in free_nodes]
        for var in vars_freed:
            if var and vars_freed.count(var) >= 2:
                return PathFeasibilityResult(is_feasible=True, reason=f"multiple frees of {var}")

        return PathFeasibilityResult(is_feasible=False, reason="no duplicate frees found")

    def _check_uaf_feasibility(
        self,
        cfg: dict[int, CFGNode],
        warning: Warning,
    ) -> PathFeasibilityResult:
        """Check if use-after-free path is feasible."""
        free_nodes = [n for n in cfg.values() if n.node_type == NodeType.FREE]

        if not free_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no free found")

        return PathFeasibilityResult(is_feasible=True, reason="free found, needs detailed analysis")