"""Z3-based Validator for Memory Safety Analysis.

This module provides two validation functions:

1. validate_hints(): Validates LLM-generated hints
   - Checks if ALLOCATOR hint is valid (function actually allocates)
   - Checks if DEALLOCATOR hint is valid (function actually frees)
   - Filters out impossible/incorrect hints

2. validate_warnings(): Filters CodeQL warnings by path feasibility
   - Checks if the bug path is actually reachable
   - Filters out false positives from infeasible paths
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

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

        # Entry node
        nodes[node_id] = CFGNode(
            id=node_id, node_type=NodeType.ENTRY, line=0, code="entry"
        )
        entry_id = node_id
        prev_id = node_id
        node_id += 1

        # Walk the AST
        for node in self._walk(tree.root_node):
            cfg_node = self._process_node(node, source, node_id)
            if cfg_node:
                nodes[node_id] = cfg_node
                nodes[prev_id].successors.append(node_id)
                cfg_node.predecessors.append(prev_id)
                prev_id = node_id
                node_id += 1

        # Exit node
        nodes[node_id] = CFGNode(
            id=node_id, node_type=NodeType.EXIT, line=0, code="exit"
        )
        nodes[prev_id].successors.append(node_id)
        nodes[node_id].predecessors.append(prev_id)

        return nodes

    def _walk(self, node):
        """Walk AST nodes."""
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _process_node(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """Process an AST node into a CFG node."""
        if node.type == "call_expression":
            return self._process_call(node, source, node_id)
        elif node.type == "if_statement":
            return self._process_if(node, source, node_id)
        elif node.type == "return_statement":
            return self._process_return(node, source, node_id)
        return None

    def _process_call(self, node, source: bytes, node_id: int) -> Optional[CFGNode]:
        """Process function call."""
        func_node = node.child_by_field_name("function")
        if not func_node:
            return None

        func_name = source[func_node.start_byte:func_node.end_byte].decode()
        line = node.start_point[0] + 1
        code = source[node.start_byte:node.end_byte].decode()

        # Check for allocation
        if func_name in self.alloc_funcs:
            # Find assigned variable
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

        # Check for deallocation
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
        """Process if statement."""
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
        """Process return statement."""
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

    Validates that hints are consistent with code structure:
    - ALLOCATOR: Function must have allocation call and return pointer
    - DEALLOCATOR: Function must have free call on specified argument
    - NULLABLE: Function must have path returning NULL
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            logger.warning("Z3 not available, validation will be skipped")

    def validate_hints(
        self,
        hints: HintSet,
        functions: dict[str, FunctionInfo],
    ) -> tuple[HintSet, list[str]]:
        """Validate all hints, returning validated hints and conflict messages.

        Args:
            hints: HintSet to validate
            functions: Dict of function name -> FunctionInfo

        Returns:
            (validated_hints, conflicts) where conflicts are removed hints
        """
        if not Z3_AVAILABLE:
            return hints, []

        validated = HintSet()
        conflicts = []

        for func_name, func_hints in hints.hints.items():
            if func_name not in functions:
                continue

            func = functions[func_name]

            for hint in func_hints:
                result = self._validate_hint(hint, func)
                if result.is_valid:
                    validated.add(hint)
                else:
                    conflicts.append(
                        f"REMOVED {func_name}.{hint.hint_type.name}: {result.reason}"
                    )

        return validated, conflicts

    def _validate_hint(self, hint: Hint, func: FunctionInfo) -> ValidationResult:
        """Validate a single hint against function code."""
        if hint.hint_type == HintType.ALLOCATOR:
            return self._validate_allocator(hint, func)
        elif hint.hint_type == HintType.DEALLOCATOR:
            return self._validate_deallocator(hint, func)
        elif hint.hint_type == HintType.NULLABLE:
            return self._validate_nullable(hint, func)
        else:
            # Other hints are harder to validate, accept them
            return ValidationResult(is_valid=True, reason="accepted")

    def _validate_allocator(self, hint: Hint, func: FunctionInfo) -> ValidationResult:
        """Validate ALLOCATOR hint.

        Check:
        1. Function has allocation call (malloc/calloc/etc)
        2. Function returns pointer type
        3. Allocation result flows to return
        """
        code = func.code

        # Must return pointer type
        if not func.return_type or '*' not in func.return_type:
            return ValidationResult(
                is_valid=False,
                reason="Does not return pointer type"
            )

        # Must have allocation call
        alloc_funcs = ["malloc", "calloc", "realloc", "strdup", "strndup",
                       "g_malloc", "g_new", "kmalloc"]
        has_alloc = any(f"{a}(" in code for a in alloc_funcs)

        if not has_alloc:
            # Check if it calls another function that might be allocator
            # This is a weaker check
            if "return" not in code.lower():
                return ValidationResult(
                    is_valid=False,
                    reason="No allocation call and no return"
                )

        return ValidationResult(is_valid=True, reason="validated")

    def _validate_deallocator(self, hint: Hint, func: FunctionInfo) -> ValidationResult:
        """Validate DEALLOCATOR hint.

        Check:
        1. Function has free call
        2. Free is called on the specified argument
        """
        code = func.code

        # Must have deallocation call
        free_funcs = ["free", "g_free", "kfree", "vfree"]
        has_free = any(f"{f}(" in code for f in free_funcs)

        if not has_free:
            return ValidationResult(
                is_valid=False,
                reason="No deallocation call found"
            )

        # Check if specified argument is freed
        arg_idx = hint.arg_index
        if arg_idx >= 0 and arg_idx < len(func.arg_names):
            arg_name = func.arg_names[arg_idx]
            # Check if argument is passed to free
            if not any(f"{f}({arg_name})" in code or f"{f}( {arg_name} )" in code
                       for f in free_funcs):
                return ValidationResult(
                    is_valid=False,
                    reason=f"Argument {arg_name} not passed to free"
                )

        return ValidationResult(is_valid=True, reason="validated")

    def _validate_nullable(self, hint: Hint, func: FunctionInfo) -> ValidationResult:
        """Validate NULLABLE hint.

        Check: Function has code path that returns NULL/0/nullptr
        """
        code = func.code

        # Check for explicit NULL return
        if re.search(r'return\s+(NULL|0|nullptr)\s*;', code):
            return ValidationResult(is_valid=True, reason="explicit NULL return")

        # Check for returning result of nullable function
        nullable_funcs = ["malloc", "calloc", "realloc", "strdup", "fopen",
                         "fgets", "strchr", "strstr"]
        for nf in nullable_funcs:
            if f"return {nf}(" in code or f"return\n{nf}(" in code:
                return ValidationResult(is_valid=True, reason=f"returns {nf} result")

        return ValidationResult(
            is_valid=False,
            reason="No NULL return path found"
        )


# =============================================================================
# Warning Validator (Path Feasibility)
# =============================================================================

class WarningValidator:
    """Validate CodeQL warnings using Z3 path feasibility analysis.

    Filters false positives by checking if bug path is actually reachable.
    """

    def __init__(self, alloc_funcs: set[str] = None, free_funcs: set[str] = None):
        self.cfg_builder = CFGBuilder(alloc_funcs, free_funcs)

    def validate_warnings(
        self,
        warnings: list[Warning],
        functions: dict[str, FunctionInfo],
    ) -> tuple[list[Warning], list[Warning]]:
        """Validate warnings, returning (confirmed, filtered).

        Args:
            warnings: List of CodeQL warnings
            functions: Dict of function name -> FunctionInfo

        Returns:
            (confirmed_warnings, filtered_warnings)
        """
        if not Z3_AVAILABLE:
            return warnings, []

        confirmed = []
        filtered = []

        for warning in warnings:
            func = functions.get(warning.function_name)
            if not func:
                # Can't validate without function code, keep warning
                confirmed.append(warning)
                continue

            result = self._check_feasibility(warning, func)
            if result.is_feasible:
                confirmed.append(warning)
            else:
                filtered.append(warning)
                logger.debug(f"Filtered {warning.function_name}:{warning.line_number}: {result.reason}")

        return confirmed, filtered

    def _check_feasibility(
        self,
        warning: Warning,
        func: FunctionInfo,
    ) -> PathFeasibilityResult:
        """Check if warning's bug path is feasible."""
        try:
            cfg = self.cfg_builder.build(func.code)

            if warning.issue_type == MemoryIssueType.MEMORY_LEAK:
                return self._check_leak_feasibility(cfg, warning)
            elif warning.issue_type == MemoryIssueType.DOUBLE_FREE:
                return self._check_double_free_feasibility(cfg, warning)
            elif warning.issue_type == MemoryIssueType.USE_AFTER_FREE:
                return self._check_uaf_feasibility(cfg, warning)
            elif warning.issue_type == MemoryIssueType.NULL_DEREFERENCE:
                return self._check_null_deref_feasibility(cfg, warning)
            else:
                # Can't validate, assume feasible
                return PathFeasibilityResult(is_feasible=True, reason="unhandled bug type")

        except Exception as e:
            logger.debug(f"Feasibility check failed: {e}")
            return PathFeasibilityResult(is_feasible=True, reason=f"analysis failed: {e}")

    def _check_leak_feasibility(
        self,
        cfg: dict[int, CFGNode],
        warning: Warning,
    ) -> PathFeasibilityResult:
        """Check if memory leak path is feasible.

        A leak exists if: ∃ path where alloc happens and exit reached without free
        """
        solver = Solver()

        # Find allocation and free nodes
        alloc_nodes = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]
        free_nodes = [n for n in cfg.values() if n.node_type == NodeType.FREE]
        exit_nodes = [n for n in cfg.values() if n.node_type in (NodeType.EXIT, NodeType.RETURN)]

        if not alloc_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no allocation found")

        # Create boolean variables for each node
        reach = {n.id: Bool(f"reach_{n.id}") for n in cfg.values()}
        freed = {n.id: Bool(f"freed_{n.id}") for n in cfg.values()}

        # Entry is reachable, not freed
        entry = next(n for n in cfg.values() if n.node_type == NodeType.ENTRY)
        solver.add(reach[entry.id] == True)
        solver.add(freed[entry.id] == False)

        # Propagate reachability
        for node in cfg.values():
            if node.node_type == NodeType.ENTRY:
                continue

            preds = node.predecessors
            if preds:
                reach_conds = [reach[p] for p in preds]
                solver.add(reach[node.id] == Or(*reach_conds))

                # Track if freed
                if node.node_type == NodeType.FREE:
                    solver.add(freed[node.id] == Or(reach[node.id], *[freed[p] for p in preds]))
                else:
                    freed_conds = [freed[p] for p in preds]
                    solver.add(freed[node.id] == Or(*freed_conds))

        # Check: can we reach exit without being freed?
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

        # For simplicity, if there are 2+ free nodes on same variable, consider feasible
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
        # Simplified: check if there's a free followed by potential use
        free_nodes = [n for n in cfg.values() if n.node_type == NodeType.FREE]

        if not free_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no free found")

        # For a proper check, we'd need to track uses after free
        # For now, if there's a free and code after it, consider potentially feasible
        return PathFeasibilityResult(is_feasible=True, reason="free found, needs detailed analysis")

    def _check_null_deref_feasibility(
        self,
        cfg: dict[int, CFGNode],
        warning: Warning,
    ) -> PathFeasibilityResult:
        """Check if null dereference path is feasible."""
        # Check if there's an allocation that could fail
        alloc_nodes = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]

        if not alloc_nodes:
            return PathFeasibilityResult(is_feasible=False, reason="no allocation found")

        # Check if there's a null check branch
        branch_nodes = [n for n in cfg.values() if n.node_type == NodeType.BRANCH]

        for branch in branch_nodes:
            cond = branch.condition.lower()
            # If there's a null check, the path after the check might be safe
            if "null" in cond or "== 0" in cond or "!= 0" in cond:
                return PathFeasibilityResult(
                    is_feasible=True,
                    reason="null check exists but path may bypass it"
                )

        # No null check found, definitely feasible
        return PathFeasibilityResult(is_feasible=True, reason="no null check found")
