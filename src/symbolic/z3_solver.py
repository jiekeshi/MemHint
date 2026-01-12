"""Z3-based constraint solver for memory safety analysis.

This module extracts path conditions from C code using tree-sitter AST,
builds Z3 constraints, and solves them to determine:
1. Whether allocation-without-free paths are feasible
2. Whether annotations are consistent with code structure
3. Whether memory leaks are reachable under certain conditions

Key concepts:
- Path Condition: Boolean constraint that must be true for a path to execute
- Allocation Point: Where memory is allocated (malloc, custom allocator)
- Free Point: Where memory is freed
- Leak Path: A path from allocation to function exit without free
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

try:
    from z3 import (
        Solver, Bool, Int, And, Or, Not, Implies, If,
        sat, unsat, unknown, BoolRef, IntVal
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logging.warning("Z3 not available - install with: pip install z3-solver")

import tree_sitter_c as tsc
from tree_sitter import Language, Parser

from src.core.models import (
    Annotation, AnnotationType, AnnotationSet, FunctionInfo,
    MemoryIssueType
)

logger = logging.getLogger(__name__)

C_LANGUAGE = Language(tsc.language())


class NodeType(Enum):
    """Types of CFG nodes."""
    ENTRY = auto()
    EXIT = auto()
    ALLOC = auto()
    FREE = auto()
    RETURN = auto()
    BRANCH = auto()
    ASSIGN = auto()
    CALL = auto()


@dataclass
class CFGNode:
    """Control Flow Graph node."""
    id: int
    node_type: NodeType
    line: int
    code: str
    variable: str = ""  # Variable being allocated/freed/assigned
    condition: str = ""  # For branch nodes
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)


@dataclass
class PathCondition:
    """A condition extracted from a branch."""
    variable: str
    operator: str  # ==, !=, <, >, <=, >=
    value: str  # NULL, 0, or numeric
    is_null_check: bool = False


@dataclass
class ReachabilityResult:
    """Result of Z3 reachability analysis."""
    reachable: bool
    reason: str
    model: dict = None  # Variable assignments that make it reachable
    conflicting_conditions: list[str] = field(default_factory=list)


class CFGBuilder:
    """Build Control Flow Graph from tree-sitter AST."""

    def __init__(self, alloc_funcs: set[str] = None, free_funcs: set[str] = None):
        self.parser = Parser(C_LANGUAGE)
        self.alloc_funcs = alloc_funcs or {"malloc", "calloc", "realloc", "strdup", "strndup"}
        self.free_funcs = free_funcs or {"free", "g_free", "cfree"}
        self.node_id = 0
        self.nodes: dict[int, CFGNode] = {}

    def build(self, code: str) -> dict[int, CFGNode]:
        """Build CFG from C function code."""
        self.nodes = {}
        self.node_id = 0

        tree = self.parser.parse(code.encode())
        root = tree.root_node

        # Find function definition
        func_node = self._find_function(root)
        if not func_node:
            return {}

        # Create entry node
        entry = self._create_node(NodeType.ENTRY, 0, "entry")
        exit_node = self._create_node(NodeType.EXIT, 0, "exit")

        # Process function body
        body = None
        for child in func_node.children:
            if child.type == "compound_statement":
                body = child
                break

        if body:
            last_nodes = self._process_block(body, [entry.id], code.encode())
            # Connect last nodes to exit
            for node_id in last_nodes:
                self._add_edge(node_id, exit_node.id)

        return self.nodes

    def _find_function(self, root):
        """Find function definition in AST."""
        for node in self._walk(root):
            if node.type == "function_definition":
                return node
        return None

    def _walk(self, node):
        """Walk AST depth-first."""
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _create_node(self, node_type: NodeType, line: int, code: str, **kwargs) -> CFGNode:
        """Create a new CFG node."""
        node = CFGNode(
            id=self.node_id,
            node_type=node_type,
            line=line,
            code=code,
            **kwargs
        )
        self.nodes[self.node_id] = node
        self.node_id += 1
        return node

    def _add_edge(self, from_id: int, to_id: int):
        """Add edge between nodes."""
        if from_id in self.nodes and to_id in self.nodes:
            if to_id not in self.nodes[from_id].successors:
                self.nodes[from_id].successors.append(to_id)
            if from_id not in self.nodes[to_id].predecessors:
                self.nodes[to_id].predecessors.append(from_id)

    def _process_block(self, block, entry_nodes: list[int], source: bytes) -> list[int]:
        """Process a compound statement block. Returns list of exit node IDs."""
        current_nodes = entry_nodes

        for child in block.children:
            if child.type in ("{", "}"):
                continue
            current_nodes = self._process_statement(child, current_nodes, source)

        return current_nodes

    def _process_statement(self, stmt, entry_nodes: list[int], source: bytes) -> list[int]:
        """Process a statement. Returns list of exit node IDs."""
        stmt_type = stmt.type
        line = stmt.start_point[0] + 1
        code = source[stmt.start_byte:stmt.end_byte].decode()

        if stmt_type == "if_statement":
            return self._process_if(stmt, entry_nodes, source)

        elif stmt_type == "while_statement":
            return self._process_while(stmt, entry_nodes, source)

        elif stmt_type == "for_statement":
            return self._process_for(stmt, entry_nodes, source)

        elif stmt_type == "return_statement":
            node = self._create_node(NodeType.RETURN, line, code)
            for e in entry_nodes:
                self._add_edge(e, node.id)
            # Return statements don't have successors (handled specially)
            return []

        elif stmt_type == "expression_statement":
            return self._process_expression(stmt, entry_nodes, source)

        elif stmt_type == "declaration":
            return self._process_declaration(stmt, entry_nodes, source)

        elif stmt_type == "compound_statement":
            return self._process_block(stmt, entry_nodes, source)

        else:
            # Unknown statement - pass through
            return entry_nodes

    def _process_if(self, if_stmt, entry_nodes: list[int], source: bytes) -> list[int]:
        """Process if statement, creating branch node."""
        line = if_stmt.start_point[0] + 1

        # Extract condition
        condition = ""
        for child in if_stmt.children:
            if child.type == "parenthesized_expression":
                condition = source[child.start_byte:child.end_byte].decode()
                condition = condition[1:-1]  # Remove parentheses
                break

        branch_node = self._create_node(
            NodeType.BRANCH, line, f"if ({condition})", condition=condition
        )
        for e in entry_nodes:
            self._add_edge(e, branch_node.id)

        # Process then branch
        then_exits = []
        else_exits = []

        children = list(if_stmt.children)
        then_block = None
        else_block = None

        for i, child in enumerate(children):
            if child.type == "compound_statement" or (child.type not in ("if", "else", "parenthesized_expression", "(", ")")):
                if then_block is None and child.type not in ("if", "(", ")"):
                    then_block = child
                elif else_block is None and child.type not in ("else",):
                    else_block = child
            if child.type == "else":
                # Next non-trivial child is else block
                for j in range(i + 1, len(children)):
                    if children[j].type not in ("else",):
                        else_block = children[j]
                        break

        if then_block:
            then_exits = self._process_statement(then_block, [branch_node.id], source)

        if else_block:
            else_exits = self._process_statement(else_block, [branch_node.id], source)
        else:
            # No else - branch node itself is an exit for the false path
            else_exits = [branch_node.id]

        return then_exits + else_exits

    def _process_while(self, while_stmt, entry_nodes: list[int], source: bytes) -> list[int]:
        """Process while loop."""
        line = while_stmt.start_point[0] + 1

        condition = ""
        for child in while_stmt.children:
            if child.type == "parenthesized_expression":
                condition = source[child.start_byte:child.end_byte].decode()
                condition = condition[1:-1]
                break

        branch_node = self._create_node(
            NodeType.BRANCH, line, f"while ({condition})", condition=condition
        )
        for e in entry_nodes:
            self._add_edge(e, branch_node.id)

        # Process body
        for child in while_stmt.children:
            if child.type == "compound_statement":
                body_exits = self._process_block(child, [branch_node.id], source)
                # Loop back
                for exit_id in body_exits:
                    self._add_edge(exit_id, branch_node.id)
                break

        # Exit when condition is false
        return [branch_node.id]

    def _process_for(self, for_stmt, entry_nodes: list[int], source: bytes) -> list[int]:
        """Process for loop (simplified - treat like while)."""
        line = for_stmt.start_point[0] + 1
        code = source[for_stmt.start_byte:for_stmt.end_byte].decode().split("{")[0]

        branch_node = self._create_node(NodeType.BRANCH, line, code.strip())
        for e in entry_nodes:
            self._add_edge(e, branch_node.id)

        for child in for_stmt.children:
            if child.type == "compound_statement":
                body_exits = self._process_block(child, [branch_node.id], source)
                for exit_id in body_exits:
                    self._add_edge(exit_id, branch_node.id)
                break

        return [branch_node.id]

    def _process_expression(self, expr_stmt, entry_nodes: list[int], source: bytes) -> list[int]:
        """Process expression statement (assignments, function calls)."""
        line = expr_stmt.start_point[0] + 1
        code = source[expr_stmt.start_byte:expr_stmt.end_byte].decode().strip()

        # Check for function calls
        for child in self._walk(expr_stmt):
            if child.type == "call_expression":
                func_name = ""
                func_node = child.child_by_field_name("function")
                if func_node:
                    func_name = source[func_node.start_byte:func_node.end_byte].decode()

                # Check for free calls
                if func_name in self.free_funcs:
                    # Extract argument (variable being freed)
                    args = child.child_by_field_name("arguments")
                    var = ""
                    if args:
                        for arg in args.children:
                            if arg.type == "identifier":
                                var = source[arg.start_byte:arg.end_byte].decode()
                                break

                    node = self._create_node(NodeType.FREE, line, code, variable=var)
                    for e in entry_nodes:
                        self._add_edge(e, node.id)
                    return [node.id]

        # Check for assignment with allocation
        if "=" in code:
            for child in self._walk(expr_stmt):
                if child.type == "call_expression":
                    func_node = child.child_by_field_name("function")
                    if func_node:
                        func_name = source[func_node.start_byte:func_node.end_byte].decode()
                        if func_name in self.alloc_funcs:
                            # Find the variable being assigned
                            var = ""
                            match = re.match(r'(\w+)\s*=', code)
                            if match:
                                var = match.group(1)

                            node = self._create_node(NodeType.ALLOC, line, code, variable=var)
                            for e in entry_nodes:
                                self._add_edge(e, node.id)
                            return [node.id]

        # Generic statement
        node = self._create_node(NodeType.ASSIGN, line, code)
        for e in entry_nodes:
            self._add_edge(e, node.id)
        return [node.id]

    def _process_declaration(self, decl, entry_nodes: list[int], source: bytes) -> list[int]:
        """Process variable declaration."""
        line = decl.start_point[0] + 1
        code = source[decl.start_byte:decl.end_byte].decode().strip()

        # Check if it's an allocation
        for child in self._walk(decl):
            if child.type == "call_expression":
                func_node = child.child_by_field_name("function")
                if func_node:
                    func_name = source[func_node.start_byte:func_node.end_byte].decode()
                    if func_name in self.alloc_funcs:
                        # Find variable name
                        var = ""
                        for d in self._walk(decl):
                            if d.type == "identifier":
                                var = source[d.start_byte:d.end_byte].decode()
                                break

                        node = self._create_node(NodeType.ALLOC, line, code, variable=var)
                        for e in entry_nodes:
                            self._add_edge(e, node.id)
                        return [node.id]

        # Regular declaration
        node = self._create_node(NodeType.ASSIGN, line, code)
        for e in entry_nodes:
            self._add_edge(e, node.id)
        return [node.id]


class Z3PathAnalyzer:
    """Analyze paths using Z3 constraint solving."""

    def __init__(self):
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 is required but not installed")

    def check_leak_path(
        self,
        cfg: dict[int, CFGNode],
        alloc_var: str
    ) -> ReachabilityResult:
        """Check if there's a feasible path from allocation to exit without free.

        Uses Z3 to check if path conditions allow reaching exit without freeing.

        Key insight: A leak exists if we can reach a RETURN node where:
        1. The allocation SUCCEEDED (not NULL)
        2. free() has NOT been called on that path

        We detect NULL checks by looking for conditions like "var == NULL" or "!var"
        and ensuring we only consider paths where allocation succeeded.
        """
        if not cfg:
            return ReachabilityResult(True, "Empty CFG")

        solver = Solver()

        # Create Z3 variables for each branch condition
        branch_vars = {}
        null_check_branches = {}  # Track which branches are NULL checks for our var

        for node_id, node in cfg.items():
            if node.node_type == NodeType.BRANCH and node.condition:
                var_name = f"branch_{node_id}"
                branch_vars[node_id] = Bool(var_name)

                # Check if this is a NULL check for our variable
                cond = node.condition.strip()
                is_null_check = False
                null_is_true_branch = True  # Does true branch mean var is NULL?

                if f'{alloc_var} == NULL' in cond or f'{alloc_var} == 0' in cond:
                    is_null_check = True
                    null_is_true_branch = True
                elif f'{alloc_var} != NULL' in cond or f'{alloc_var} != 0' in cond:
                    is_null_check = True
                    null_is_true_branch = False
                elif f'!{alloc_var}' in cond or cond == f'!{alloc_var}':
                    is_null_check = True
                    null_is_true_branch = True
                elif cond == alloc_var:  # if (ptr) - true means NOT NULL
                    is_null_check = True
                    null_is_true_branch = False

                if is_null_check:
                    null_check_branches[node_id] = null_is_true_branch

        # Find special nodes
        alloc_nodes = [n for n in cfg.values()
                       if n.node_type == NodeType.ALLOC and n.variable == alloc_var]
        free_nodes = [n for n in cfg.values()
                      if n.node_type == NodeType.FREE and n.variable == alloc_var]
        return_nodes = [n for n in cfg.values() if n.node_type == NodeType.RETURN]
        exit_nodes = [n for n in cfg.values() if n.node_type == NodeType.EXIT]

        if not alloc_nodes:
            return ReachabilityResult(False, f"No allocation of {alloc_var} found")

        all_exits = return_nodes + exit_nodes
        if not all_exits:
            return ReachabilityResult(False, "No exit point found")

        reach = {n_id: Bool(f"reach_{n_id}") for n_id in cfg}
        free_called = {n_id: Bool(f"free_{n_id}") for n_id in cfg}

        for node_id, node in cfg.items():
            if node.node_type == NodeType.ENTRY:
                solver.add(reach[node_id] == False)
                solver.add(free_called[node_id] == False)

        for alloc in alloc_nodes:
            solver.add(reach[alloc.id] == True)
            solver.add(free_called[alloc.id] == False)

        for node_id, node in cfg.items():
            if node.node_type == NodeType.ENTRY:
                continue
            if node.node_type == NodeType.ALLOC and node.variable == alloc_var:
                continue

            preds = node.predecessors
            if not preds:
                solver.add(reach[node_id] == False)
                solver.add(free_called[node_id] == False)
                continue

            # Reachability constraints
            reach_conds = []
            for pred_id in preds:
                pred = cfg[pred_id]

                if pred.node_type == NodeType.BRANCH and pred_id in branch_vars:
                    is_true_branch = (len(pred.successors) > 0 and
                                      pred.successors[0] == node_id)

                    # Special handling for NULL checks
                    if pred_id in null_check_branches:
                        null_is_true = null_check_branches[pred_id]
                        # If this branch goes to "NULL" path, allocation failed - not reachable
                        # We model this by saying: if we take the NULL path, reach is False
                        if (is_true_branch and null_is_true) or (not is_true_branch and not null_is_true):
                            # This path means var == NULL, allocation failed
                            # Don't add this as a reachable path?
                            continue

                    if is_true_branch:
                        reach_conds.append(And(reach[pred_id], branch_vars[pred_id]))
                    else:
                        reach_conds.append(And(reach[pred_id], Not(branch_vars[pred_id])))
                else:
                    reach_conds.append(reach[pred_id])

            if reach_conds:
                solver.add(reach[node_id] == Or(*reach_conds))
            else:
                solver.add(reach[node_id] == False)

            # Free tracking
            if node.node_type == NodeType.FREE and node.variable == alloc_var:
                # This node is a FREE - mark as freed
                # But only if this node is actually reachable!
                solver.add(Implies(reach[node_id], free_called[node_id] == True))
                solver.add(Implies(Not(reach[node_id]), free_called[node_id] == False))
            else:
                # Inherit free status from predecessors
                # Key: free status should consider path conditions
                free_conds = []
                for pred_id in preds:
                    pred = cfg[pred_id]

                    if pred.node_type == NodeType.BRANCH and pred_id in branch_vars:
                        is_true_branch = (len(pred.successors) > 0 and
                                          pred.successors[0] == node_id)
                        if is_true_branch:
                            # Only inherit if we actually took this branch
                            free_conds.append(And(branch_vars[pred_id], free_called[pred_id]))
                        else:
                            free_conds.append(And(Not(branch_vars[pred_id]), free_called[pred_id]))
                    elif pred.node_type == NodeType.FREE and pred.variable == alloc_var:
                        # Coming from a FREE node - check if FREE was reachable
                        free_conds.append(reach[pred_id])
                    else:
                        free_conds.append(free_called[pred_id])

                if free_conds:
                    solver.add(free_called[node_id] == Or(*free_conds))
                else:
                    solver.add(free_called[node_id] == False)

        # Check for leak: exit reachable AND free not called
        leak_conditions = []
        for exit_node in all_exits:
            leak_conditions.append(
                And(reach[exit_node.id], Not(free_called[exit_node.id]))
            )

        if not leak_conditions:
            return ReachabilityResult(False, "No exit conditions to check")

        solver.add(Or(*leak_conditions))

        # Solve
        result = solver.check()

        if result == sat:
            model = solver.model()
            branch_assignments = {}
            for node_id, var in branch_vars.items():
                val = model.evaluate(var, model_completion=True)
                branch_assignments[cfg[node_id].condition] = str(val)

            return ReachabilityResult(
                reachable=True,
                reason=f"Leak path exists for variable '{alloc_var}'",
                model=branch_assignments
            )
        elif result == unsat:
            return ReachabilityResult(
                reachable=False,
                reason=f"All paths from successful allocation of '{alloc_var}' call free()"
            )
        else:
            return ReachabilityResult(
                reachable=True,
                reason=f"Could not determine (Z3 returned unknown) - assuming leak possible"
            )

    def check_double_free(
        self,
        cfg: dict[int, CFGNode],
        var: str
    ) -> ReachabilityResult:
        """Check if double-free is possible for a variable.

        Double-free occurs when:
        1. free(p) is called
        2. free(p) is called again on the same path

        We track: can we reach two FREE nodes sequentially?
        """
        if not cfg:
            return ReachabilityResult(False, "Empty CFG")

        free_nodes = [n for n in cfg.values()
                      if n.node_type == NodeType.FREE and n.variable == var]

        if len(free_nodes) < 2:
            return ReachabilityResult(False, f"Less than 2 free() calls for {var}")

        solver = Solver()

        # Create branch variables
        branch_vars = {}
        for node_id, node in cfg.items():
            if node.node_type == NodeType.BRANCH:
                branch_vars[node_id] = Bool(f"branch_{node_id}")

        # Track reachability from entry
        reach = {n_id: Bool(f"reach_{n_id}") for n_id in cfg}

        # Track: has first free been called?
        first_free_called = {n_id: Bool(f"ff_{n_id}") for n_id in cfg}

        # Entry is reachable, no free called yet
        for node_id, node in cfg.items():
            if node.node_type == NodeType.ENTRY:
                solver.add(reach[node_id] == True)
                solver.add(first_free_called[node_id] == False)

        # Process nodes
        for node_id, node in cfg.items():
            if node.node_type == NodeType.ENTRY:
                continue

            preds = node.predecessors
            if not preds:
                solver.add(reach[node_id] == False)
                solver.add(first_free_called[node_id] == False)
                continue

            # Reachability
            reach_conds = []
            for pred_id in preds:
                pred = cfg[pred_id]
                if pred.node_type == NodeType.BRANCH and pred_id in branch_vars:
                    is_true = pred.successors and pred.successors[0] == node_id
                    if is_true:
                        reach_conds.append(And(reach[pred_id], branch_vars[pred_id]))
                    else:
                        reach_conds.append(And(reach[pred_id], Not(branch_vars[pred_id])))
                else:
                    reach_conds.append(reach[pred_id])

            if reach_conds:
                solver.add(reach[node_id] == Or(*reach_conds))

            # First free tracking
            if node.node_type == NodeType.FREE and node.variable == var:
                # At a FREE node: first_free becomes True
                solver.add(Implies(reach[node_id], first_free_called[node_id] == True))
                solver.add(Implies(Not(reach[node_id]), first_free_called[node_id] == False))
            else:
                # Inherit from predecessors
                ff_conds = []
                for pred_id in preds:
                    pred = cfg[pred_id]
                    if pred.node_type == NodeType.FREE and pred.variable == var:
                        ff_conds.append(reach[pred_id])
                    elif pred.node_type == NodeType.BRANCH and pred_id in branch_vars:
                        is_true = pred.successors and pred.successors[0] == node_id
                        if is_true:
                            ff_conds.append(And(branch_vars[pred_id], first_free_called[pred_id]))
                        else:
                            ff_conds.append(And(Not(branch_vars[pred_id]), first_free_called[pred_id]))
                    else:
                        ff_conds.append(first_free_called[pred_id])

                if ff_conds:
                    solver.add(first_free_called[node_id] == Or(*ff_conds))
                else:
                    solver.add(first_free_called[node_id] == False)

        # Double-free condition: reach a FREE node where first_free is already True
        double_free_conds = []
        for free_node in free_nodes:
            # Check predecessors: was first_free already True before this FREE?
            for pred_id in free_node.predecessors:
                double_free_conds.append(
                    And(reach[free_node.id], first_free_called[pred_id])
                )

        if not double_free_conds:
            return ReachabilityResult(False, "No double-free path possible")

        solver.add(Or(*double_free_conds))

        result = solver.check()

        if result == sat:
            model = solver.model()
            branch_vals = {cfg[nid].condition: str(model.evaluate(v, model_completion=True))
                          for nid, v in branch_vars.items()}
            return ReachabilityResult(
                reachable=True,
                reason=f"Double-free possible for '{var}'",
                model=branch_vals
            )
        else:
            return ReachabilityResult(
                reachable=False,
                reason=f"Double-free not possible - frees are mutually exclusive"
            )

    def check_use_after_free(
        self,
        cfg: dict[int, CFGNode],
        var: str
    ) -> ReachabilityResult:
        """Check if use-after-free is possible for a variable.

        Use-after-free occurs when:
        1. free(p) is called
        2. p is dereferenced/used after the free

        We look for: ALLOC -> FREE -> USE pattern
        """
        if not cfg:
            return ReachabilityResult(False, "Empty CFG")

        free_nodes = [n for n in cfg.values()
                      if n.node_type == NodeType.FREE and n.variable == var]

        if not free_nodes:
            return ReachabilityResult(False, f"No free() call for {var}")

        # Find potential use nodes (any node that uses the variable after free)
        # We look for nodes that reference the variable in their code
        use_nodes = []
        for node in cfg.values():
            if node.node_type in (NodeType.ASSIGN, NodeType.CALL):
                # Check if variable is used (not just assigned)
                code = node.code
                # Simple heuristic: var appears but not as "var =" or "free(var)"
                if var in code:
                    if not code.strip().startswith(f"{var} =") and \
                       not code.strip().startswith(f"{var}=") and \
                       f"free({var})" not in code:
                        use_nodes.append(node)

        if not use_nodes:
            return ReachabilityResult(False, f"No use of {var} after free detected")

        solver = Solver()

        # Branch variables
        branch_vars = {}
        for node_id, node in cfg.items():
            if node.node_type == NodeType.BRANCH:
                branch_vars[node_id] = Bool(f"branch_{node_id}")

        # Reachability and freed status
        reach = {n_id: Bool(f"reach_{n_id}") for n_id in cfg}
        freed = {n_id: Bool(f"freed_{n_id}") for n_id in cfg}

        # Entry
        for node_id, node in cfg.items():
            if node.node_type == NodeType.ENTRY:
                solver.add(reach[node_id] == True)
                solver.add(freed[node_id] == False)

        # Process nodes
        for node_id, node in cfg.items():
            if node.node_type == NodeType.ENTRY:
                continue

            preds = node.predecessors
            if not preds:
                solver.add(reach[node_id] == False)
                solver.add(freed[node_id] == False)
                continue

            # Reachability
            reach_conds = []
            for pred_id in preds:
                pred = cfg[pred_id]
                if pred.node_type == NodeType.BRANCH and pred_id in branch_vars:
                    is_true = pred.successors and pred.successors[0] == node_id
                    if is_true:
                        reach_conds.append(And(reach[pred_id], branch_vars[pred_id]))
                    else:
                        reach_conds.append(And(reach[pred_id], Not(branch_vars[pred_id])))
                else:
                    reach_conds.append(reach[pred_id])

            if reach_conds:
                solver.add(reach[node_id] == Or(*reach_conds))

            # Freed status
            if node.node_type == NodeType.FREE and node.variable == var:
                solver.add(Implies(reach[node_id], freed[node_id] == True))
                solver.add(Implies(Not(reach[node_id]), freed[node_id] == False))
            else:
                freed_conds = []
                for pred_id in preds:
                    pred = cfg[pred_id]
                    if pred.node_type == NodeType.FREE and pred.variable == var:
                        freed_conds.append(reach[pred_id])
                    elif pred.node_type == NodeType.BRANCH and pred_id in branch_vars:
                        is_true = pred.successors and pred.successors[0] == node_id
                        if is_true:
                            freed_conds.append(And(branch_vars[pred_id], freed[pred_id]))
                        else:
                            freed_conds.append(And(Not(branch_vars[pred_id]), freed[pred_id]))
                    else:
                        freed_conds.append(freed[pred_id])

                if freed_conds:
                    solver.add(freed[node_id] == Or(*freed_conds))
                else:
                    solver.add(freed[node_id] == False)

        # UAF condition: reach a USE node where freed is True
        uaf_conds = []
        for use_node in use_nodes:
            uaf_conds.append(And(reach[use_node.id], freed[use_node.id]))

        if not uaf_conds:
            return ReachabilityResult(False, "No use-after-free path")

        solver.add(Or(*uaf_conds))

        result = solver.check()

        if result == sat:
            model = solver.model()
            branch_vals = {cfg[nid].condition: str(model.evaluate(v, model_completion=True))
                          for nid, v in branch_vars.items()}
            return ReachabilityResult(
                reachable=True,
                reason=f"Use-after-free possible for '{var}'",
                model=branch_vals
            )
        else:
            return ReachabilityResult(
                reachable=False,
                reason=f"No use-after-free path found"
            )

    def check_null_deref(
        self,
        cfg: dict[int, CFGNode],
        var: str
    ) -> ReachabilityResult:
        """Check if null pointer dereference is possible.

        Null-deref occurs when:
        1. A pointer might be NULL (from malloc which can fail)
        2. It's dereferenced without a NULL check guarding it

        Key insight: After `if (p == NULL) return;`, the rest of the code
        knows p is NOT NULL. We track this information through the CFG.
        """
        if not cfg:
            return ReachabilityResult(False, "Empty CFG")

        solver = Solver()

        # Branch variables
        branch_vars = {}
        null_check_branches = {}  # branch_id -> null_is_true_branch

        for node_id, node in cfg.items():
            if node.node_type == NodeType.BRANCH:
                branch_vars[node_id] = Bool(f"branch_{node_id}")

                cond = node.condition.strip()
                # Detect NULL checks for our variable
                if f'{var} == NULL' in cond or f'{var} == 0' in cond:
                    null_check_branches[node_id] = True  # true branch = var IS NULL
                elif f'!{var}' == cond or cond == f'!{var}':
                    null_check_branches[node_id] = True  # !var means var IS NULL/0
                elif f'{var} != NULL' in cond or f'{var} != 0' in cond:
                    null_check_branches[node_id] = False  # true branch = var NOT NULL
                elif cond == var:  # if (var) means var is truthy (not NULL)
                    null_check_branches[node_id] = False  # true branch = var NOT NULL

        # Track reachability and "might be null" status
        reach = {n_id: Bool(f"reach_{n_id}") for n_id in cfg}
        might_null = {n_id: Bool(f"null_{n_id}") for n_id in cfg}

        # Entry: reachable, var status unknown (before allocation)
        for node_id, node in cfg.items():
            if node.node_type == NodeType.ENTRY:
                solver.add(reach[node_id] == True)
                solver.add(might_null[node_id] == False)  # Not allocated yet

        # Process nodes
        for node_id, node in cfg.items():
            if node.node_type == NodeType.ENTRY:
                continue

            preds = node.predecessors
            if not preds:
                solver.add(reach[node_id] == False)
                solver.add(might_null[node_id] == False)
                continue

            # Build reachability and null status conditions
            reach_conds = []
            null_conds = []

            for pred_id in preds:
                pred = cfg[pred_id]

                if pred.node_type == NodeType.BRANCH and pred_id in branch_vars:
                    is_true_branch = pred.successors and pred.successors[0] == node_id
                    branch_val = branch_vars[pred_id] if is_true_branch else Not(branch_vars[pred_id])

                    reach_conds.append(And(reach[pred_id], branch_val))

                    # Determine null status after this branch
                    if pred_id in null_check_branches:
                        null_is_true = null_check_branches[pred_id]
                        # If true branch means NULL and we're taking true branch, var IS NULL
                        # If true branch means NULL and we're taking false branch, var is NOT NULL
                        if is_true_branch:
                            var_is_null_here = null_is_true
                        else:
                            var_is_null_here = not null_is_true

                        if var_is_null_here:
                            # On this path, var is definitely NULL
                            null_conds.append(branch_val)
                        else:
                            # On this path, var is definitely NOT NULL
                            # Don't add to null_conds (contributes False to OR)
                            pass
                    else:
                        # Not a null check - inherit null status from predecessor
                        null_conds.append(And(branch_val, might_null[pred_id]))

                elif pred.node_type == NodeType.ALLOC and pred.variable == var:
                    reach_conds.append(reach[pred_id])
                    # After malloc, var MIGHT be NULL (malloc can fail)
                    null_conds.append(True)  # Conservatively assume might be null

                else:
                    reach_conds.append(reach[pred_id])
                    null_conds.append(might_null[pred_id])

            # Set reachability
            if reach_conds:
                solver.add(reach[node_id] == Or(*reach_conds))
            else:
                solver.add(reach[node_id] == False)

            # Set null status: might_null is True if ANY path has null
            if null_conds:
                solver.add(might_null[node_id] == Or(*null_conds))
            else:
                solver.add(might_null[node_id] == False)

        # Find dereference nodes
        deref_nodes = []
        for node in cfg.values():
            code = node.code
            # Look for dereference patterns: *var, var->, var[
            if f'*{var}' in code or f'{var}->' in code or f'{var}[' in code:
                # Exclude free(var) which is not a deref in the problematic sense
                if f'free({var})' not in code:
                    deref_nodes.append(node)

        if not deref_nodes:
            return ReachabilityResult(False, f"No dereference of {var} found")

        # Null-deref condition: reach deref node AND might_null is True
        deref_conds = []
        for deref_node in deref_nodes:
            deref_conds.append(And(reach[deref_node.id], might_null[deref_node.id]))

        solver.add(Or(*deref_conds))

        result = solver.check()

        if result == sat:
            model = solver.model()
            branch_vals = {}
            for nid, v in branch_vars.items():
                val = model.evaluate(v, model_completion=True)
                branch_vals[cfg[nid].condition] = str(val)
            return ReachabilityResult(
                reachable=True,
                reason=f"Null dereference possible for '{var}'",
                model=branch_vals
            )
        else:
            return ReachabilityResult(
                reachable=False,
                reason=f"No null dereference - all paths check for NULL before use"
            )


class AnnotationValidator:
    """Validates LLM-generated annotations using Z3 constraint solving."""

    def __init__(self):
        self.cfg_builder = CFGBuilder()
        if Z3_AVAILABLE:
            self.path_analyzer = Z3PathAnalyzer()
        else:
            self.path_analyzer = None
            logger.warning("Z3 not available - using simplified validation")

    def validate_annotations(
        self,
        annotations: AnnotationSet,
        functions: dict[str, FunctionInfo]
    ) -> tuple[AnnotationSet, list[str]]:
        """Validate all annotations using Z3 constraint solving.

        Returns:
            (valid_annotations, conflicts)
        """
        valid = AnnotationSet()
        conflicts = []

        for func_name, ann_list in annotations.annotations.items():
            func_info = functions.get(func_name)

            for ann in ann_list:
                if not func_info:
                    # External function - keep annotation
                    valid.add(ann)
                    continue

                # Check if this is a bug annotation
                if ann.is_bug_annotation():
                    is_valid, reason = self._validate_bug_annotation(ann, func_info)
                else:
                    is_valid, reason = self._validate_single(ann, func_info)

                if is_valid:
                    valid.add(ann)
                    logger.debug(f"Valid: {func_name} -> {ann.annotation_type.name}")
                else:
                    conflicts.append(f"CONFLICT: {func_name} as {ann.annotation_type.name}: {reason}")
                    logger.info(f"Conflict: {func_name} - {reason}")

        return valid, conflicts

    def _validate_bug_annotation(
        self,
        annotation: Annotation,
        func_info: FunctionInfo
    ) -> tuple[bool, str]:
        """Validate bug annotations using Z3 sliced analysis."""
        from src.symbolic.slicer import analyze_with_slicing

        # Map annotation type to issue type name
        type_map = {
            AnnotationType.POTENTIAL_LEAK: "MEMORY_LEAK",
            AnnotationType.USE_AFTER_FREE: "USE_AFTER_FREE",
            AnnotationType.DOUBLE_FREE: "DOUBLE_FREE",
            AnnotationType.NULL_DEREF: "NULL_DEREFERENCE",
        }

        issue_key = type_map.get(annotation.annotation_type)
        if not issue_key:
            return True, "Unknown bug type"

        # Run sliced analysis
        results = analyze_with_slicing(func_info.code)

        if issue_key in results and results[issue_key]:
            # Z3 confirms the bug is feasible
            return True, f"Z3 confirms feasible {issue_key}"
        else:
            # Z3 says bug is not feasible
            return False, f"Z3 says {issue_key} is infeasible on all paths"

    def _validate_single(
        self,
        annotation: Annotation,
        func_info: FunctionInfo
    ) -> tuple[bool, str]:
        """Validate a single function annotation."""
        code = func_info.code

        if annotation.annotation_type == AnnotationType.ALLOC_SOURCE:
            return self._validate_alloc(code, annotation.function_name)
        elif annotation.annotation_type == AnnotationType.FREE_SINK:
            return self._validate_free(code, annotation.function_name)
        else:
            return True, "OK"

    def _validate_alloc(self, code: str, func_name: str) -> tuple[bool, str]:
        """Validate allocation annotation by checking if function allocates memory."""
        alloc_patterns = [
            r'\bmalloc\s*\(',
            r'\bcalloc\s*\(',
            r'\brealloc\s*\(',
            r'\bstrdup\s*\(',
            r'\bstrndup\s*\(',
            r'\bnew\s+',
            r'\bnew\s*\[',
        ]

        for pattern in alloc_patterns:
            if re.search(pattern, code):
                # Check if result is returned
                if re.search(r'return\s+', code):
                    return True, "Allocates and returns memory"

        return False, "No allocation call found or result not returned"

    def _validate_free(self, code: str, func_name: str) -> tuple[bool, str]:
        """Validate free annotation."""
        free_patterns = [
            r'\bfree\s*\(',
            r'\bg_free\s*\(',
            r'\bdelete\s+',
            r'\bdelete\s*\[',
        ]

        for pattern in free_patterns:
            if re.search(pattern, code):
                return True, "Contains free call"

        return False, "No deallocation call found"

    def check_leak_in_caller(
        self,
        caller_func: FunctionInfo,
        alloc_func: str,
        annotations: AnnotationSet
    ) -> ReachabilityResult:
        """Check if calling alloc_func in caller can cause a leak.

        This is the main Z3-based analysis that checks path feasibility.
        """
        if not self.path_analyzer:
            return ReachabilityResult(True, "Z3 not available - assuming leak possible")

        # Add the allocating function to CFG builder's known allocators
        self.cfg_builder.alloc_funcs.add(alloc_func)

        # Build CFG
        cfg = self.cfg_builder.build(caller_func.code)

        if not cfg:
            return ReachabilityResult(True, "Could not build CFG")

        # Find allocation variable
        alloc_var = None
        for node in cfg.values():
            if node.node_type == NodeType.ALLOC and alloc_func in node.code:
                alloc_var = node.variable
                break

        if not alloc_var:
            # Check if alloc is called but result not stored
            if alloc_func + "(" in caller_func.code:
                if not re.search(rf'\w+\s*=\s*{alloc_func}\s*\(', caller_func.code):
                    return ReachabilityResult(
                        True,
                        f"Call to {alloc_func}() result not stored - immediate leak"
                    )
            return ReachabilityResult(False, f"No call to {alloc_func} found")

        # Use Z3 to check leak path
        return self.path_analyzer.check_leak_path(cfg, alloc_var)


def analyze_function_for_leaks(
    func_info: FunctionInfo,
    alloc_functions: set[str]
) -> list[ReachabilityResult]:
    """Analyze a function for potential memory leaks using Z3.

    Args:
        func_info: The function to analyze
        alloc_functions: Set of known allocation function names

    Returns:
        List of ReachabilityResult for each potential leak
    """
    results = []

    if not Z3_AVAILABLE:
        logger.warning("Z3 not available")
        return results

    cfg_builder = CFGBuilder(alloc_funcs=alloc_functions)
    path_analyzer = Z3PathAnalyzer()

    cfg = cfg_builder.build(func_info.code)

    if not cfg:
        return results

    # Find all allocation nodes
    alloc_nodes = [n for n in cfg.values() if n.node_type == NodeType.ALLOC]

    for alloc_node in alloc_nodes:
        if alloc_node.variable:
            result = path_analyzer.check_leak_path(cfg, alloc_node.variable)
            result.conflicting_conditions.append(f"Variable: {alloc_node.variable}")
            result.conflicting_conditions.append(f"Allocation: {alloc_node.code}")
            results.append(result)

    return results


def analyze_function_for_issues(
    func_info: FunctionInfo,
    alloc_functions: set[str],
    free_functions: set[str] = None,
    issue_types: list = None
) -> dict[str, list[ReachabilityResult]]:
    """Comprehensive analysis for all memory safety issues.

    Args:
        func_info: The function to analyze
        alloc_functions: Set of known allocation function names
        free_functions: Set of known free function names
        issue_types: List of MemoryIssueType to check (None = all)

    Returns:
        Dict mapping issue type name to list of results
    """
    from src.core.models import MemoryIssueType

    results = {
        "MEMORY_LEAK": [],
        "DOUBLE_FREE": [],
        "USE_AFTER_FREE": [],
        "NULL_DEREFERENCE": [],
    }

    if not Z3_AVAILABLE:
        logger.warning("Z3 not available")
        return results

    free_functions = free_functions or {"free", "g_free", "cfree"}

    cfg_builder = CFGBuilder(alloc_funcs=alloc_functions, free_funcs=free_functions)
    path_analyzer = Z3PathAnalyzer()

    cfg = cfg_builder.build(func_info.code)

    if not cfg:
        return results

    # Get all pointer variables (from ALLOC and FREE nodes)
    pointer_vars = set()
    for node in cfg.values():
        if node.node_type == NodeType.ALLOC and node.variable:
            pointer_vars.add(node.variable)
        if node.node_type == NodeType.FREE and node.variable:
            pointer_vars.add(node.variable)

    # Check each issue type for each pointer variable
    for var in pointer_vars:
        # Memory Leak
        if issue_types is None or MemoryIssueType.MEMORY_LEAK in issue_types:
            result = path_analyzer.check_leak_path(cfg, var)
            if result.reachable:
                result.conflicting_conditions.insert(0, f"Variable: {var}")
                results["MEMORY_LEAK"].append(result)

        # Double Free
        if issue_types is None or MemoryIssueType.DOUBLE_FREE in issue_types:
            result = path_analyzer.check_double_free(cfg, var)
            if result.reachable:
                result.conflicting_conditions.insert(0, f"Variable: {var}")
                results["DOUBLE_FREE"].append(result)

        # Use After Free
        if issue_types is None or MemoryIssueType.USE_AFTER_FREE in issue_types:
            result = path_analyzer.check_use_after_free(cfg, var)
            if result.reachable:
                result.conflicting_conditions.insert(0, f"Variable: {var}")
                results["USE_AFTER_FREE"].append(result)

        # Null Dereference
        if issue_types is None or MemoryIssueType.NULL_DEREFERENCE in issue_types:
            result = path_analyzer.check_null_deref(cfg, var)
            if result.reachable:
                result.conflicting_conditions.insert(0, f"Variable: {var}")
                results["NULL_DEREFERENCE"].append(result)

    return results