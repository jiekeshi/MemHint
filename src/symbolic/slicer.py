"""Constraint-Relevant Slicing for lightweight Z3 analysis.

Core idea: Before feeding CFG to Z3, we slice it to keep only:
1. The allocation/free/use nodes we care about
2. Branch conditions that affect reachability between them
3. Nothing else

This makes Z3 analysis O(relevant_path) instead of O(whole_function).

Example:
    Original function with 100 lines, but we only care about:
    - Line 5: malloc(p)
    - Line 20: if (flag)
    - Line 50: free(p)

    Sliced CFG: ALLOC -> BRANCH(flag) -> FREE -> EXIT
    Only 4 nodes instead of 100!
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

try:
    from z3 import Solver, Bool, And, Or, Not, Implies, sat, unsat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

import tree_sitter_c as tsc
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)
C_LANGUAGE = Language(tsc.language())


class SliceNodeType(Enum):
    """Types of nodes in a sliced CFG."""
    ENTRY = auto()
    EXIT = auto()
    ALLOC = auto()
    FREE = auto()
    USE = auto()       # Dereference/use of pointer
    BRANCH = auto()    # Relevant branch condition


@dataclass
class SliceNode:
    """A node in the sliced CFG - much simpler than full CFG."""
    id: int
    node_type: SliceNodeType
    line: int
    variable: str = ""
    condition: str = ""  # For BRANCH nodes
    successors: list[int] = field(default_factory=list)


@dataclass
class Slice:
    """A constraint-relevant slice for a specific variable."""
    variable: str
    nodes: dict[int, SliceNode]
    entry_id: int
    exit_ids: list[int]

    # Quick access
    alloc_ids: list[int] = field(default_factory=list)
    free_ids: list[int] = field(default_factory=list)
    use_ids: list[int] = field(default_factory=list)
    branch_ids: list[int] = field(default_factory=list)


@dataclass
class SliceResult:
    """Result of Z3 analysis on a slice."""
    feasible: bool
    reason: str
    condition: dict = None  # Branch values that trigger the bug


class ConstraintSlicer:
    """Extract constraint-relevant slices from C code.

    For a given pointer variable, we extract:
    1. Where it's allocated
    2. Where it's freed
    3. Where it's used/dereferenced
    4. Branch conditions between these points that could affect control flow
    """

    def __init__(self, alloc_funcs: set[str] = None, free_funcs: set[str] = None):
        self.parser = Parser(C_LANGUAGE)
        self.alloc_funcs = alloc_funcs or {"malloc", "calloc", "realloc", "strdup"}
        self.free_funcs = free_funcs or {"free", "g_free"}

    def slice_for_leak(self, code: str, var: str = None) -> list[Slice]:
        """Extract slices relevant for memory leak detection.

        For leak: we need ALLOC -> ... -> EXIT paths without FREE.
        Relevant branches: those between ALLOC and EXIT that could skip FREE.
        """
        tree = self.parser.parse(code.encode())
        source = code.encode()

        slices = []

        # Find all allocations
        allocs = self._find_allocations(tree.root_node, source)

        for alloc_var, alloc_line in allocs:
            if var and alloc_var != var:
                continue

            # Find frees and relevant branches for this variable
            frees = self._find_frees(tree.root_node, source, alloc_var)
            branches = self._find_relevant_branches(tree.root_node, source, alloc_var, alloc_line)
            exits = self._find_exits(tree.root_node, source)

            # Build minimal slice
            slice_obj = self._build_slice(
                alloc_var, alloc_line, frees, branches, exits,
                issue_type="leak"
            )
            slices.append(slice_obj)

        return slices

    def slice_for_double_free(self, code: str, var: str = None) -> list[Slice]:
        """Extract slices for double-free detection.

        For double-free: we need paths where FREE appears twice.
        """
        tree = self.parser.parse(code.encode())
        source = code.encode()

        slices = []

        # Find all frees and group by variable
        all_frees = self._find_all_frees(tree.root_node, source)

        for free_var, free_lines in all_frees.items():
            if len(free_lines) < 2:
                continue
            if var and free_var != var:
                continue

            # Find branches between frees
            min_line = min(free_lines)
            max_line = max(free_lines)
            branches = self._find_branches_in_range(tree.root_node, source, min_line, max_line)

            slice_obj = self._build_double_free_slice(free_var, free_lines, branches)
            slices.append(slice_obj)

        return slices

    def slice_for_uaf(self, code: str, var: str = None) -> list[Slice]:
        """Extract slices for use-after-free detection.

        For UAF: we need FREE -> USE paths.
        """
        tree = self.parser.parse(code.encode())
        source = code.encode()

        slices = []

        # Find all frees
        all_frees = self._find_all_frees(tree.root_node, source)

        for free_var, free_lines in all_frees.items():
            if var and free_var != var:
                continue

            # Find uses after any free
            uses = self._find_uses_after(tree.root_node, source, free_var, min(free_lines))

            if not uses:
                continue

            branches = self._find_branches_in_range(
                tree.root_node, source,
                min(free_lines), max(uses)
            )

            slice_obj = self._build_uaf_slice(free_var, free_lines, uses, branches)
            slices.append(slice_obj)

        return slices

    def slice_for_null_deref(self, code: str, var: str = None) -> list[Slice]:
        """Extract slices for null-dereference detection.

        For null-deref: ALLOC -> USE paths without NULL check.
        """
        tree = self.parser.parse(code.encode())
        source = code.encode()

        slices = []

        allocs = self._find_allocations(tree.root_node, source)

        for alloc_var, alloc_line in allocs:
            if var and alloc_var != var:
                continue

            # Find dereferences
            derefs = self._find_dereferences(tree.root_node, source, alloc_var, alloc_line)

            if not derefs:
                continue

            # Find NULL check branches
            null_checks = self._find_null_checks(tree.root_node, source, alloc_var)

            slice_obj = self._build_null_deref_slice(
                alloc_var, alloc_line, derefs, null_checks
            )
            slices.append(slice_obj)

        return slices

    def _find_allocations(self, root, source: bytes) -> list[tuple[str, int]]:
        """Find all allocations: var = malloc/calloc/...()

        Handles:
        - int *p = malloc(...)
        - p = malloc(...)
        - p = (int *)malloc(...)
        - p = (void*)malloc(...)
        """
        allocs = []

        for node in self._walk(root):
            if node.type in ("declaration", "expression_statement"):
                code = source[node.start_byte:node.end_byte].decode()
                line = node.start_point[0] + 1

                for alloc_func in self.alloc_funcs:
                    if f"{alloc_func}(" in code:
                        import re
                        # Match: var = alloc() or var = (type*)alloc() or *var = alloc()
                        # Pattern handles optional cast before alloc function
                        patterns = [
                            # p = malloc(...) or p = (type*)malloc(...)
                            rf'(\w+)\s*=\s*(?:\([^)]*\)\s*)?{alloc_func}\s*\(',
                            # int *p = malloc(...)
                            rf'\*\s*(\w+)\s*=\s*(?:\([^)]*\)\s*)?{alloc_func}\s*\(',
                            # type *p = malloc(...)
                            rf'\w+\s+\*\s*(\w+)\s*=\s*(?:\([^)]*\)\s*)?{alloc_func}\s*\(',
                        ]

                        for pattern in patterns:
                            match = re.search(pattern, code)
                            if match:
                                var = match.group(1)
                                allocs.append((var, line))
                                break
                        break

        return allocs

    def _find_frees(self, root, source: bytes, var: str) -> list[int]:
        """Find all free(var) calls."""
        frees = []

        for node in self._walk(root):
            if node.type == "call_expression":
                code = source[node.start_byte:node.end_byte].decode()
                for free_func in self.free_funcs:
                    if f"{free_func}({var})" in code or f"{free_func}( {var} )" in code:
                        frees.append(node.start_point[0] + 1)
                        break

        return frees

    def _find_all_frees(self, root, source: bytes) -> dict[str, list[int]]:
        """Find all free() calls grouped by variable."""
        import re
        frees = {}

        for node in self._walk(root):
            if node.type == "call_expression":
                code = source[node.start_byte:node.end_byte].decode()
                for free_func in self.free_funcs:
                    match = re.search(rf'{free_func}\s*\(\s*(\w+)\s*\)', code)
                    if match:
                        var = match.group(1)
                        if var not in frees:
                            frees[var] = []
                        frees[var].append(node.start_point[0] + 1)
                        break

        return frees

    def _find_relevant_branches(self, root, source: bytes, var: str, after_line: int) -> list[tuple[int, str]]:
        """Find branches that could affect whether free(var) is called."""
        branches = []

        for node in self._walk(root):
            if node.type == "if_statement":
                line = node.start_point[0] + 1
                if line <= after_line:
                    continue

                # Extract condition
                for child in node.children:
                    if child.type == "parenthesized_expression":
                        cond = source[child.start_byte:child.end_byte].decode()
                        cond = cond[1:-1].strip()  # Remove parens

                        # Check if this branch contains free(var)
                        if_code = source[node.start_byte:node.end_byte].decode()
                        for free_func in self.free_funcs:
                            if f"{free_func}({var})" in if_code:
                                branches.append((line, cond))
                                break
                        break

        return branches

    def _find_branches_in_range(self, root, source: bytes, start_line: int, end_line: int) -> list[tuple[int, str]]:
        """Find all branches within a line range."""
        branches = []

        for node in self._walk(root):
            if node.type == "if_statement":
                line = node.start_point[0] + 1
                if start_line <= line <= end_line:
                    for child in node.children:
                        if child.type == "parenthesized_expression":
                            cond = source[child.start_byte:child.end_byte].decode()
                            cond = cond[1:-1].strip()
                            branches.append((line, cond))
                            break

        return branches

    def _find_exits(self, root, source: bytes) -> list[int]:
        """Find return statements."""
        exits = []

        for node in self._walk(root):
            if node.type == "return_statement":
                exits.append(node.start_point[0] + 1)

        # Also add implicit exit (end of function)
        exits.append(9999)  # Sentinel for function end

        return exits

    def _find_uses_after(self, root, source: bytes, var: str, after_line: int) -> list[int]:
        """Find uses of variable after a given line."""
        uses = []

        for node in self._walk(root):
            line = node.start_point[0] + 1
            if line <= after_line:
                continue

            if node.type == "identifier":
                if source[node.start_byte:node.end_byte].decode() == var:
                    # Check if it's a use (not in free() call)
                    parent = node.parent
                    if parent and parent.type != "argument_list":
                        uses.append(line)
                    elif parent:
                        # Check if parent's parent is a free call
                        gparent = parent.parent
                        if gparent and gparent.type == "call_expression":
                            call_code = source[gparent.start_byte:gparent.end_byte].decode()
                            if not any(f in call_code for f in self.free_funcs):
                                uses.append(line)

        return list(set(uses))

    def _find_dereferences(self, root, source: bytes, var: str, after_line: int) -> list[int]:
        """Find dereferences: *var, var->, var[...]"""
        derefs = []

        for node in self._walk(root):
            line = node.start_point[0] + 1
            if line <= after_line:
                continue

            code = source[node.start_byte:node.end_byte].decode()

            if node.type in ("pointer_expression", "subscript_expression", "field_expression"):
                if var in code:
                    derefs.append(line)
            elif node.type == "expression_statement":
                if f"*{var}" in code or f"{var}->" in code or f"{var}[" in code:
                    derefs.append(line)

        return list(set(derefs))

    def _find_null_checks(self, root, source: bytes, var: str) -> list[tuple[int, str, bool]]:
        """Find NULL checks for variable.

        Returns: [(line, condition, null_is_true_branch), ...]
        """
        checks = []

        for node in self._walk(root):
            if node.type == "if_statement":
                line = node.start_point[0] + 1

                for child in node.children:
                    if child.type == "parenthesized_expression":
                        cond = source[child.start_byte:child.end_byte].decode()
                        cond = cond[1:-1].strip()

                        # Check patterns
                        if f"{var} == NULL" in cond or f"{var} == 0" in cond:
                            checks.append((line, cond, True))
                        elif f"{var} != NULL" in cond or f"{var} != 0" in cond:
                            checks.append((line, cond, False))
                        elif cond == f"!{var}":
                            checks.append((line, cond, True))
                        elif cond == var:
                            checks.append((line, cond, False))
                        break

        return checks

    def _build_slice(self, var: str, alloc_line: int, free_lines: list[int],
                     branches: list[tuple[int, str]], exit_lines: list[int],
                     issue_type: str) -> Slice:
        """Build a minimal slice."""
        nodes = {}
        node_id = 0

        # Entry
        entry = SliceNode(node_id, SliceNodeType.ENTRY, 0)
        nodes[node_id] = entry
        entry_id = node_id
        node_id += 1

        # Alloc
        alloc = SliceNode(node_id, SliceNodeType.ALLOC, alloc_line, variable=var)
        nodes[node_id] = alloc
        entry.successors.append(node_id)
        alloc_ids = [node_id]
        last_id = node_id
        node_id += 1

        # Branches (sorted by line)
        branch_ids = []
        for line, cond in sorted(branches, key=lambda x: x[0]):
            branch = SliceNode(node_id, SliceNodeType.BRANCH, line, condition=cond)
            nodes[node_id] = branch
            nodes[last_id].successors.append(node_id)
            branch_ids.append(node_id)
            last_id = node_id
            node_id += 1

        # Frees
        free_ids = []
        for line in sorted(free_lines):
            free_node = SliceNode(node_id, SliceNodeType.FREE, line, variable=var)
            nodes[node_id] = free_node
            nodes[last_id].successors.append(node_id)
            free_ids.append(node_id)
            last_id = node_id
            node_id += 1

        # Exit
        exit_node = SliceNode(node_id, SliceNodeType.EXIT, min(exit_lines))
        nodes[node_id] = exit_node
        nodes[last_id].successors.append(node_id)
        exit_ids = [node_id]

        # For branches, add skip edges (false branch skips to next)
        for i, bid in enumerate(branch_ids):
            # False branch could skip to exit or next non-branch
            if i < len(branch_ids) - 1:
                nodes[bid].successors.append(branch_ids[i + 1])
            elif free_ids:
                # Skip to first free or exit
                pass  # Already connected through true path
            nodes[bid].successors.append(exit_ids[0])

        return Slice(
            variable=var,
            nodes=nodes,
            entry_id=entry_id,
            exit_ids=exit_ids,
            alloc_ids=alloc_ids,
            free_ids=free_ids,
            branch_ids=branch_ids
        )

    def _build_double_free_slice(self, var: str, free_lines: list[int],
                                  branches: list[tuple[int, str]]) -> Slice:
        """Build slice for double-free detection."""
        nodes = {}
        node_id = 0

        entry = SliceNode(node_id, SliceNodeType.ENTRY, 0)
        nodes[node_id] = entry
        entry_id = node_id
        node_id += 1

        last_id = entry_id
        free_ids = []
        branch_ids = []

        # Interleave branches and frees by line number
        events = [(line, "free", None) for line in free_lines]
        events += [(line, "branch", cond) for line, cond in branches]
        events.sort(key=lambda x: x[0])

        for line, etype, cond in events:
            if etype == "free":
                node = SliceNode(node_id, SliceNodeType.FREE, line, variable=var)
                free_ids.append(node_id)
            else:
                node = SliceNode(node_id, SliceNodeType.BRANCH, line, condition=cond)
                branch_ids.append(node_id)

            nodes[node_id] = node
            nodes[last_id].successors.append(node_id)
            last_id = node_id
            node_id += 1

        # Exit
        exit_node = SliceNode(node_id, SliceNodeType.EXIT, 9999)
        nodes[node_id] = exit_node
        nodes[last_id].successors.append(node_id)

        return Slice(
            variable=var,
            nodes=nodes,
            entry_id=entry_id,
            exit_ids=[node_id],
            free_ids=free_ids,
            branch_ids=branch_ids
        )

    def _build_uaf_slice(self, var: str, free_lines: list[int],
                         use_lines: list[int], branches: list[tuple[int, str]]) -> Slice:
        """Build slice for use-after-free detection."""
        nodes = {}
        node_id = 0

        entry = SliceNode(node_id, SliceNodeType.ENTRY, 0)
        nodes[node_id] = entry
        entry_id = node_id
        node_id += 1

        last_id = entry_id
        free_ids = []
        use_ids = []
        branch_ids = []

        events = [(line, "free", None) for line in free_lines]
        events += [(line, "use", None) for line in use_lines]
        events += [(line, "branch", cond) for line, cond in branches]
        events.sort(key=lambda x: x[0])

        for line, etype, cond in events:
            if etype == "free":
                node = SliceNode(node_id, SliceNodeType.FREE, line, variable=var)
                free_ids.append(node_id)
            elif etype == "use":
                node = SliceNode(node_id, SliceNodeType.USE, line, variable=var)
                use_ids.append(node_id)
            else:
                node = SliceNode(node_id, SliceNodeType.BRANCH, line, condition=cond)
                branch_ids.append(node_id)

            nodes[node_id] = node
            nodes[last_id].successors.append(node_id)
            last_id = node_id
            node_id += 1

        exit_node = SliceNode(node_id, SliceNodeType.EXIT, 9999)
        nodes[node_id] = exit_node
        nodes[last_id].successors.append(node_id)

        return Slice(
            variable=var,
            nodes=nodes,
            entry_id=entry_id,
            exit_ids=[node_id],
            free_ids=free_ids,
            use_ids=use_ids,
            branch_ids=branch_ids
        )

    def _build_null_deref_slice(self, var: str, alloc_line: int,
                                 deref_lines: list[int],
                                 null_checks: list[tuple[int, str, bool]]) -> Slice:
        """Build slice for null-dereference detection."""
        nodes = {}
        node_id = 0

        entry = SliceNode(node_id, SliceNodeType.ENTRY, 0)
        nodes[node_id] = entry
        entry_id = node_id
        node_id += 1

        # Alloc
        alloc = SliceNode(node_id, SliceNodeType.ALLOC, alloc_line, variable=var)
        nodes[node_id] = alloc
        entry.successors.append(node_id)
        alloc_ids = [node_id]
        last_id = node_id
        node_id += 1

        use_ids = []
        branch_ids = []

        # Sort by line
        events = [(line, "check", (cond, null_is_true)) for line, cond, null_is_true in null_checks]
        events += [(line, "deref", None) for line in deref_lines]
        events.sort(key=lambda x: x[0])

        for line, etype, data in events:
            if etype == "check":
                cond, null_is_true = data
                # Store null_is_true in condition string for later
                node = SliceNode(node_id, SliceNodeType.BRANCH, line,
                                condition=f"{cond}|{null_is_true}")
                branch_ids.append(node_id)
            else:
                node = SliceNode(node_id, SliceNodeType.USE, line, variable=var)
                use_ids.append(node_id)

            nodes[node_id] = node
            nodes[last_id].successors.append(node_id)
            last_id = node_id
            node_id += 1

        exit_node = SliceNode(node_id, SliceNodeType.EXIT, 9999)
        nodes[node_id] = exit_node
        nodes[last_id].successors.append(node_id)

        return Slice(
            variable=var,
            nodes=nodes,
            entry_id=entry_id,
            exit_ids=[node_id],
            alloc_ids=alloc_ids,
            use_ids=use_ids,
            branch_ids=branch_ids
        )

    def _walk(self, node):
        """Walk AST."""
        yield node
        for child in node.children:
            yield from self._walk(child)


class SliceAnalyzer:
    """Analyze slices with Z3 - very fast because slices are tiny."""

    def check_leak(self, slice_obj: Slice) -> SliceResult:
        """Check if leak is feasible in this slice.

        Leak = reach EXIT without going through FREE.
        """
        if not Z3_AVAILABLE:
            return SliceResult(True, "Z3 not available")

        if not slice_obj.alloc_ids:
            return SliceResult(False, "No allocation in slice")

        if slice_obj.free_ids:
            # Has free - need to check if all paths go through it
            if not slice_obj.branch_ids:
                # No branches, free is always called
                return SliceResult(False, "Free always called (no branches)")

            # Check with Z3 if we can skip free
            return self._check_can_skip(slice_obj, slice_obj.free_ids, "free")
        else:
            # No free at all
            return SliceResult(True, "No free() in any path")

    def check_double_free(self, slice_obj: Slice) -> SliceResult:
        """Check if double-free is feasible."""
        if not Z3_AVAILABLE:
            return SliceResult(True, "Z3 not available")

        if len(slice_obj.free_ids) < 2:
            return SliceResult(False, "Less than 2 frees")

        if not slice_obj.branch_ids:
            # No branches - both frees always called
            return SliceResult(True, "Both frees always executed")

        # Check if both frees can be reached
        return self._check_can_reach_both(slice_obj, slice_obj.free_ids)

    def check_uaf(self, slice_obj: Slice) -> SliceResult:
        """Check if use-after-free is feasible."""
        if not Z3_AVAILABLE:
            return SliceResult(True, "Z3 not available")

        if not slice_obj.free_ids or not slice_obj.use_ids:
            return SliceResult(False, "Missing free or use")

        # Check if we can reach USE after FREE
        return self._check_free_then_use(slice_obj)

    def check_null_deref(self, slice_obj: Slice) -> SliceResult:
        """Check if null-dereference is feasible."""
        if not Z3_AVAILABLE:
            return SliceResult(True, "Z3 not available")

        if not slice_obj.use_ids:
            return SliceResult(False, "No dereference")

        if not slice_obj.branch_ids:
            # No NULL check - definitely possible
            return SliceResult(True, "No NULL check before dereference")

        # Check if we can reach USE while might-be-null
        return self._check_null_path(slice_obj)

    def _check_can_skip(self, slice_obj: Slice, target_ids: list[int],
                        target_name: str) -> SliceResult:
        """Check if we can reach EXIT while skipping all target nodes."""
        solver = Solver()

        # Create branch variables
        branch_vars = {}
        for bid in slice_obj.branch_ids:
            branch_vars[bid] = Bool(f"b_{bid}")

        # We want: reach EXIT AND NOT reach any target
        # With branches, this means: some branch assignment skips all targets

        # Simple model: each branch either goes to next or skips ahead
        # If skipping covers all targets, leak is possible

        # For now, simple heuristic: if any branch could skip a target, SAT
        # Real implementation would build full path constraints

        # Check if targets are "inside" any branch
        target_lines = {slice_obj.nodes[tid].line for tid in target_ids}

        for bid in slice_obj.branch_ids:
            branch_line = slice_obj.nodes[bid].line
            # If branch is before targets and could skip them
            if branch_line < min(target_lines):
                # This branch might skip the targets
                solver.add(Or(*[Not(branch_vars[b]) for b in slice_obj.branch_ids]))
                break

        if solver.check() == sat:
            model = solver.model()
            conds = {}
            for bid in slice_obj.branch_ids:
                conds[slice_obj.nodes[bid].condition] = str(
                    model.evaluate(branch_vars[bid], model_completion=True)
                )
            return SliceResult(True, f"Can skip {target_name}", conds)

        return SliceResult(False, f"All paths go through {target_name}")

    def _check_can_reach_both(self, slice_obj: Slice, target_ids: list[int]) -> SliceResult:
        """Check if we can reach multiple targets on same path."""
        solver = Solver()

        branch_vars = {}
        for bid in slice_obj.branch_ids:
            branch_vars[bid] = Bool(f"b_{bid}")

        # Both frees reachable = at least one branch configuration allows both
        # Simplification: if no branch separates them, both always reached

        if len(target_ids) >= 2:
            # Check if any branch is between the two frees
            t1_line = slice_obj.nodes[target_ids[0]].line
            t2_line = slice_obj.nodes[target_ids[1]].line

            branches_between = [
                bid for bid in slice_obj.branch_ids
                if t1_line < slice_obj.nodes[bid].line < t2_line
            ]

            if not branches_between:
                return SliceResult(True, "Both frees always executed (no branch between)")

            # There's a branch between - check if it can be bypassed
            for bid in branches_between:
                # If true branch skips second free, need false
                # This is a simplification
                solver.add(branch_vars[bid] == True)

            if solver.check() == sat:
                model = solver.model()
                conds = {slice_obj.nodes[bid].condition: str(model.evaluate(branch_vars[bid], model_completion=True))
                        for bid in slice_obj.branch_ids}
                return SliceResult(True, "Both frees can be reached", conds)

        return SliceResult(False, "Frees are mutually exclusive")

    def _check_free_then_use(self, slice_obj: Slice) -> SliceResult:
        """Check if we can reach USE after FREE."""
        # Get line numbers
        free_lines = [slice_obj.nodes[fid].line for fid in slice_obj.free_ids]
        use_lines = [slice_obj.nodes[uid].line for uid in slice_obj.use_ids]

        # Any use after any free?
        for use_line in use_lines:
            for free_line in free_lines:
                if use_line > free_line:
                    # Check if branch could prevent this
                    branches_between = [
                        bid for bid in slice_obj.branch_ids
                        if free_line < slice_obj.nodes[bid].line < use_line
                    ]

                    if not branches_between:
                        return SliceResult(True, f"Use at line {use_line} after free at line {free_line}")

        return SliceResult(False, "No use-after-free path")

    def _check_null_path(self, slice_obj: Slice) -> SliceResult:
        """Check if we can reach USE while pointer might be NULL."""
        solver = Solver()

        # Parse null check info from branch conditions
        # Format: "cond|True" or "cond|False"

        branch_vars = {}
        for bid in slice_obj.branch_ids:
            branch_vars[bid] = Bool(f"b_{bid}")

        # might_null = True after alloc (malloc can fail)
        # After null check branch, might_null depends on which branch taken

        might_null = Bool("might_null")
        solver.add(might_null == True)  # After malloc, might be null

        for bid in sorted(slice_obj.branch_ids, key=lambda x: slice_obj.nodes[x].line):
            cond = slice_obj.nodes[bid].condition
            if "|" in cond:
                _, null_is_true = cond.rsplit("|", 1)
                null_is_true = null_is_true == "True"

                # If we take the "not null" path, might_null becomes False
                if null_is_true:
                    # True branch = NULL, False branch = not NULL
                    # To reach USE without null, need False branch
                    solver.add(Implies(Not(branch_vars[bid]), might_null == False))
                else:
                    # True branch = not NULL
                    solver.add(Implies(branch_vars[bid], might_null == False))

        # Can we reach USE with might_null == True?
        solver.add(might_null == True)

        if solver.check() == sat:
            model = solver.model()
            conds = {slice_obj.nodes[bid].condition.split("|")[0]:
                    str(model.evaluate(branch_vars[bid], model_completion=True))
                    for bid in slice_obj.branch_ids}
            return SliceResult(True, "Can reach dereference while might be NULL", conds)

        return SliceResult(False, "All paths check for NULL before use")


def analyze_with_slicing(code: str, alloc_funcs: set[str] = None) -> dict[str, list[SliceResult]]:
    """Lightweight analysis using constraint-relevant slicing.

    This is the main entry point for sliced analysis.
    Much faster than full CFG analysis for large functions.
    """
    slicer = ConstraintSlicer(alloc_funcs=alloc_funcs)
    analyzer = SliceAnalyzer()

    results = {
        "MEMORY_LEAK": [],
        "DOUBLE_FREE": [],
        "USE_AFTER_FREE": [],
    }

    # Memory leak
    for slice_obj in slicer.slice_for_leak(code):
        result = analyzer.check_leak(slice_obj)
        if result.feasible:
            results["MEMORY_LEAK"].append(result)

    # Double free
    for slice_obj in slicer.slice_for_double_free(code):
        result = analyzer.check_double_free(slice_obj)
        if result.feasible:
            results["DOUBLE_FREE"].append(result)

    # Use after free
    for slice_obj in slicer.slice_for_uaf(code):
        result = analyzer.check_uaf(slice_obj)
        if result.feasible:
            results["USE_AFTER_FREE"].append(result)

    return results