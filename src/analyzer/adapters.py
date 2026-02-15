"""CodeQL Analyzer with Enhanced Queries.

This module provides:
1. CodeQL database creation and management
2. Custom memory model injection
3. Enhanced memory safety queries (hardcoded improvements over standard queries)
"""

import json
import logging
import subprocess
import tempfile
import shutil
import yaml
from pathlib import Path
from shutil import which
from typing import Optional
from src.core.models import Warning, HintSet, MemoryIssueType, CustomQuerySet

logger = logging.getLogger(__name__)

CODEQL_ISSUE_MAP = {
    "memory-never-freed": MemoryIssueType.MEMORY_LEAK,
    "memory-may-not-be-freed": MemoryIssueType.MEMORY_LEAK,
    "cpp/memory-never-freed": MemoryIssueType.MEMORY_LEAK,
    "cpp/memory-may-not-be-freed": MemoryIssueType.MEMORY_LEAK,
    "double-free": MemoryIssueType.DOUBLE_FREE,
    "cpp/double-free": MemoryIssueType.DOUBLE_FREE,
    "use-after-free": MemoryIssueType.USE_AFTER_FREE,
    "cpp/use-after-free": MemoryIssueType.USE_AFTER_FREE,
    # Enhanced queries
    "cpp/memory-never-freed-enhanced": MemoryIssueType.MEMORY_LEAK,
    "cpp/memory-may-not-be-freed-enhanced": MemoryIssueType.MEMORY_LEAK,
    "cpp/double-free-enhanced": MemoryIssueType.DOUBLE_FREE,
    "cpp/use-after-free-enhanced": MemoryIssueType.USE_AFTER_FREE,
    # Filtered queries
    "cpp/memory-never-freed-enhanced-filtered": MemoryIssueType.MEMORY_LEAK_FILTERED,
    "cpp/memory-may-not-be-freed-enhanced-filtered": MemoryIssueType.MEMORY_LEAK_FILTERED,
    "cpp/double-free-enhanced-filtered": MemoryIssueType.DOUBLE_FREE_FILTERED,
    "cpp/use-after-free-enhanced-filtered": MemoryIssueType.USE_AFTER_FREE_FILTERED,
    
}

# Standard memory queries
MEMORY_QUERIES = [
    "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
    "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
    "codeql/cpp-queries:Critical/DoubleFree.ql",
    "codeql/cpp-queries:Critical/UseAfterFree.ql",
]


# =============================================================================
# Enhanced Queries (Hardcoded)
# =============================================================================

ENHANCED_MEMORY_NEVER_FREED = '''/**
 * @name Memory is never freed (Enhanced)
 * @description Enhanced version with better error-path detection.
 *              Detects memory allocated but never freed, including in error paths.
 * @kind problem
 * @id cpp/memory-never-freed-enhanced
 * @problem.severity warning
 * @security-severity 7.5
 * @tags efficiency
 *       security
 *       external/cwe/cwe-401
 */
import MemoryFreed

/**
 * Additional predicate to detect allocations in error-handling branches
 * that may be missed by the standard query.
 */
predicate allocInErrorBranch(AllocationExpr alloc) {
  exists(IfStmt ifStmt, BlockStmt block |
    // Allocation is in a conditional branch
    (block = ifStmt.getThen() or block = ifStmt.getElse()) and
    alloc.getEnclosingStmt().getParentStmt*() = block
  )
}

/**
 * Detects allocation in a loop where the pointer may be overwritten.
 */
predicate allocInLoopMayOverwrite(AllocationExpr alloc) {
  exists(Loop loop, Variable v, AssignExpr assign |
    alloc.getEnclosingStmt().getParentStmt*() = loop.getStmt() and
    assign.getRValue() = alloc and
    assign.getLValue().(VariableAccess).getTarget() = v and
    // Same variable assigned elsewhere in the loop
    exists(AssignExpr other |
      other != assign and
      other.getLValue().(VariableAccess).getTarget() = v and
      other.getEnclosingStmt().getParentStmt*() = loop.getStmt()
    )
  )
}

from AllocationExpr alloc, string reason
where
  alloc.requiresDealloc() and
  not allocMayBeFreed(alloc) and
  (
    // Standard case
    not allocInErrorBranch(alloc) and not allocInLoopMayOverwrite(alloc) and reason = "never freed"
    or
    // Error branch case
    allocInErrorBranch(alloc) and reason = "never freed (in conditional branch)"
    or
    // Loop overwrite case
    allocInLoopMayOverwrite(alloc) and reason = "may leak in loop (pointer overwritten)"
  )
select alloc, "This memory is " + reason + "."
'''

ENHANCED_MEMORY_MAY_NOT_BE_FREED = '''/**
 * @name Memory may not be freed (Enhanced)
 * @description Enhanced version that better tracks memory through early returns.
 * @kind problem
 * @id cpp/memory-may-not-be-freed-enhanced
 * @problem.severity warning
 * @security-severity 7.5
 * @tags efficiency
 *       security
 *       external/cwe/cwe-401
 */

import MemoryFreed
import semmle.code.cpp.controlflow.StackVariableReachability

predicate mayCallFunction(Expr call, Function f) {
  call.(FunctionCall).getTarget() = f or
  call.(VariableCall).getVariable().getAnAssignedValue().getAChild*().(FunctionAccess).getTarget() = f
}

predicate allocCallOrIndirect(Expr e) {
  e.(AllocationExpr).requiresDealloc() and
  allocMayBeFreed(e)
  or
  exists(ReturnStmt rtn |
    mayCallFunction(e, rtn.getEnclosingFunction()) and
    (
      allocCallOrIndirect(rtn.getExpr())
      or
      exists(StackVariable v |
        v = rtn.getExpr().(VariableAccess).getTarget() and
        allocCallOrIndirect(v.getAnAssignedValue()) and
        not assignedToFieldOrGlobal(v, _)
      )
    )
  )
}

predicate verifiedRealloc(FunctionCall reallocCall, Variable v, ControlFlowNode verified) {
  reallocCall.(AllocationExpr).getReallocPtr() = v.getAnAccess() and
  (
    exists(Variable newV, ControlFlowNode node |
      newV.getAnAssignedValue() = reallocCall and
      node.(AnalysedExpr).getNonNullSuccessor(newV) = verified and
      newV != v
    )
    or
    reallocCall.(AllocationExpr).getReallocPtr().getValue() = "0" and
    verified = reallocCall
  )
}

predicate freeCallOrIndirect(ControlFlowNode n, Variable v) {
  n.(DeallocationExpr).getFreedExpr() = v.getAnAccess() and
  not exists(n.(AllocationExpr).getReallocPtr())
  or
  verifiedRealloc(_, v, n)
  or
  exists(FunctionCall midcall, Function mid, int arg |
    n.(Call).getArgument(arg) = v.getAnAccess() and
    mayCallFunction(n, mid) and
    midcall.getEnclosingFunction() = mid and
    freeCallOrIndirect(midcall, mid.getParameter(arg))
  )
}

predicate allocationDefinition(StackVariable v, ControlFlowNode def) {
  exists(Expr expr | exprDefinition(v, def, expr) and allocCallOrIndirect(expr))
}

class AllocVariableReachability extends StackVariableReachabilityWithReassignment {
  AllocVariableReachability() { this = "AllocVariableReachability" }

  override predicate isSourceActual(ControlFlowNode node, StackVariable v) {
    allocationDefinition(v, node)
  }

  override predicate isSinkActual(ControlFlowNode node, StackVariable v) {
    exists(node.(AnalysedExpr).getNullSuccessor(v)) or
    freeCallOrIndirect(node, v) or
    assignedToFieldOrGlobal(v, node) or
    v.getFunction() = node.(ReturnStmt).getEnclosingFunction()
  }

  override predicate isBarrier(ControlFlowNode node, StackVariable v) { definitionBarrier(v, node) }
}

predicate allocatedVariableReaches(StackVariable v, ControlFlowNode def, ControlFlowNode node) {
  exists(AllocVariableReachability r |
    r.reachesTo(def, _, node, v)
    or
    r.isSource(def, v) and node = def
  )
}

class AllocReachability extends StackVariableReachabilityExt {
  AllocReachability() { this = "AllocReachability" }

  override predicate isSource(ControlFlowNode node, StackVariable v) {
    allocationDefinition(v, node)
  }

  override predicate isSink(ControlFlowNode node, StackVariable v) {
    v.getFunction() = node.(ReturnStmt).getEnclosingFunction()
  }

  override predicate isBarrier(
    ControlFlowNode source, ControlFlowNode node, ControlFlowNode next, StackVariable v
  ) {
    this.isSource(source, v) and
    next = node.getASuccessor() and
    exists(StackVariable v0 | allocatedVariableReaches(v0, source, node) |
      node.(AnalysedExpr).getNullSuccessor(v0) = next or
      freeCallOrIndirect(node, v0) or
      assignedToFieldOrGlobal(v0, node)
    )
  }
}

predicate allocationReaches(ControlFlowNode def, ControlFlowNode node) {
  exists(AllocReachability r | r.reaches(def, _, node))
}

predicate assignedToFieldOrGlobal(StackVariable v, Expr e) {
  e.(Assignment).getRValue() = v.getAnAccess() and
  not e.(Assignment).getLValue().(VariableAccess).getTarget() instanceof StackVariable
  or
  exists(Expr midExpr, Function mid, int arg |
    e.(FunctionCall).getArgument(arg) = v.getAnAccess() and
    mayCallFunction(e, mid) and
    midExpr.getEnclosingFunction() = mid and
    assignedToFieldOrGlobal(mid.getParameter(arg), midExpr)
  )
  or
  e.(ConstructorFieldInit).getExpr() = v.getAnAccess()
}

/**
 * Enhanced: Check if return is in an error-handling context
 */
predicate isErrorReturn(ReturnStmt ret) {
  exists(IfStmt ifStmt |
    ret.getEnclosingStmt().getParentStmt*() = ifStmt.getThen() and
    (
      ifStmt.getCondition().(EqualityOperation).getAnOperand() instanceof NullValue
      or
      exists(ComparisonOperation cmp | cmp = ifStmt.getCondition() |
        cmp.getAnOperand().(Literal).getValue().toInt() <= 0
      )
    )
  )
}

from ControlFlowNode def, ReturnStmt ret, string context
where
  allocationReaches(def, ret) and
  not exists(StackVariable v |
    allocatedVariableReaches(v, def, ret) and
    ret.getAChild*() = v.getAnAccess()
  ) and
  (
    isErrorReturn(ret) and context = "error return"
    or
    not isErrorReturn(ret) and context = "exit point"
  )
select def, "This memory allocation may not be released at $@ (" + context + ").", ret, "this " + context
'''

ENHANCED_DOUBLE_FREE = '''/**
 * @name Potential double free (Enhanced)
 * @description Enhanced version with better control-flow sensitivity for conditional frees.
 * @kind path-problem
 * @precision high
 * @id cpp/double-free-enhanced
 * @problem.severity warning
 * @security-severity 9.3
 * @tags reliability
 *       security
 *       external/cwe/cwe-415
 */

import cpp
import semmle.code.cpp.dataflow.new.DataFlow
import semmle.code.cpp.security.flowafterfree.FlowAfterFree
import DoubleFree::PathGraph

predicate isFree(DataFlow::Node n, Expr e) { isFree(_, n, e, _) }

/**
 * Enhanced: Detect free in both branches of an if statement
 */
predicate freeInBothBranches(DeallocationExpr free1, DeallocationExpr free2, Variable v) {
  exists(IfStmt ifStmt |
    free1.getFreedExpr().(VariableAccess).getTarget() = v and
    free2.getFreedExpr().(VariableAccess).getTarget() = v and
    free1.getEnclosingStmt().getParentStmt*() = ifStmt.getThen() and
    free2.getEnclosingStmt().getParentStmt*() = ifStmt.getElse() and
    exists(DeallocationExpr free3 |
      free3.getFreedExpr().(VariableAccess).getTarget() = v and
      ifStmt.getASuccessor+() = free3
    )
  )
}

/**
 * Enhanced: Detect free in a loop that may execute multiple times
 */
predicate freeInLoop(DeallocationExpr free, Variable v) {
  exists(Loop loop |
    free.getFreedExpr().(VariableAccess).getTarget() = v and
    free.getEnclosingStmt().getParentStmt*() = loop.getStmt() and
    not exists(AllocationExpr alloc |
      alloc.getEnclosingStmt().getParentStmt*() = loop.getStmt() and
      exists(AssignExpr assign |
        assign.getRValue() = alloc and
        assign.getLValue().(VariableAccess).getTarget() = v
      )
    )
  )
}

module DoubleFreeParam implements FlowFromFreeParamSig {
  predicate isSink = isFree/2;
  predicate isExcluded = isExcludedMmFreePageFromMdl/2;
  predicate sourceSinkIsRelated = defaultSourceSinkIsRelated/2;
}

module DoubleFree = FlowFromFree<DoubleFreeParam>;

from DoubleFree::PathNode source, DoubleFree::PathNode sink, DeallocationExpr dealloc, Expr e2, string detail
where
  DoubleFree::flowPath(source, sink) and
  isFree(source.getNode(), _, _, dealloc) and
  isFree(sink.getNode(), e2) and
  (
    exists(Variable v |
      freeInBothBranches(dealloc, _, v) and detail = " (freed in both branches)"
      or
      freeInLoop(dealloc, v) and detail = " (freed in loop)"
    )
    or
    not exists(Variable v | freeInBothBranches(dealloc, _, v) or freeInLoop(dealloc, v)) and detail = ""
  )
select sink.getNode(), source, sink, "Memory pointed to by $@ may already have been freed by $@" + detail + ".",
  e2, e2.toString(), dealloc, dealloc.toString()
'''

ENHANCED_USE_AFTER_FREE = '''/**
 * @name Potential use after free (Enhanced)
 * @description Enhanced version with better detection of uses in different control flow paths.
 * @kind path-problem
 * @precision high
 * @id cpp/use-after-free-enhanced
 * @problem.severity warning
 * @security-severity 9.3
 * @tags reliability
 *       security
 *       external/cwe/cwe-416
 */

import cpp
import semmle.code.cpp.dataflow.new.DataFlow
import semmle.code.cpp.ir.IR
import semmle.code.cpp.security.flowafterfree.FlowAfterFree
import semmle.code.cpp.security.flowafterfree.UseAfterFree
import UseAfterFreeTrace::PathGraph

/**
 * Enhanced: Detect use in a different branch after free
 */
predicate useInDifferentBranch(DeallocationExpr free, Expr use, Variable v) {
  exists(IfStmt ifStmt |
    free.getFreedExpr().(VariableAccess).getTarget() = v and
    use.(VariableAccess).getTarget() = v and
    (
      free.getEnclosingStmt().getParentStmt*() = ifStmt.getThen() and
      use.getEnclosingStmt().getParentStmt*() = ifStmt.getElse()
      or
      free.getEnclosingStmt().getParentStmt*() = ifStmt.getElse() and
      use.getEnclosingStmt().getParentStmt*() = ifStmt.getThen()
    )
  )
}

/**
 * Enhanced: Detect use after conditional free
 */
predicate useAfterConditionalFree(DeallocationExpr free, Expr use, Variable v) {
  exists(IfStmt ifStmt |
    free.getFreedExpr().(VariableAccess).getTarget() = v and
    use.(VariableAccess).getTarget() = v and
    (
      free.getEnclosingStmt().getParentStmt*() = ifStmt.getThen() or
      free.getEnclosingStmt().getParentStmt*() = ifStmt.getElse()
    ) and
    ifStmt.getASuccessor+() = use.getEnclosingStmt()
  )
}

module UseAfterFreeParam implements FlowFromFreeParamSig {
  predicate isSink = isUse/2;
  predicate isExcluded = isExcludedMmFreePageFromMdl/2;
  predicate sourceSinkIsRelated = defaultSourceSinkIsRelated/2;
}

import UseAfterFreeParam

module UseAfterFreeTrace = FlowFromFree<UseAfterFreeParam>;

from UseAfterFreeTrace::PathNode source, UseAfterFreeTrace::PathNode sink, DeallocationExpr dealloc, string detail
where
  UseAfterFreeTrace::flowPath(source, sink) and
  isFree(source.getNode(), _, _, dealloc) and
  (
    exists(Variable v, Expr use |
      sink.getNode().asExpr() = use and
      (
        useInDifferentBranch(dealloc, use, v) and detail = " (use in different branch)"
        or
        useAfterConditionalFree(dealloc, use, v) and detail = " (use after conditional free)"
      )
    )
    or
    not exists(Variable v, Expr use |
      sink.getNode().asExpr() = use and
      (useInDifferentBranch(dealloc, use, v) or useAfterConditionalFree(dealloc, use, v))
    ) and detail = ""
  )
select sink.getNode(), source, sink, "Memory may have been previously freed by $@" + detail + ".", dealloc,
  dealloc.toString()
'''


ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE = '''
/**
 * @name Potential double free (Enhanced + filters)
 * @description Enhanced version with better control-flow sensitivity for conditional frees, plus pluggable special-case filters.
 * @kind path-problem
 * @precision high
 * @id cpp/double-free-enhanced-filtered
 * @problem.severity warning
 * @security-severity 9.3
 * @tags reliability
 *       security
 *       external/cwe/cwe-415
 */

import cpp
import semmle.code.cpp.dataflow.new.DataFlow
import semmle.code.cpp.security.flowafterfree.FlowAfterFree
import DoubleFree::PathGraph

predicate isFree(DataFlow::Node n, Expr e) { isFree(_, n, e, _) }

/**
 * Enhanced: Detect free in both branches of an if statement
 */
predicate freeInBothBranches(DeallocationExpr free1, DeallocationExpr free2, Variable v) {
  exists(IfStmt ifStmt |
    free1.getFreedExpr().(VariableAccess).getTarget() = v and
    free2.getFreedExpr().(VariableAccess).getTarget() = v and
    free1.getEnclosingStmt().getParentStmt*() = ifStmt.getThen() and
    free2.getEnclosingStmt().getParentStmt*() = ifStmt.getElse() and
    exists(DeallocationExpr free3 |
      free3.getFreedExpr().(VariableAccess).getTarget() = v and
      ifStmt.getASuccessor+() = free3
    )
  )
}

/**
 * Enhanced: Detect free in a loop that may execute multiple times
 */
predicate freeInLoop(DeallocationExpr free, Variable v) {
  exists(Loop loop |
    free.getFreedExpr().(VariableAccess).getTarget() = v and
    free.getEnclosingStmt().getParentStmt*() = loop.getStmt() and
    not exists(AllocationExpr alloc |
      alloc.getEnclosingStmt().getParentStmt*() = loop.getStmt() and
      exists(AssignExpr assign |
        assign.getRValue() = alloc and
        assign.getLValue().(VariableAccess).getTarget() = v
      )
    )
  )
}

module DoubleFreeParam implements FlowFromFreeParamSig {
  predicate isSink = isFree/2;
  predicate isExcluded = isExcludedMmFreePageFromMdl/2;
  predicate sourceSinkIsRelated = defaultSourceSinkIsRelated/2;
}

module DoubleFree = FlowFromFree<DoubleFreeParam>;

from DoubleFree::PathNode source, DoubleFree::PathNode sink, DeallocationExpr dealloc, Expr e2, string detail
where
  DoubleFree::flowPath(source, sink) and
  isFree(source.getNode(), _, _, dealloc) and
  isFree(sink.getNode(), e2) and
  not dfFiltered(dealloc, sink.getNode(), e2) and
  (
    exists(Variable v |
      freeInBothBranches(dealloc, _, v) and detail = " (freed in both branches)"
      or
      freeInLoop(dealloc, v) and detail = " (freed in loop)"
    )
    or
    not exists(Variable v | freeInBothBranches(dealloc, _, v) or freeInLoop(dealloc, v)) and detail = ""
  )
select sink.getNode(), source, sink,
  "Memory pointed to by $@ may already have been freed by $@" + detail + ".",
  e2, e2.toString(), dealloc, dealloc.toString()
'''

ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE = '''
/**
 * @name Memory is never freed (Enhanced)
 * @description Enhanced version with better error-path detection.
 *              Detects memory allocated but never freed, including in error paths.
 * @kind problem
 * @id cpp/memory-never-freed-enhanced-filtered
 * @problem.severity warning
 * @security-severity 7.5
 * @tags efficiency
 *       security
 *       external/cwe/cwe-401
 */
import MemoryFreed

/**
 * Additional predicate to detect allocations in error-handling branches
 * that may be missed by the standard query.
 */
predicate allocInErrorBranch(AllocationExpr alloc) {
  exists(IfStmt ifStmt, BlockStmt block |
    // Allocation is in a conditional branch
    (block = ifStmt.getThen() or block = ifStmt.getElse()) and
    alloc.getEnclosingStmt().getParentStmt*() = block
  )
}

/**
 * Detects allocation in a loop where the pointer may be overwritten.
 */
predicate allocInLoopMayOverwrite(AllocationExpr alloc) {
  exists(Loop loop, Variable v, AssignExpr assign |
    alloc.getEnclosingStmt().getParentStmt*() = loop.getStmt() and
    assign.getRValue() = alloc and
    assign.getLValue().(VariableAccess).getTarget() = v and
    // Same variable assigned elsewhere in the loop
    exists(AssignExpr other |
      other != assign and
      other.getLValue().(VariableAccess).getTarget() = v and
      other.getEnclosingStmt().getParentStmt*() = loop.getStmt()
    )
  )
}

from AllocationExpr alloc, string reason
where
  alloc.requiresDealloc() and
  not allocMayBeFreed(alloc) and
  not leakFiltered(alloc) and
  (
    // Standard case
    not allocInErrorBranch(alloc) and not allocInLoopMayOverwrite(alloc) and reason = "never freed"
    or
    // Error branch case
    allocInErrorBranch(alloc) and reason = "never freed (in conditional branch)"
    or
    // Loop overwrite case
    allocInLoopMayOverwrite(alloc) and reason = "may leak in loop (pointer overwritten)"
  )
select alloc, "This memory is " + reason + "."
'''

ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE = '''/**
 * @name Memory may not be freed (Enhanced)
 * @description Enhanced version that better tracks memory through early returns.
 * @kind problem
 * @id cpp/memory-may-not-be-freed-enhanced-filtered
 * @problem.severity warning
 * @security-severity 7.5
 * @tags efficiency
 *       security
 *       external/cwe/cwe-401
 */

import MemoryFreed
import semmle.code.cpp.controlflow.StackVariableReachability

predicate mayCallFunction(Expr call, Function f) {
  call.(FunctionCall).getTarget() = f or
  call.(VariableCall).getVariable().getAnAssignedValue().getAChild*().(FunctionAccess).getTarget() = f
}

predicate allocCallOrIndirect(Expr e) {
  e.(AllocationExpr).requiresDealloc() and
  allocMayBeFreed(e)
  or
  exists(ReturnStmt rtn |
    mayCallFunction(e, rtn.getEnclosingFunction()) and
    (
      allocCallOrIndirect(rtn.getExpr())
      or
      exists(StackVariable v |
        v = rtn.getExpr().(VariableAccess).getTarget() and
        allocCallOrIndirect(v.getAnAssignedValue()) and
        not assignedToFieldOrGlobal(v, _)
      )
    )
  )
}

predicate verifiedRealloc(FunctionCall reallocCall, Variable v, ControlFlowNode verified) {
  reallocCall.(AllocationExpr).getReallocPtr() = v.getAnAccess() and
  (
    exists(Variable newV, ControlFlowNode node |
      newV.getAnAssignedValue() = reallocCall and
      node.(AnalysedExpr).getNonNullSuccessor(newV) = verified and
      newV != v
    )
    or
    reallocCall.(AllocationExpr).getReallocPtr().getValue() = "0" and
    verified = reallocCall
  )
}

predicate freeCallOrIndirect(ControlFlowNode n, Variable v) {
  n.(DeallocationExpr).getFreedExpr() = v.getAnAccess() and
  not exists(n.(AllocationExpr).getReallocPtr())
  or
  verifiedRealloc(_, v, n)
  or
  exists(FunctionCall midcall, Function mid, int arg |
    n.(Call).getArgument(arg) = v.getAnAccess() and
    mayCallFunction(n, mid) and
    midcall.getEnclosingFunction() = mid and
    freeCallOrIndirect(midcall, mid.getParameter(arg))
  )
}

predicate allocationDefinition(StackVariable v, ControlFlowNode def) {
  exists(Expr expr | exprDefinition(v, def, expr) and allocCallOrIndirect(expr))
}

class AllocVariableReachability extends StackVariableReachabilityWithReassignment {
  AllocVariableReachability() { this = "AllocVariableReachability" }

  override predicate isSourceActual(ControlFlowNode node, StackVariable v) {
    allocationDefinition(v, node)
  }

  override predicate isSinkActual(ControlFlowNode node, StackVariable v) {
    exists(node.(AnalysedExpr).getNullSuccessor(v)) or
    freeCallOrIndirect(node, v) or
    assignedToFieldOrGlobal(v, node) or
    v.getFunction() = node.(ReturnStmt).getEnclosingFunction()
  }

  override predicate isBarrier(ControlFlowNode node, StackVariable v) { definitionBarrier(v, node) }
}

predicate allocatedVariableReaches(StackVariable v, ControlFlowNode def, ControlFlowNode node) {
  exists(AllocVariableReachability r |
    r.reachesTo(def, _, node, v)
    or
    r.isSource(def, v) and node = def
  )
}

class AllocReachability extends StackVariableReachabilityExt {
  AllocReachability() { this = "AllocReachability" }

  override predicate isSource(ControlFlowNode node, StackVariable v) {
    allocationDefinition(v, node)
  }

  override predicate isSink(ControlFlowNode node, StackVariable v) {
    v.getFunction() = node.(ReturnStmt).getEnclosingFunction()
  }

  override predicate isBarrier(
    ControlFlowNode source, ControlFlowNode node, ControlFlowNode next, StackVariable v
  ) {
    this.isSource(source, v) and
    next = node.getASuccessor() and
    exists(StackVariable v0 | allocatedVariableReaches(v0, source, node) |
      node.(AnalysedExpr).getNullSuccessor(v0) = next or
      freeCallOrIndirect(node, v0) or
      assignedToFieldOrGlobal(v0, node)
    )
  }
}

predicate allocationReaches(ControlFlowNode def, ControlFlowNode node) {
  exists(AllocReachability r | r.reaches(def, _, node))
}

predicate assignedToFieldOrGlobal(StackVariable v, Expr e) {
  e.(Assignment).getRValue() = v.getAnAccess() and
  not e.(Assignment).getLValue().(VariableAccess).getTarget() instanceof StackVariable
  or
  exists(Expr midExpr, Function mid, int arg |
    e.(FunctionCall).getArgument(arg) = v.getAnAccess() and
    mayCallFunction(e, mid) and
    midExpr.getEnclosingFunction() = mid and
    assignedToFieldOrGlobal(mid.getParameter(arg), midExpr)
  )
  or
  e.(ConstructorFieldInit).getExpr() = v.getAnAccess()
}

/**
 * Enhanced: Check if return is in an error-handling context
 */
predicate isErrorReturn(ReturnStmt ret) {
  exists(IfStmt ifStmt |
    ret.getEnclosingStmt().getParentStmt*() = ifStmt.getThen() and
    (
      ifStmt.getCondition().(EqualityOperation).getAnOperand() instanceof NullValue
      or
      exists(ComparisonOperation cmp | cmp = ifStmt.getCondition() |
        cmp.getAnOperand().(Literal).getValue().toInt() <= 0
      )
    )
  )
}

from ControlFlowNode def, ReturnStmt ret, string context
where
  allocationReaches(def, ret) and
  not exists(StackVariable v |
    allocatedVariableReaches(v, def, ret) and
    ret.getAChild*() = v.getAnAccess()
  ) and
  not mayNotBeFreedFiltered(def) and
  (
    isErrorReturn(ret) and context = "error return"
    or
    not isErrorReturn(ret) and context = "exit point"
  )
select def, "This memory allocation may not be released at $@ (" + context + ").", ret, "this " + context
'''


ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE = '''
/**
 * @name Potential use after free (Enhanced)
 * @description Enhanced version with better detection of uses in different control flow paths.
 * @kind path-problem
 * @precision high
 * @id cpp/use-after-free-enhanced-filtered
 * @problem.severity warning
 * @security-severity 9.3
 * @tags reliability
 *       security
 *       external/cwe/cwe-416
 */

import cpp
import semmle.code.cpp.dataflow.new.DataFlow
import semmle.code.cpp.ir.IR
import semmle.code.cpp.security.flowafterfree.FlowAfterFree
import semmle.code.cpp.security.flowafterfree.UseAfterFree
import UseAfterFreeTrace::PathGraph

/**
 * Enhanced: Detect use in a different branch after free
 */
predicate useInDifferentBranch(DeallocationExpr free, Expr use, Variable v) {
  exists(IfStmt ifStmt |
    free.getFreedExpr().(VariableAccess).getTarget() = v and
    use.(VariableAccess).getTarget() = v and
    (
      free.getEnclosingStmt().getParentStmt*() = ifStmt.getThen() and
      use.getEnclosingStmt().getParentStmt*() = ifStmt.getElse()
      or
      free.getEnclosingStmt().getParentStmt*() = ifStmt.getElse() and
      use.getEnclosingStmt().getParentStmt*() = ifStmt.getThen()
    )
  )
}

/**
 * Enhanced: Detect use after conditional free
 */
predicate useAfterConditionalFree(DeallocationExpr free, Expr use, Variable v) {
  exists(IfStmt ifStmt |
    free.getFreedExpr().(VariableAccess).getTarget() = v and
    use.(VariableAccess).getTarget() = v and
    (
      free.getEnclosingStmt().getParentStmt*() = ifStmt.getThen() or
      free.getEnclosingStmt().getParentStmt*() = ifStmt.getElse()
    ) and
    ifStmt.getASuccessor+() = use.getEnclosingStmt()
  )
}

module UseAfterFreeParam implements FlowFromFreeParamSig {
  predicate isSink = isUse/2;
  predicate isExcluded = isExcludedMmFreePageFromMdl/2;
  predicate sourceSinkIsRelated = defaultSourceSinkIsRelated/2;
}

import UseAfterFreeParam

module UseAfterFreeTrace = FlowFromFree<UseAfterFreeParam>;

from UseAfterFreeTrace::PathNode source, UseAfterFreeTrace::PathNode sink, DeallocationExpr dealloc, string detail
where
  UseAfterFreeTrace::flowPath(source, sink) and
  isFree(source.getNode(), _, _, dealloc) and
  not uafFiltered(dealloc, sink.getNode()) and
  (
    exists(Variable v, Expr use |
      sink.getNode().asExpr() = use and
      (
        useInDifferentBranch(dealloc, use, v) and detail = " (use in different branch)"
        or
        useAfterConditionalFree(dealloc, use, v) and detail = " (use after conditional free)"
      )
    )
    or
    not exists(Variable v, Expr use |
      sink.getNode().asExpr() = use and
      (useInDifferentBranch(dealloc, use, v) or useAfterConditionalFree(dealloc, use, v))
    ) and detail = ""
  )
select sink.getNode(), source, sink, "Memory may have been previously freed by $@" + detail + ".", dealloc,
  dealloc.toString()
'''



ENHANCED_DOUBLE_FREE_FILTERED = ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE
ENHANCED_MEMORY_NEVER_FREED_FILTERED = ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE
ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED = ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE
ENHANCED_USE_AFTER_FREE_FILTERED = ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE

# Map query names to enhanced versions
ENHANCED_QUERIES = {
    "MemoryNeverFreed": ENHANCED_MEMORY_NEVER_FREED,
    "MemoryMayNotBeFreed": ENHANCED_MEMORY_MAY_NOT_BE_FREED,
    "DoubleFree": ENHANCED_DOUBLE_FREE,
    "UseAfterFree": ENHANCED_USE_AFTER_FREE,
}

# Base (built-in) filtered query text; may be extended with LLM-generated filters at runtime.
ENHANCED_QUERIES_FILTERED = {
    "MemoryNeverFreed": ENHANCED_MEMORY_NEVER_FREED_FILTERED,
    "MemoryMayNotBeFreed": ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED,
    "DoubleFree": ENHANCED_DOUBLE_FREE_FILTERED,
    "UseAfterFree": ENHANCED_USE_AFTER_FREE_FILTERED,
}


class CodeQLAnalyzer:
    """CodeQL analyzer with model injection and enhanced queries."""

    def __init__(
        self,
        binary: str = "codeql",
        timeout: int = 600,
        codeql_dir: Path | None = None,
        cpp_queries_dir: Path | None = None,
        reuse_db: bool = True,
    ):
        self.binary = binary
        self.timeout = timeout
        self.codeql_dir = codeql_dir
        self.cpp_queries_dir = cpp_queries_dir
        self.reuse_db = reuse_db
        self._injected_files: list[Path] = []
        # Dynamic enhanced filtered query for DoubleFree; may be overridden
        # with LLM-generated filters at analysis time.
        self._double_free_filtered_query: str | None = None
        self._memory_never_freed_filtered_query: str | None = None
        self._memory_may_not_be_freed_filtered_query: str | None = None
        self._use_after_free_filtered_query: str | None = None

    def analyze(
        self,
        project_path: Path,
        hints: HintSet | None = None,
        issue_types: list[MemoryIssueType] | None = None,
        use_enhanced_queries: bool = True,
        custom_queries: CustomQuerySet | None = None,
    ) -> list[Warning]:
        """
        Run CodeQL analysis.

        Note: If you want to "filter out" known false positives (e.g., _dictClear(d,0) then _dictClear(d,1)),
        you MUST do it in *your pipeline* after SARIF parsing. A CodeQL "query" cannot remove another query's results.
        This function now supports that via CustomQuerySet filter predicates.

        Args:
            project_path: Project to analyze
            hints: Allocator/deallocator hints
            issue_types: Which issue types to map/return (currently parsed from SARIF ruleId)
            use_enhanced_queries: If True, write and run hardcoded enhanced queries; otherwise run standard CodeQL queries
            custom_queries: Custom CodeQL queries generated from hints (optional). Includes filter predicates.
        """
        output_dir = Path(tempfile.mkdtemp())
        results_path = output_dir / "results.sarif"
        query_files_to_cleanup: list[Path] = []

        try:
            # Step 1: Database
            logger.info("Step 1: Getting CodeQL database...")
            db_path = self._get_or_create_database(project_path)
            if not db_path:
                return []

            # Step 2: Inject models
            if hints and getattr(hints, "hints", None):
                logger.info("Step 2: Injecting custom memory models...")
                self._inject_models(hints)
                self._verify_injected_models()

            # Step 3: Prepare queries
            queries: list[Path] = []

            # Build dynamic enhanced filtered queries by merging the hard-coded
            # templates with any LLM-generated filter snippets in custom_queries.
            if custom_queries and len(custom_queries) > 0:
                logger.info("Merging custom filters into enhanced templates...")
                self._double_free_filtered_query = self._build_double_free_filtered_with_custom_filters(
                    custom_queries
                )
                self._memory_never_freed_filtered_query = self._build_memory_never_freed_filtered_with_custom_filters(
                    custom_queries
                )
                self._memory_may_not_be_freed_filtered_query = self._build_memory_may_not_be_freed_filtered_with_custom_filters(
                        custom_queries
                )
                
                self._use_after_free_filtered_query = self._build_use_after_free_filtered_with_custom_filters(
                    custom_queries
                )

            # Add enhanced queries if enabled
            if use_enhanced_queries:
                logger.info("Step 3b: Writing enhanced queries...")
                enhanced_query_files_list, enhanced_cleanup, enhanced_count, filtered_count = self._prepare_enhanced_queries()
                queries.extend(enhanced_query_files_list)
                query_files_to_cleanup.extend(enhanced_cleanup)
                logger.info(f"  Added {enhanced_count} enhanced query files and {filtered_count} enhanced filtered query files")

            # If no custom or enhanced queries, use standard queries
            if not queries:
                logger.info("Step 3c: Using standard queries...")
                queries_to_run: list[Path] | None = None
            else:
                queries_to_run = queries

            # Step 4: Run analysis
            logger.info("Step 4: Running CodeQL analysis...")
            success = self._run_queries(db_path, results_path, queries_to_run)

            if not success:
                return []

            warnings = self._parse_sarif(results_path, project_path)
            return warnings

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            self._cleanup_models()
            # Clean up query files
            for f in query_files_to_cleanup:
                try:
                    if isinstance(f, Path) and f.exists():
                        f.unlink()
                except Exception:
                    pass
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)


    # -------------------------------------------------------------------------
    # Custom query writing (unchanged except: ignore entries that are "suppression-only")
    # -------------------------------------------------------------------------

    def _insert_imports_into_template(self, template: str, required_imports: set[str]) -> str:
        """
        Insert additional imports into a CodeQL query template.
        
        Args:
            template: The CodeQL query template string
            required_imports: Set of import module paths (without 'import' keyword)
        
        Returns:
            Template with imports inserted after existing imports
        """
        if not required_imports:
            return template
        
        # Sort imports for consistent output
        import_lines = sorted(required_imports)
        additional_imports = "\n".join(f"import {imp}" for imp in import_lines)
        
        # Find the last import statement in the template
        # Look for common import patterns
        last_import_pos = -1
        last_import_end = -1
        
        # Try to find the last import line
        import_patterns = [
            "import semmle.code.cpp.dataflow.new.DataFlow",
            "import semmle.code.cpp.controlflow.StackVariableReachability",
            "import MemoryFreed",
            "import cpp",
        ]
        
        for pattern in import_patterns:
            pos = template.rfind(pattern)
            if pos != -1:
                # Find the end of this import line
                end_pos = template.find("\n", pos)
                if end_pos != -1:
                    if pos > last_import_pos:
                        last_import_pos = pos
                        last_import_end = end_pos
        
        if last_import_pos != -1 and last_import_end != -1:
            # Find the blank line after imports (if any)
            blank_line_pos = template.find("\n\n", last_import_end)
            if blank_line_pos != -1:
                # Insert after the blank line
                return (
                    template[:blank_line_pos + 1] +
                    additional_imports + "\n" +
                    template[blank_line_pos + 1:]
                )
            else:
                # No blank line, insert after the last import with a newline
                return (
                    template[:last_import_end] +
                    "\n" + additional_imports + "\n" +
                    template[last_import_end:]
                )
        
        # Fallback: prepend imports (shouldn't happen with proper templates)
        return additional_imports + "\n\n" + template

    def _write_custom_queries(self, custom_queries: CustomQuerySet) -> tuple[list[Path], list[Path]]:
        """
        Legacy method - no longer writes custom queries since query_code is removed.
        Filters are now integrated into enhanced filtered queries instead.
        """
        # No longer needed - filters are integrated into enhanced filtered queries
        return [], []

    def _build_double_free_filtered_with_custom_filters(
        self,
        custom_queries: CustomQuerySet,
    ) -> str:
        """
        Build the final enhanced+filtered DoubleFree query text by combining:
          - the hard-coded ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE, and
          - all LLM-generated *double_free_filter* snippets from CustomQuerySet.

        This matches the JSON structure:

          "queries": {
            "<func>": {
              "double_free_filter": {
              },
              "use_after_free_filter": { ... },
              "memory_never_freed_filter": { ... },
              "memory_may_not_be_freed_filter": { ... },
              ...
            }
          }
        """
        predicates_snippets: list[str] = []
        use_exprs: list[str] = []
        required_imports: set[str] = set()

        cq_json = custom_queries.to_json()
        for func_name, qinfo in (cq_json.get("queries") or {}).items():
            if not isinstance(qinfo, dict):
                continue
            df_block = qinfo.get("double_free_filter") or {}
            if not isinstance(df_block, dict):
                continue

            predicates_code = (df_block.get("predicates_code", "") or "").strip()
            use_expr = (df_block.get("use_expr", "") or "").strip()
            
            # Collect required imports (use set to avoid duplicates)
            imports = df_block.get("required_imports", [])
            if isinstance(imports, list):
                for imp in imports:
                    if isinstance(imp, str) and imp.strip():
                        required_imports.add(imp.strip())
            
            # Skip if validation failed or block is empty
            validated = df_block.get("validated")
            if validated is False:
                continue  # Skip invalid blocks
            if not predicates_code and not use_expr:
                continue  # Skip empty blocks

            if predicates_code:
                predicates_snippets.append(
                    f"// ---- LLM double-free predicates for {func_name} ----\n{predicates_code}\n"
                )
            if use_expr:
                use_exprs.append(use_expr)

        # Only build query if we have validated filters
        if use_exprs or predicates_snippets:
            preds_part = "\n\n".join(predicates_snippets) if predicates_snippets else ""
            # Build dfFiltered aggregator from all use_exprs
            body = " or\n  ".join(use_exprs) if use_exprs else "false"
            df_agg = (
                "predicate dfFiltered(DeallocationExpr srcDealloc, "
                "DataFlow::Node sinkNode, Expr sinkFreedExpr) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + df_agg
            return merged

        # No validated filters: return None to skip creating the filtered query file
        return None

    def _build_memory_never_freed_filtered_with_custom_filters(
        self,
        custom_queries: CustomQuerySet,
    ) -> str:
        """
        Build the final enhanced+filtered MemoryNeverFreed query text by combining:
          - ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE, and
          - all LLM-generated *memory_never_freed_filter* snippets from CustomQuerySet.
        """
        predicates_snippets: list[str] = []
        use_exprs: list[str] = []
        required_imports: set[str] = set()

        cq_json = custom_queries.to_json()
        for func_name, qinfo in (cq_json.get("queries") or {}).items():
            if not isinstance(qinfo, dict):
                continue
            # Try new field first, fall back to legacy memory_leak_filter for backward compatibility
            leak_block = qinfo.get("memory_never_freed_filter") or qinfo.get("memory_leak_filter") or {}
            if not isinstance(leak_block, dict):
                continue

            predicates_code = (leak_block.get("predicates_code", "") or "").strip()
            use_expr = (leak_block.get("use_expr", "") or "").strip()
            
            # Collect required imports (use set to avoid duplicates)
            imports = leak_block.get("required_imports", [])
            if isinstance(imports, list):
                for imp in imports:
                    if isinstance(imp, str) and imp.strip():
                        required_imports.add(imp.strip())
            
            # Skip if validation failed or block is empty
            validated = leak_block.get("validated")
            if validated is False:
                continue  # Skip invalid blocks
            if not predicates_code and not use_expr:
                continue  # Skip empty blocks

            if predicates_code:
                predicates_snippets.append(
                    f"// ---- LLM memory-leak predicates for {func_name} ----\n{predicates_code}\n"
                )
            if use_expr:
                use_exprs.append(use_expr)

        # Only build query if we have validated filters
        if use_exprs or predicates_snippets:
            preds_part = "\n\n".join(predicates_snippets) if predicates_snippets else ""
            body = " or\n  ".join(use_exprs) if use_exprs else "false"
            leak_agg = (
                "predicate leakFiltered(AllocationExpr alloc) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + leak_agg
            return merged

        # No validated filters: return None to skip creating the filtered query file
        return None

    def _build_memory_may_not_be_freed_filtered_with_custom_filters(
        self,
        custom_queries: CustomQuerySet,
    ) -> str:
        """
        Build the final enhanced+filtered MemoryMayNotBeFreed query text by combining:
          - ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE, and
          - all LLM-generated *memory_may_not_be_freed_filter* snippets from CustomQuerySet.
        """
        predicates_snippets: list[str] = []
        use_exprs: list[str] = []
        required_imports: set[str] = set()

        cq_json = custom_queries.to_json()
        for func_name, qinfo in (cq_json.get("queries") or {}).items():
            if not isinstance(qinfo, dict):
                continue
            # Try new field first, fall back to legacy memory_leak_filter for backward compatibility
            leak_block = qinfo.get("memory_may_not_be_freed_filter") or qinfo.get("memory_leak_filter") or {}
            if not isinstance(leak_block, dict):
                continue

            predicates_code = (leak_block.get("predicates_code", "") or "").strip()
            use_expr = (leak_block.get("use_expr", "") or "").strip()
            
            # Collect required imports (use set to avoid duplicates)
            imports = leak_block.get("required_imports", [])
            if isinstance(imports, list):
                for imp in imports:
                    if isinstance(imp, str) and imp.strip():
                        required_imports.add(imp.strip())
            
            # Skip if validation failed or block is empty
            validated = leak_block.get("validated")
            if validated is False:
                continue  # Skip invalid blocks
            if not predicates_code and not use_expr:
                continue  # Skip empty blocks

            if predicates_code:
                predicates_snippets.append(
                    f"// ---- LLM memory-leak predicates for {func_name} ----\n{predicates_code}\n"
                )
            if use_expr:
                use_exprs.append(use_expr)

        # Only build query if we have validated filters
        if use_exprs or predicates_snippets:
            preds_part = "\n\n".join(predicates_snippets) if predicates_snippets else ""
            body = " or\n  ".join(use_exprs) if use_exprs else "false"
            leak_agg = (
                "predicate mayNotBeFreedFiltered(ControlFlowNode def) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + leak_agg
            return merged

        # No validated filters: return None to skip creating the filtered query file
        return None

    def _build_use_after_free_filtered_with_custom_filters(
        self,
        custom_queries: CustomQuerySet,
    ) -> str:
        """
        Build the final enhanced+filtered UseAfterFree query text by combining:
          - ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE, and
          - all LLM-generated *use_after_free_filter* snippets from CustomQuerySet.
        """
        predicates_snippets: list[str] = []
        use_exprs: list[str] = []
        required_imports: set[str] = set()

        cq_json = custom_queries.to_json()
        for func_name, qinfo in (cq_json.get("queries") or {}).items():
            if not isinstance(qinfo, dict):
                continue
            uaf_block = qinfo.get("use_after_free_filter") or {}
            if not isinstance(uaf_block, dict):
                continue

            predicates_code = (uaf_block.get("predicates_code", "") or "").strip()
            use_expr = (uaf_block.get("use_expr", "") or "").strip()
            
            # Collect required imports (use set to avoid duplicates)
            imports = uaf_block.get("required_imports", [])
            if isinstance(imports, list):
                for imp in imports:
                    if isinstance(imp, str) and imp.strip():
                        required_imports.add(imp.strip())
            
            # Skip if validation failed or block is empty
            validated = uaf_block.get("validated")
            if validated is False:
                continue  # Skip invalid blocks
            if not predicates_code and not use_expr:
                continue  # Skip empty blocks

            if predicates_code:
                predicates_snippets.append(
                    f"// ---- LLM use-after-free predicates for {func_name} ----\n{predicates_code}\n"
                )
            if use_expr:
                use_exprs.append(use_expr)

        # Only build query if we have validated filters
        if use_exprs or predicates_snippets:
            preds_part = "\n\n".join(predicates_snippets) if predicates_snippets else ""
            body = " or\n  ".join(use_exprs) if use_exprs else "false"
            uaf_agg = (
                "predicate uafFiltered(DeallocationExpr dealloc, DataFlow::Node sinkNode) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + uaf_agg
            return merged

        # No validated filters: return None to skip creating the filtered query file
        return None

    def _prepare_enhanced_queries(self) -> tuple[list[Path], list[Path], int, int]:
        """
        Write enhanced queries next to original queries.
        
        Returns:
            Tuple of (query_files, cleanup_files, enhanced_count, filtered_count)
            - enhanced_count: number of enhanced query files written
            - filtered_count: number of enhanced filtered query files written
        """
        query_files: list[Path] = []
        cleanup_files: list[Path] = []
        enhanced_count = 0
        filtered_count = 0

        for query_ref in MEMORY_QUERIES:
            query_name = query_ref.split("/")[-1].replace(".ql", "")
            if query_name in ENHANCED_QUERIES:
              original_path = self._find_query_file_path(query_ref)
              if not original_path:
                  logger.warning(f"Could not find {query_ref}")
                  continue

              enhanced_path = original_path.parent / f"{query_name}_enhanced.ql"
              enhanced_path.write_text(ENHANCED_QUERIES[query_name])

              query_files.append(enhanced_path)
              cleanup_files.append(enhanced_path)
              enhanced_count += 1
              logger.info(f"  Written: {enhanced_path}")
              
            if query_name in ENHANCED_QUERIES_FILTERED:
              original_path = self._find_query_file_path(query_ref)
              if not original_path:
                  logger.warning(f"Could not find {query_ref}")
                  continue

              enhanced_path = original_path.parent / f"{query_name}_enhanced_filtered.ql"

              # Prefer dynamically built filtered query (with LLM filters) if available.
              if query_name == "DoubleFree":
                  qtext = self._double_free_filtered_query
                  label = "DoubleFree"
              elif query_name == "MemoryNeverFreed":
                  qtext = self._memory_never_freed_filtered_query
                  label = "MemoryNeverFreed"
              elif query_name == "MemoryMayNotBeFreed":
                  qtext = self._memory_may_not_be_freed_filtered_query
                  label = "MemoryMayNotBeFreed"
              elif query_name == "UseAfterFree":
                  qtext = self._use_after_free_filtered_query
                  label = "UseAfterFree"
              else:
                  qtext = ENHANCED_QUERIES_FILTERED[query_name]
                  label = query_name

              # Skip writing the file if there are no validated filters (qtext is None)
              if qtext is None:
                  logger.info(f"  Skipped: {enhanced_path} (no validated filters)")
                  continue

              # No validation here - filters are already validated at generation time in llm_client.py
              enhanced_path.write_text(qtext)

              query_files.append(enhanced_path)
              cleanup_files.append(enhanced_path)
              filtered_count += 1
              logger.info(f"  Written: {enhanced_path}")
              
            if query_name not in ENHANCED_QUERIES and query_name not in ENHANCED_QUERIES_FILTERED:
              logger.warning(f"No enhanced/filtered version for {query_name}")
              continue

        return query_files, cleanup_files, enhanced_count, filtered_count

    def _find_query_file_path(self, query_ref: str) -> Optional[Path]:
        """Find the actual path of a query file."""
        if ":" not in query_ref:
            return None

        _, query_path = query_ref.split(":", 1)

        # Determine base directory
        if self.cpp_queries_dir:
            base = self.cpp_queries_dir
        elif self.codeql_dir:
            base = self.codeql_dir / "packages" / "codeql" / "cpp-queries"
        else:
            base = Path.home() / ".codeql" / "packages" / "codeql" / "cpp-queries"

        if not base.exists():
            return None

        versions = sorted([d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")], reverse=True)
        if not versions:
            return None

        query_file = versions[0] / query_path
        return query_file if query_file.exists() else None

    def _run_queries(self, db_path: Path, results_path: Path, query_files: list[Path] | None = None) -> bool:
        """Run CodeQL queries."""
        if query_files:
            query_args = [str(q) for q in query_files]
        else:
            query_args = MEMORY_QUERIES

        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "--format=sarif-latest",
            f"--output={results_path}",
            "--download",
        ] + query_args

        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)

        if result.returncode != 0:
            logger.warning(f"CodeQL stderr: {result.stderr.decode(errors='replace')[:1000]}")

        return results_path.exists()

    # =========================================================================
    # Database Management
    # =========================================================================

    def _get_db_path(self, project_path: Path) -> Path:
        return project_path / ".codeql-db"

    def _get_or_create_database(self, project_path: Path) -> Optional[Path]:
        db_path = self._get_db_path(project_path)

        if not self.reuse_db and db_path.exists():
            logger.debug(f"Removing existing CodeQL database (reuse_db=False): {db_path}")
            shutil.rmtree(db_path, ignore_errors=True)

        if self.reuse_db and self._is_valid_database(db_path):
            logger.debug(f"Reusing existing CodeQL database: {db_path}")
            if self._finalize_database(db_path):
                return db_path
            logger.debug(f"Existing database invalid or could not be finalized, recreating: {db_path}")
            shutil.rmtree(db_path, ignore_errors=True)

        if db_path.exists():
            logger.debug(f"Cleaning up old CodeQL database before recreation: {db_path}")
            shutil.rmtree(db_path, ignore_errors=True)

        logger.debug(f"Creating new CodeQL database at: {db_path}")
        if self._create_database(project_path, db_path):
            logger.debug(f"Successfully created CodeQL database: {db_path}")
            return db_path
        logger.error(f"Failed to create CodeQL database at: {db_path}")
        return None

    def _is_valid_database(self, db_path: Path) -> bool:
        if not db_path.exists():
            return False
        metadata = db_path / "codeql-database.yml"
        if not metadata.exists():
            return False
        try:
            with open(metadata) as f:
                info = yaml.safe_load(f)
            return info.get("primaryLanguage", "").lower() in ("cpp", "c++")
        except Exception:
            return False

    def _finalize_database(self, db_path: Path) -> bool:
        try:
            metadata = db_path / "codeql-database.yml"
            with open(metadata) as f:
                if yaml.safe_load(f).get("finalised", False):
                    return True
        except Exception:
            pass

        result = subprocess.run(
            [self.binary, "database", "finalize", str(db_path)],
            timeout=self.timeout, capture_output=True, text=True
        )
        return result.returncode == 0 or "already finalized" in (result.stderr or "").lower()

    def _create_database(self, project_path: Path, db_path: Path) -> bool:
        config = self._load_build_config(project_path)

        if config:
            pre_build = config.get("prepare_for_build")
            if pre_build:
                logger.info(f"Running pre-build commands for {project_path}")
                self._run_commands(project_path, pre_build)

        compile_commands = project_path / "compile_commands.json"
        if compile_commands.exists():
            logger.info(f"Using compile_commands.json for database creation: {compile_commands}")
            cmd = [
                self.binary, "database", "create", str(db_path),
                f"--source-root={project_path}",
                "--language=cpp", "--overwrite",
                f"--compilation-database={compile_commands}"
            ]
        else:
            build_cmd = None
            if config and "build_command" in config:
                bc = config["build_command"]
                build_cmd = bc.get("command") if isinstance(bc, dict) else bc
            if not build_cmd:
                build_cmd = self._detect_build_command(project_path)
            if not build_cmd:
                logger.error("Could not determine build command")
                return False

            logger.info(f"Using build command for CodeQL database creation: {build_cmd}")
            cmd = [
                self.binary, "database", "create", str(db_path),
                f"--source-root={project_path}",
                "--language=cpp", "--overwrite",
                "--command", build_cmd
            ]

        logger.info(f"Running CodeQL database create: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True, cwd=str(project_path))
        if result.returncode != 0:
            logger.error(f"Database creation failed: {result.stderr.decode(errors='replace')[:500]}")
            return False

        return self._finalize_database(db_path)

    def _load_build_config(self, project_path: Path) -> Optional[dict]:
        config_file = Path(__file__).parent.parent.parent / "proj_build_command.json"
        if not config_file.exists():
            return None
        try:
            with open(config_file) as f:
                return json.load(f).get(project_path.name)
        except Exception:
            return None

    def _run_commands(self, cwd: Path, commands) -> None:
        if isinstance(commands, str):
            commands = [{"command": c.strip(), "can_error": True} for c in commands.split("&&")]
        elif isinstance(commands, list):
            commands = [
                {"command": c.get("command", c) if isinstance(c, dict) else c,
                 "can_error": c.get("can_error", True) if isinstance(c, dict) else True}
                for c in commands
            ]

        for cmd_info in commands:
            if cmd_info["command"]:
                subprocess.run(cmd_info["command"], shell=True, cwd=str(cwd),
                               timeout=self.timeout, capture_output=True)

    def _detect_build_command(self, project_path: Path) -> Optional[str]:
        if (project_path / "Makefile").exists() and which("make"):
            return "make"
        if (project_path / "CMakeLists.txt").exists() and which("cmake"):
            build_dir = project_path / "build"
            build_dir.mkdir(exist_ok=True)
            subprocess.run(["cmake", ".."], cwd=str(build_dir), capture_output=True, timeout=120)
            if (build_dir / "Makefile").exists():
                return "make -C build"

        sources = list(project_path.rglob("*.c")) + list(project_path.rglob("*.cpp"))
        if sources:
            files = " ".join(str(f.relative_to(project_path)) for f in sources[:30])
            return f"clang -I. -c -fsyntax-only {files}"
        return None

    # =========================================================================
    # Model Injection (BUGFIX: allocators type mismatch)
    # =========================================================================

    def _get_ext_dir(self) -> Path:
        if self.cpp_queries_dir:
            base = self.cpp_queries_dir
        elif self.codeql_dir:
            base = self.codeql_dir / "packages" / "codeql" / "cpp-queries"
        else:
            base = Path.home() / ".codeql" / "packages" / "codeql" / "cpp-queries"

        versions = sorted([d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")], reverse=True)
        if not versions:
            raise FileNotFoundError(f"No cpp-queries versions in {base}")

        cpp_all = versions[0] / ".codeql" / "libraries" / "codeql" / "cpp-all"
        cpp_versions = sorted([d for d in cpp_all.iterdir() if d.is_dir()], reverse=True)
        if not cpp_versions:
            raise FileNotFoundError(f"No cpp-all versions in {cpp_all}")

        return cpp_all / cpp_versions[0] / "ext"

    def _inject_models(self, hints: HintSet) -> None:
        """
        BUGFIX:
          - Your original code did: allocators = [f for f in hints.get_allocators() ...]
            but _build_alloc_yaml expects list[(name, idx)].
          - This version accepts BOTH:
              hints.get_allocators() returning ["malloc_like", ...]  OR [("malloc_like",-1), ...]
            and normalizes.
        """
        ext_dir = self._get_ext_dir()

        raw_allocs = [a for a in (hints.get_allocators() or []) if a not in ("main", "_main", "")]
        allocators: list[tuple[str, int]] = []
        for a in raw_allocs:
            if isinstance(a, (list, tuple)) and len(a) == 2:
                allocators.append((str(a[0]), int(a[1])))
            else:
                # default: treat as return-value allocator
                allocators.append((str(a), -1))

        deallocators = [(f, i) for f, i in (hints.get_deallocators() or []) if f not in ("main", "_main", "")]

        if allocators:
            path = ext_dir / "allocation" / "hint.allocation.model.yml"
            path.parent.mkdir(parents=True, exist_ok=True)
            yaml_content = self._build_alloc_yaml(allocators)
            path.write_text(yaml_content)
            self._injected_files.append(path)
            logger.info(f"Injected {len(allocators)} allocators")

        if deallocators:
            path = ext_dir / "deallocation" / "hint.deallocation.model.yml"
            path.parent.mkdir(parents=True, exist_ok=True)
            yaml_content = self._build_dealloc_yaml(deallocators)
            path.write_text(yaml_content)
            self._injected_files.append(path)
            logger.info(f"Injected {len(deallocators)} deallocators")

    def _build_alloc_yaml(self, funcs: list[tuple[str, int]]) -> str:
        """Build allocation function model YAML.

        Args:
            funcs: List of (function_name, arg_index) where:
                - arg_index = -1 means return value ("ReturnValue")
                - arg_index >= 0 means output parameter at that index
        """
        lines = [
            "extensions:",
            "  - addsTo:",
            "      pack: codeql/cpp-all",
            "      extensible: allocationFunctionModel",
            "    data:"
        ]
        for name, idx in funcs:
            # CodeQL uses "ReturnValue" for return, or index string for out-parameter
            output = "ReturnValue" if idx == -1 else str(idx)
            lines.append(f'      - ["", "", False, "{name}", "{output}", "", "", True]')
        return "\n".join(lines) + "\n"

    def _build_dealloc_yaml(self, funcs: list[tuple[str, int]]) -> str:
        """Build deallocation function model YAML.

        Args:
            funcs: List of (function_name, arg_index) where arg_index is the
                0-based index of the freed argument.
        """
        lines = [
            "extensions:",
            "  - addsTo:",
            "      pack: codeql/cpp-all",
            "      extensible: deallocationFunctionModel",
            "    data:"
        ]
        for name, idx in funcs:
            lines.append(f'      - ["", "", False, "{name}", "{idx}"]')
        return "\n".join(lines) + "\n"

    def _verify_injected_models(self) -> None:
        result = subprocess.run(
            [self.binary, "resolve", "extensions", "codeql/cpp-queries"],
            capture_output=True, timeout=60
        )
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout.decode())
                all_files = [e.get("file", "") for v in data.get("data", {}).values() for e in v]
                for f in self._injected_files:
                    found = any(str(f) in x for x in all_files)
                    logger.info(f"  {'✓' if found else '✗'} {f.name}")
            except Exception:
                pass

    def _cleanup_models(self) -> None:
        for f in self._injected_files:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass
        self._injected_files.clear()

    # =========================================================================
    # SARIF Parsing (unchanged; consider moving suppression here if preferred)
    # =========================================================================

    def _parse_sarif(self, sarif_path: Path, project_path: Path) -> list[Warning]:
        if not sarif_path.exists():
            return []

        warnings: list[Warning] = []
        try:
            data = json.loads(sarif_path.read_text())
            for run in data.get("runs", []):
                for result in run.get("results", []):
                    rule_id = (result.get("ruleId", "") or "").lower()

                    # Default issue type; will refine based on rule_id
                    issue_type = MemoryIssueType.MEMORY_LEAK

                    # First try exact match on rule_id
                    if rule_id in CODEQL_ISSUE_MAP:
                        issue_type = CODEQL_ISSUE_MAP[rule_id]
                    else:
                        # Fallback: substring match (for older CodeQL ids), preferring longer keys first
                        for key, val in sorted(CODEQL_ISSUE_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
                            if key in rule_id:
                                issue_type = val
                                break

                    locs = result.get("locations", []) or []
                    if not locs:
                        continue

                    loc0 = (locs[0].get("physicalLocation", {}) or {})
                    file_path = (loc0.get("artifactLocation", {}) or {}).get("uri", "") or ""
                    line = (loc0.get("region", {}) or {}).get("startLine", 0) or 0

                    alloc_site = self._extract_allocation_site(result, run)
                    trace = self._extract_trace(result, run)

                    warnings.append(Warning(
                        file_path=file_path,
                        line_number=line,
                        function_name=self._find_function(file_path, line, project_path),
                        warning_type=rule_id,
                        message=(result.get("message", {}) or {}).get("text", "") or "",
                        issue_type=issue_type,
                        allocation_site=alloc_site,
                        trace=trace
                    ))

        except Exception as e:
            logger.error(f"SARIF parse error: {e}")

        return warnings

    def _find_function(self, file_path: str, line: int, project_path: Path) -> str:
        if not file_path or not line:
            return ""
        try:
            full_path = project_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
            if not full_path.exists():
                return ""
            from src.tree_sitter_parser import CodeParser
            for name, info in CodeParser().parse_file(full_path).items():
                if info.start_line <= line <= info.end_line:
                    return name
        except Exception:
            pass
        return ""

    def _sarif_loc_to_str(self, loc: dict) -> str:
        """
        Convert a SARIF location-like object to "file:line" string.
        Accepts either:
          - result["locations"][i]
          - result["relatedLocations"][i]
          - threadFlowLocation["location"]
        """
        if not loc:
            return ""
        phys = loc.get("physicalLocation") or {}
        art = phys.get("artifactLocation") or {}
        uri = art.get("uri") or ""
        region = phys.get("region") or {}
        line = region.get("startLine") or 0
        if not uri:
            return ""
        return f"{uri}:{line}" if line else uri

    def _resolve_related_locations(self, result: dict, run: dict) -> list[dict]:
        related = result.get("relatedLocations") or []
        if not related:
            return []

        resolved = []
        pool = (run.get("relatedLocations") or [])
        for rl in related:
            if isinstance(rl, dict) and ("physicalLocation" in rl or "location" in rl):
                resolved.append(rl.get("location") if "location" in rl else rl)
                continue
            if isinstance(rl, int) and 0 <= rl < len(pool):
                resolved.append(pool[rl])
                continue
            if isinstance(rl, dict):
                idx = rl.get("id")
                if isinstance(idx, int) and 0 <= idx < len(pool):
                    resolved.append(pool[idx])
        return resolved

    def _extract_allocation_site(self, result: dict, run: dict) -> str:
        locs = result.get("locations") or []
        if locs:
            s = self._sarif_loc_to_str(locs[0].get("location") or locs[0])
            if s:
                return s
        rls = self._resolve_related_locations(result, run)
        for rl in rls:
            s = self._sarif_loc_to_str(rl)
            if s:
                return s
        return ""

    def _extract_trace(self, result: dict, run: dict) -> list[str]:
        trace: list[str] = []
        codeflows = result.get("codeFlows") or []
        for cf in codeflows:
            for tf in (cf.get("threadFlows") or []):
                for tfl in (tf.get("locations") or []):
                    loc = tfl.get("location") or {}
                    s = self._sarif_loc_to_str(loc)
                    if s:
                        trace.append(s)

        if trace:
            seen = set()
            out = []
            for x in trace:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        # 2) fallback: related locations
        rls = self._resolve_related_locations(result, run)
        for rl in rls:
            s = self._sarif_loc_to_str(rl)
            if s:
                trace.append(s)

        locs = result.get("locations") or []
        if locs:
            primary = self._sarif_loc_to_str(locs[0].get("location") or locs[0])
            if primary:
                trace = [primary] + [x for x in trace if x != primary]

        seen = set()
        out = []
        for x in trace:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
