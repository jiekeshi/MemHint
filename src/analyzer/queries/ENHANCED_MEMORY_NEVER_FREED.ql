/**
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