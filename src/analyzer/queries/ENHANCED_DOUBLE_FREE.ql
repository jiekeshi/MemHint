/**
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