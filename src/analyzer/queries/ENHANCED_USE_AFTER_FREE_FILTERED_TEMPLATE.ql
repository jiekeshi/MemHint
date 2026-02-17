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