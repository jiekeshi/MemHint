/**
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