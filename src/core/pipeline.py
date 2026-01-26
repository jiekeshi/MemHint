"""HINT Pipeline - Memory Safety Analysis with LLM-Assisted Hints.

Pipeline flow:
1. Parse: Extract functions from source code (tree-sitter)
2. Generate: LLM generates memory safety hints (ALLOCATOR, DEALLOCATOR)
3. Validate Hints: Z3 validates hints are consistent with code
4. Analyze: CodeQL scans with custom model extensions based on hints
5. Filter: Z3 filters false positives by path feasibility

The key insight: LLM identifies FUNCTION SEMANTICS (hints), not bugs.
CodeQL uses these hints to find bugs. Z3 filters impossible scenarios.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

from src.core.models import (
    FunctionInfo, Hint, HintType, HintSet,
    Warning, Evidence, AnalysisResult, MemoryIssueType
)
from src.tree_sitter_parser import CodeParser
from src.llm_client import HintGenerator, LLMClient
from src.symbolic.z3_solver import HintValidator, WarningValidator
from src.analyzer.adapters import CodeQLAnalyzer

logger = logging.getLogger(__name__)

class Pipeline:
    """HINT analysis pipeline.

    Pipeline stages:
    1. Parse: Extract function information using tree-sitter
    2. Generate: LLM generates memory safety hints
    3. Validate Hints: Z3 validates hints are consistent
    4. Analyze: CodeQL scans with hint-based model extensions
    5. Filter: Z3 filters infeasible warning paths
    """

    def __init__(
        self,
        api_key: str = None,  # Kept for compatibility, not used with Vertex AI
        model: str = "gemini-2.5-pro",
        analyzer_type: str = "codeql",
        issue_types: list[MemoryIssueType] = None,
        use_merge: bool = False,  # Kept for compatibility
        codeql_dir: Path = None,
        cpp_queries_dir: Path = None,
        reuse_db: bool = True,
        skip_hint_injection: bool = False,
        use_enhanced_queries: bool = True,
    ):
        """Initialize pipeline.

        Args:
            api_key: Unused (Vertex AI uses service account)
            model: LLM model name
            analyzer_type: "codeql" or "infer"
            issue_types: Which bug types to detect (default: all)
            use_merge: Whether to merge code (kept for compatibility)
            codeql_dir: Optional custom CodeQL directory path (default: ~/.codeql)
            cpp_queries_dir: Optional direct path to cpp-queries directory
            reuse_db: Whether to reuse existing CodeQL database (default: True)
        """
        # Components
        self.parser = CodeParser()
        self.llm_client = LLMClient(model=model)
        self.hint_generator = HintGenerator(self.llm_client)
        self.hint_validator = HintValidator()
        self.warning_validator = None  # Initialized with known allocators
        self.analyzer = CodeQLAnalyzer(codeql_dir=codeql_dir, cpp_queries_dir=cpp_queries_dir, reuse_db=reuse_db)

        # Configuration
        self.use_merge = use_merge
        self.skip_hint_injection = skip_hint_injection
        self.use_enhanced_queries = use_enhanced_queries
        self.issue_types = issue_types or [
            MemoryIssueType.MEMORY_LEAK,
            MemoryIssueType.DOUBLE_FREE,
            MemoryIssueType.USE_AFTER_FREE,
        ]

        # State
        self.functions: dict[str, FunctionInfo] = {}
        self.macro_names: set[str] = set()  # Track which function names are from macros
        self.hints = HintSet()
        self.conflicts: list[str] = []
        self.validated_hints_count: int = 0  # Track number of hints that passed Z3 validation

    def analyze(
        self,
        project_path: Path,
        output_dir: Path = None,
        single_sources: list[Path] | None = None,
    ) -> AnalysisResult:
        """Run the full analysis pipeline.

        Pipeline:
        1. Parse source code
        2. LLM generates hints (function semantics)
        3. Z3 validates hints
        4. CodeQL scans with hints as model extensions
        5. Z3 filters infeasible warning paths

        Args:
            project_path: Path to C/C++ project
            output_dir: Where to save results

        Returns:
            AnalysisResult with confirmed bugs and hints
        """
        project_path = Path(project_path).resolve()
        output_dir = output_dir or Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        single_sources = [Path(p).resolve() for p in single_sources] if single_sources else None

        logger.info(f"Analyzing {project_path}")
        logger.info(f"Bug types: {[t.name for t in self.issue_types]}")

        # =====================================================================
        # Phase 1: Parse source code
        # =====================================================================
        logger.info("Phase 1: Parsing source code...")
        if single_sources is not None:
            logger.info(f"  Single-source mode: parsing {len(single_sources)} file(s)")
            for single_source in single_sources:
                if not single_source.exists():
                    logger.error(f"Single source file not found: {single_source}")
                    return AnalysisResult(
                        confirmed_bugs=[],
                        hints=HintSet(),
                        iterations=0,
                        spurious_filtered=0,
                    )

            # Parse all specified files and merge results
            self.functions = {}
            self.macro_names = set()
            for single_source in single_sources:
                logger.info(f"    Parsing {single_source}")
                file_functions, file_macros = self.parser.parse_file_with_macros(single_source, preprocess=False)
                for func_name, info in file_functions.items():
                    # Ensure file_path is set for downstream components
                    if not getattr(info, "file_path", None):
                        info.file_path = str(single_source)
                    # Handle function name conflicts by prefixing with file name if needed
                    if func_name in self.functions:
                        logger.warning(f"    Function {func_name} found in multiple files, keeping first occurrence")
                    else:
                        self.functions[func_name] = info
                # Convert function-like macros to FunctionInfo and add to functions
                for macro_name, macro_info in file_macros.items():
                    if macro_info.is_function_like:
                        # Include expansion in code so LLM can see what the macro expands to
                        macro_code = macro_info.code
                        if macro_info.expansion:
                            macro_code += f"\n// Expands to: {macro_info.expansion}"
                        
                        func_info = FunctionInfo(
                            name=macro_name,
                            code=macro_code,
                            file_path=macro_info.file_path or str(single_source),
                            start_line=macro_info.start_line,
                            end_line=macro_info.end_line,
                            arg_names=macro_info.arg_names,
                            arg_types=[""] * len(macro_info.arg_names),
                            return_type="",
                            callees=set(),
                            callers=set(),
                            return_expressions={macro_info.expansion} if macro_info.expansion else set(),
                        )
                        if macro_name not in self.functions:
                            self.functions[macro_name] = func_info
                            self.macro_names.add(macro_name)
                        else:
                            logger.warning(f"    Macro/Function {macro_name} found in multiple files, keeping first occurrence")

            # Best-effort call graph resolution within this subset
            try:
                # parse_project normally does this; call it explicitly here.
                self.parser._resolve_calls(self.functions)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"  Warning: could not resolve calls in single-source mode: {e}")
        else:
            self.functions = self.parser.parse_project(project_path)
            self.macro_names = set()  # Full project mode doesn't extract macros yet
        
        # Count functions vs macros
        num_functions = len(self.functions) - len(self.macro_names)
        num_macros = len(self.macro_names)
        if num_macros > 0:
            logger.info(f"  Found {num_functions} functions and {num_macros} macros (total: {len(self.functions)})")
        else:
            logger.info(f"  Found {num_functions} functions")
        # Log which files/functions will be scanned
        for func_name, func_info in self.functions.items():
            file_hint = getattr(func_info, "file_path", "unknown")
            if func_name in self.macro_names:
                logger.debug(f"    Scan target macro: {func_name} (file: {file_hint})")
            else:
                logger.debug(f"    Scan target function: {func_name} (file: {file_hint})")

        if not self.functions:
            logger.warning("No functions found")
            return AnalysisResult(
                confirmed_bugs=[],
                hints=HintSet(),
                iterations=0,
                spurious_filtered=0,
            )

        if self.skip_hint_injection:
            logger.info("Hint injection disabled: skipping hint generation/validation (phases 2-3).")
            self.hints = HintSet()
            self.validated_hints_count = 0
            # Export empty cost file
            self._export_costs(output_dir)
        else:
            # Check for cached hints
            hints_file = output_dir / "hints.json"
            if hints_file.exists():
                logger.info(f"  Loading cached hints from {hints_file}")
                self.hints = self._load_hints(hints_file)
                # Count validated hints from cached hints (approximate - all cached hints are considered validated)
                self.validated_hints_count = sum(len(hints) for hints in self.hints.hints.values())
                # Export cost file (will show zeros since no LLM calls were made)
                self._export_costs(output_dir)
            else:
                # =================================================================
                # Phase 2-3: hint generation and validation
                # =================================================================

                all_conflicts = []

                # Initial hint generation
                logger.info("Phase 2: Generating hints with LLM...")
                self.hints = self.hint_generator.generate_hints(self.functions)
                logger.info(f"  {self.hints.summary()}")

                # =================================================================
                # Phase 3: Z3 validates hints
                # =================================================================
                logger.info(f"Phase 3: Validating hints with Z3...")
                self.hints, conflicts = self.hint_validator.validate_hints(
                    self.hints, self.functions
                )
                for conflict in conflicts:
                    all_conflicts.append(conflict)

                if conflicts:
                    logger.info(f" Z3 suggested to remove {len(conflicts)} hints:")
                    for c in conflicts:
                        logger.info(f"    - {c}")

                    conflict_functions = set()
                    for conflict in conflicts:
                        if conflict.startswith("REMOVED "):
                            parts = conflict[8:].split(": ", 1)
                            if len(parts) >= 1:
                                func_hint = parts[0]
                                if "." in func_hint:
                                    func_name = func_hint.split(".")[0]
                                    conflict_functions.add(func_name)

                    logger.info(f"Phase 2: Regenerating hints for {len(conflict_functions)} conflict function(s) ...")
                    # Regenerate hints only for functions that had conflicts in this iteration
                    new_hints = self.hint_generator.regenerate_hints_for_functions(
                        self.functions,
                        conflict_functions,
                        previous_conflicts=conflicts  # Pass current iteration's conflicts for LLM feedback
                    )
                    # Merge new hints with existing validated hints
                    for func_name, func_hints in new_hints.hints.items():
                        for hint in func_hints:
                            self.hints.add(hint)
                    logger.info(f"  After validation: {self.hints.summary()}")
                else:
                    # No conflicts, we're done
                    logger.info(f"  No conflicts found! Hints are consistent.")
                    logger.info(f"  Final hints: {self.hints.summary()}")

                self.conflicts.extend(all_conflicts)

                # Save validated hints
                self._save_hints(hints_file)
                
                # Export cost information right after hint generation completes
                self._export_costs(output_dir)

        # =====================================================================
        # Phase 4: CodeQL analysis with hints
        # =====================================================================
        logger.info("Phase 4: Running CodeQL with hint-based models...")

        # Run CodeQL
        warnings = self.analyzer.analyze(
            project_path,
            self.hints,
            self.issue_types,
            use_enhanced_queries=self.use_enhanced_queries,
        )
        logger.info(f"  CodeQL found {len(warnings)} warnings")

        # =====================================================================
        # Phase 5: Z3 filters false positives
        # =====================================================================
        logger.info("Phase 5: Filtering warnings with Z3 path analysis...")

        # Initialize warning validator with known allocators
        alloc_funcs = set(self.hints.get_allocators())
        alloc_funcs.update({"malloc", "calloc", "realloc", "strdup"})
        free_funcs = set(fn for fn, _ in self.hints.get_deallocators())
        free_funcs.update({"free"})

        self.warning_validator = WarningValidator(alloc_funcs, free_funcs)
        confirmed_warnings, filtered_warnings = self.warning_validator.validate_warnings(
            warnings, self.functions
        )

        logger.info(f"  Confirmed: {len(confirmed_warnings)}, Filtered: {len(filtered_warnings)}")

        # Convert to Evidence
        confirmed_bugs = [
            Evidence(
                warning=w,
                concrete_trace=[],
                root_cause="",
                suggested_fix=self._suggest_fix(w),
                z3_validated=True,
            )
            for w in confirmed_warnings
        ]

        # =====================================================================
        # Export results
        # =====================================================================
        self._export_results(output_dir, confirmed_bugs, filtered_warnings)

        return AnalysisResult(
            confirmed_bugs=confirmed_bugs,
            hints=self.hints,
            iterations=1,
            spurious_filtered=len(filtered_warnings),
        )

    def _load_hints(self, file_path: Path) -> HintSet:
        """Load hints from JSON file."""
        data = json.loads(file_path.read_text())
        return HintSet.from_json(data)

    def _save_hints(self, file_path: Path) -> None:
        """Save hints to JSON file."""
        file_path.write_text(json.dumps(self.hints.to_json(), indent=2))

    def _export_results(
        self,
        output_dir: Path,
        confirmed: list[Evidence],
        filtered: list[Warning],
    ) -> None:
        """Export analysis results."""
        # Confirmed bugs
        bugs_data = {}
        for bug in confirmed:
            t = bug.warning.issue_type.name
            if t not in bugs_data:
                bugs_data[t] = []
            bugs_data[t].append({
                "file": bug.warning.file_path,
                "line": bug.warning.line_number,
                "function": bug.warning.function_name,
                "message": bug.warning.message,
                "suggested_fix": bug.suggested_fix,
            })

        bugs_file = output_dir / "memory_safety_bugs.json"
        bugs_file.write_text(json.dumps(bugs_data, indent=2))

        # Filtered warnings (for debugging)
        filtered_data = [
            {
                "file": w.file_path,
                "line": w.line_number,
                "type": w.issue_type.name,
                "message": w.message,
            }
            for w in filtered
        ]
        filtered_file = output_dir / "filtered_warnings.json"
        filtered_file.write_text(json.dumps(filtered_data, indent=2))

        # Conflicts log
        if self.conflicts:
            conflicts_file = output_dir / "validation_conflicts.txt"
            conflicts_file.write_text("\n".join(self.conflicts))
            logger.info(f"  Validation conflicts saved to {conflicts_file}")

        # Human-readable report
        report = self._generate_report(confirmed)
        report_file = output_dir / "report.md"
        report_file.write_text(report)

        logger.info(f"Results saved to {output_dir}")

    def _export_costs(self, output_dir: Path) -> None:
        """Export LLM cost information to JSON file."""
        # Calculate total number of functions analyzed
        total_functions = len(self.functions)
        
        # Count total validated hints (hints that passed Z3 validation)
        total_validated_hints = self.validated_hints_count
        
        if self.skip_hint_injection:
            # No costs if hints were skipped
            cost_data = {
                "total_llm_cost": 0.0,
                "total_tokens": 0,
                "total_functions": total_functions,
                "total_validated_hints": total_validated_hints,
                "cost_per_validated_hint": 0.0,
                "function_costs": {},
                "hint_costs": {},
            }
            cost_file = output_dir / "llm_costs.json"
            cost_file.write_text(json.dumps(cost_data, indent=2))
            return
        
        # Get cost summary from hint generator
        cost_summary = self.hint_generator.get_cost_summary()
        total_llm_cost = cost_summary["total_cost"]
        
        # Count total output hints (final hints in self.hints)
        total_output_hints = sum(len(hints) for hints in self.hints.hints.values())
        
        # Calculate cost per validated hint (validated hints are from Z3, but we calculate cost per validated hint)
        cost_per_validated_hint = 0.0
        if total_validated_hints > 0:
            cost_per_validated_hint = total_llm_cost / total_validated_hints
        
        # Calculate cost per output hint (final hints in output)
        cost_per_output_hint = 0.0
        if total_output_hints > 0:
            cost_per_output_hint = total_llm_cost / total_output_hints
        
        # Calculate per-hint costs
        # Distribute each function's cost evenly across its hints
        hint_costs = {}
        for func_name, func_hints in self.hints.hints.items():
            func_cost_info = self.hint_generator.function_costs.get(func_name, {})
            func_cost = func_cost_info.get("cost", 0.0)
            num_hints = len(func_hints)
            
            if num_hints > 0:
                cost_per_hint = func_cost / num_hints
                for i, hint in enumerate(func_hints):
                    hint_key = f"{func_name}.{hint.hint_type.name}.{hint.target}"
                    hint_costs[hint_key] = {
                        "function_name": func_name,
                        "hint_type": hint.hint_type.name,
                        "target": hint.target,
                        "cost": cost_per_hint,
                        "tokens": func_cost_info.get("tokens", 0) // num_hints if num_hints > 0 else 0,
                    }
            else:
                # Function had cost but no hints generated
                hint_key = f"{func_name}.no_hints"
                hint_costs[hint_key] = {
                    "function_name": func_name,
                    "hint_type": "NONE",
                    "target": "N/A",
                    "cost": func_cost,
                    "tokens": func_cost_info.get("tokens", 0),
                }
        
        # Build final cost data structure
        cost_data = {
            "total_llm_cost": total_llm_cost,
            "total_tokens": cost_summary["total_tokens"],
            "total_functions": total_functions,
            "total_validated_hints": total_validated_hints,
            "total_output_hints": total_output_hints,
            "cost_per_validated_hint": cost_per_validated_hint,
            "cost_per_output_hint": cost_per_output_hint,
            "function_costs": {
                func_name: {
                    "cost": info["cost"],
                    "tokens": info["tokens"],
                    "prompt_tokens": info.get("prompt_tokens", 0),
                    "completion_tokens": info.get("completion_tokens", 0),
                    "input_cost": info.get("input_cost", 0.0),
                    "output_cost": info.get("output_cost", 0.0),
                }
                for func_name, info in cost_summary["function_costs"].items()
            },
            "hint_costs": hint_costs,
        }
        
        cost_file = output_dir / "llm_costs.json"
        cost_file.write_text(json.dumps(cost_data, indent=2))
        logger.info(f"  LLM cost information saved to {cost_file}")
        logger.info(f"  Total LLM cost: ${cost_data['total_llm_cost']:.6f}")
        logger.info(f"  Total tokens: {cost_data['total_tokens']}")
        logger.info(f"  Total functions analyzed: {cost_data['total_functions']}")
        logger.info(f"  Total validated hints: {cost_data['total_validated_hints']}")
        logger.info(f"  Total output hints: {cost_data['total_output_hints']}")
        logger.info(f"  Cost per validated hint: ${cost_data['cost_per_validated_hint']:.6f}")
        logger.info(f"  Cost per output hint: ${cost_data['cost_per_output_hint']:.6f}")

    def _generate_report(self, bugs: list[Evidence]) -> str:
        """Generate markdown report."""
        lines = [
            "# HINT Memory Safety Analysis Report",
            "",
            "## Summary",
            "",
            f"- **Bugs found**: {len(bugs)}",
            f"- **Functions analyzed**: {len(self.functions)}",
            f"- **Hints generated**: {len(self.hints)}",
            "",
            "## Hints Summary",
            "",
            self.hints.summary(),
            "",
        ]

        # Group bugs by type
        by_type: dict[MemoryIssueType, list[Evidence]] = {}
        for bug in bugs:
            t = bug.warning.issue_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(bug)

        for bug_type, type_bugs in by_type.items():
            lines.append(f"## {bug_type.name} ({len(type_bugs)} issues)")
            lines.append("")
            for i, bug in enumerate(type_bugs, 1):
                lines.append(f"### {i}. {bug.warning.file_path}:{bug.warning.line_number}")
                lines.append(f"- **Function**: {bug.warning.function_name}")
                lines.append(f"- **Message**: {bug.warning.message}")
                lines.append(f"- **Fix**: {bug.suggested_fix}")
                lines.append("")

        return "\n".join(lines)

    def _suggest_fix(self, warning: Warning) -> str:
        """Generate fix suggestion for a warning."""
        fixes = {
            MemoryIssueType.MEMORY_LEAK: "Free allocated memory before function exit",
            MemoryIssueType.DOUBLE_FREE: "Remove duplicate free() or add null guard",
            MemoryIssueType.USE_AFTER_FREE: "Use memory before freeing, not after",
        }
        return fixes.get(warning.issue_type, "Review memory safety issue")