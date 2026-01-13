"""Hint's Pipeline - Memory Safety Analysis with LLM-Assisted Annotations and Code Slicing and Merging

Architecture:
1. Parse source code with tree-sitter
2. LLM generates memory safety annotations
3. Z3 constraint solver validates annotations
4. Merge relevant code into single file (with dependency tracking) TODO: needs to check
5. Static analyzer (CodeQL/Infer) analyzes merged code
6. Map results back to original locations
"""

import json
import logging
import tempfile
import shutil
from pathlib import Path

from tqdm import tqdm

from src.core.models import (
    FunctionInfo, Annotation, AnnotationType, AnnotationSet,
    Warning, Evidence, AnalysisResult, MemoryIssueType
)
from src.tree_sitter_parser import CodeParser
from src.llm_client import LLMClient, AnnotationGenerator
from src.symbolic.validator import AnnotationValidator, analyze_with_slicing
from src.analyzer.adapters import create_analyzer, CodeQLAnalyzer
from src.analyzer.merger import CodeMerger, OriginalLocation

logger = logging.getLogger(__name__)


class Pipeline:
    """HINT analysis pipeline with code merging and dependency slicing.

    Key features:
    1. Cross-file dependency tracking (recursive callees, types)
    2. Single-file compilation for fast analysis
    3. LLM patches merged code for compilability
    4. Results mapped back to original locations
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.5-pro",
        analyzer_type: str = "codeql",
        max_iterations: int = 3,
        issue_types: list[MemoryIssueType] = None
    ):
        self.parser = CodeParser()
        self.llm = LLMClient(api_key=api_key, model=model)
        self.generator = AnnotationGenerator(self.llm)
        self.analyzer = create_analyzer(analyzer_type)
        self.constraint_validator = AnnotationValidator()
        self.max_iterations = max_iterations

        # Default to all issue types
        self.issue_types = issue_types or [
            MemoryIssueType.MEMORY_LEAK,
            MemoryIssueType.DOUBLE_FREE,
            MemoryIssueType.USE_AFTER_FREE,
            MemoryIssueType.NULL_DEREFERENCE,
        ]

        self.functions: dict[str, FunctionInfo] = {}
        self.annotations = AnnotationSet()
        self.conflicts: list[str] = []

    def analyze(self, project_path: Path, output_dir: Path = None) -> AnalysisResult:
        """Run analysis pipeline with code merging.

        Pipeline:
        1. Parse source code
        2. LLM generates annotations (function properties + bug hints)
        3. Z3 validates annotations
        4. Merge relevant code with dependency tracking
        5. Static analyzer analyzes merged code
        6. Map results back to original locations
        7. Z3 filters infeasible paths
        """
        project_path = Path(project_path).resolve()
        output_dir = output_dir or Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Analyzing {project_path}")
        logger.info(f"Issue types: {[t.name for t in self.issue_types]}")

        # Phase 1: Parse source code
        logger.info("Phase 1: Parsing source code...")
        self.functions = self.parser.parse_project(project_path)
        logger.info(f"Found {len(self.functions)} functions")

        annotations_file = output_dir / "annotations.json"
        if annotations_file.exists():
            logger.info(f"annotations.json found in {output_dir}, reusing existing annotations")
            self.annotations = self._load_annotations_from_file(annotations_file)
        else:
            # Phase 2: Generate annotations with LLM
            logger.info("Phase 2: Generating annotations with LLM...")
            self._generate_annotations()

            func_annotations = sum(
                1 for anns in self.annotations.annotations.values()
                for ann in anns if not ann.is_bug_annotation()
            )
            bug_hints = sum(
                1 for anns in self.annotations.annotations.values()
                for ann in anns if ann.is_bug_annotation()
            )
            logger.info(f"Generated {func_annotations} function annotations, {bug_hints} bug hints")

            # Phase 3: Validate with Z3
            logger.info("Phase 3: Validating annotations with Z3...")
            self._validate_annotations_with_z3()

        # Phase 4: Merge relevant code
        logger.info("Phase 4: Merging code with dependency tracking...")
        merger = CodeMerger(llm_client=self.llm)
        merged = merger.merge(self.annotations, self.functions, project_path)

        logger.info(f"Merged {len(merged.included_functions)} functions, {len(merged.included_types)} types")

        # Save merged code
        merged_file = output_dir / "merged_analysis.c"
        merged_file.write_text(merged.code)
        logger.info(f"Wrote merged code: {merged_file}")

        # Save location map
        location_map_file = output_dir / "location_map.json"
        map_data = {
            str(line): {
                "file": loc.file_path,
                "line": loc.start_line,
                "function": loc.function_name,
            }
            for line, loc in merged.location_map.items()
        }
        location_map_file.write_text(json.dumps(map_data, indent=2))

        # Phase 5: Run analyzer on merged code
        analyzer_name = type(self.analyzer).__name__
        logger.info(f"Phase 5: Running {analyzer_name} on merged code...")

        # Create temp directory with merged file
        temp_dir = Path(tempfile.mkdtemp())
        shutil.copy(merged_file, temp_dir / "merged_analysis.c")

        try:
            warnings = self.analyzer.analyze(temp_dir, self.annotations, self.issue_types)
            logger.info(f"{analyzer_name} found {len(warnings)} warnings")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Phase 6: Map results to original locations
        logger.info("Phase 6: Mapping to original locations...")
        mapped_warnings = []
        for warning in warnings:
            original_loc = CodeMerger.map_result_to_original(
                warning.line_number, merged.location_map
            )
            if original_loc:
                mapped_warnings.append(Warning(
                    file_path=original_loc.file_path,
                    line_number=original_loc.start_line,
                    function_name=original_loc.function_name or warning.function_name,
                    warning_type=warning.warning_type,
                    message=warning.message,
                    issue_type=warning.issue_type,
                    trace=warning.trace,
                ))
            else:
                mapped_warnings.append(warning)

        # Phase 7: Filter with Z3
        logger.info("Phase 7: Z3 path feasibility filtering...")
        confirmed_bugs = self._filter_warnings_with_z3(mapped_warnings, project_path)
        logger.info(f"Confirmed {len(confirmed_bugs)} bugs after Z3 filtering")

        # Export results
        self._export_analyzer_config(output_dir)
        self._export_final_output(output_dir, confirmed_bugs)

        return AnalysisResult(
            confirmed_bugs=confirmed_bugs,
            final_annotations=self.annotations,
            iterations=1,
            spurious_filtered=len(warnings) - len(confirmed_bugs),
        )

    def _load_annotations_from_file(self, file_path: Path) -> AnnotationSet:
        """Load annotations.json produced by a previous run."""
        data = json.loads(Path(file_path).read_text())
        loaded = AnnotationSet()

        # Function annotations
        for func_name, anns in data.get("functions", {}).items():
            for ann in anns:
                ann_type_name = ann.get("type")
                if not ann_type_name:
                    continue
                try:
                    ann_type = AnnotationType[ann_type_name]
                except KeyError:
                    logger.warning(f"Unknown annotation type in file: {ann_type_name}")
                    continue
                loaded.add(Annotation(
                    function_name=func_name,
                    annotation_type=ann_type,
                    target=ann.get("target", ""),
                    reason=ann.get("reason", "")
                ))

        # Bug hints
        for hint in data.get("bug_hints", []):
            ann_type_name = hint.get("type")
            func_name = hint.get("function", "")
            if not ann_type_name or not func_name:
                continue
            try:
                ann_type = AnnotationType[ann_type_name]
            except KeyError:
                logger.warning(f"Unknown bug hint type in file: {ann_type_name}")
                continue
            loaded.add(Annotation(
                function_name=func_name,
                annotation_type=ann_type,
                target=hint.get("target", ""),
                line_number=hint.get("line"),
                confidence=hint.get("confidence", 1.0),
                reason=hint.get("reason", "")
            ))

        logger.info(f"Loaded annotations from {file_path}")
        return loaded

    def _generate_annotations(self) -> None:
        """Generate comprehensive memory safety annotations using LLM."""
        self.annotations = AnnotationSet()

        for func_name, func in tqdm(self.functions.items(), desc="Generating annotations"):
            annotations = self.generator.generate(func, self.functions)
            for ann in annotations:
                self.annotations.add(ann)

    def _validate_annotations_with_z3(self) -> None:
        """Validate annotations using Z3 constraint solver.

        This is the core CEGAR component - it checks if annotations
        are consistent with the code structure using constraint solving.
        """
        validated, conflicts = self.constraint_validator.validate_annotations(
            self.annotations, self.functions
        )

        self.annotations = validated
        self.conflicts.extend(conflicts)

        if conflicts:
            logger.info(f"Z3 validation found {len(conflicts)} conflicts:")
            for conflict in conflicts[:5]:
                logger.info(f"  - {conflict}")
            if len(conflicts) > 5:
                logger.info(f"  ... and {len(conflicts) - 5} more")

    def _validate_annotations_with_llm(self) -> None:
        """Additional LLM-based semantic validation."""
        to_remove = []

        for func_name, anns in tqdm(
            list(self.annotations.annotations.items()),
            desc="LLM validation"
        ):
            if func_name not in self.functions:
                continue
            func = self.functions[func_name]

            for ann in anns:
                # Validate allocation annotations (most prone to false positives)
                if ann.annotation_type in (AnnotationType.ALLOC_SOURCE, AnnotationType.ARRAY_ALLOC):
                    is_valid, reason = self.generator.validate(func, ann)
                    if not is_valid:
                        logger.debug(f"LLM removing invalid annotation {func_name}: {reason}")
                        to_remove.append((func_name, ann.annotation_type))
                        self.conflicts.append(f"LLM CONFLICT: {func_name} - {reason}")

        for func_name, ann_type in to_remove:
            self.annotations.remove(func_name, ann_type)

        if to_remove:
            logger.info(f"LLM validation removed {len(to_remove)} annotations")

    def _filter_warnings_with_z3(
        self, warnings: list[Warning], project_path: Path
    ) -> list[Evidence]:
        """Filter static analyzer warnings using Z3 constraint solving.

        This reduces false positives by checking path feasibility:
        1. Extract minimal slice relevant to the bug
        2. Check feasibility with Z3 on the slice
        3. Keep only feasible warnings

        The static analyzer results are authoritative -
        we only filter out provably infeasible paths.
        """
        confirmed = []
        filtered_count = 0

        # Get known allocation functions from annotations
        alloc_funcs = set(self.annotations.get_alloc_functions())
        # Always include standard allocators
        alloc_funcs.update({"malloc", "calloc", "realloc", "strdup"})

        for warning in warnings:
            func = self.functions.get(warning.function_name)

            if not func:
                # Can't analyze without function code - keep warning
                confirmed.append(Evidence(
                    warning=warning,
                    concrete_trace=["Unable to analyze: function code not found"],
                    leak_point=f"line {warning.line_number}",
                    suggested_fix=self._suggest_fix(warning),
                ))
                continue

            # Try Z3 sliced analysis for path feasibility
            try:
                slice_results = analyze_with_slicing(func.code, alloc_funcs)
                issue_key = warning.issue_type.name

                if issue_key in slice_results:
                    results = slice_results[issue_key]

                    if not results:
                        # Z3 proves no feasible path exists - filter
                        logger.debug(f"Z3 filtered: {warning.function_name}:{warning.line_number} - {issue_key}")
                        self.conflicts.append(
                            f"Z3 FILTERED: {warning.function_name}:{warning.line_number} - infeasible {issue_key}"
                        )
                        filtered_count += 1
                        continue

                    # Z3 confirms feasible path exists
                    trace = [f"Z3 confirmed: feasible path exists"]
                    for r in results:
                        if r.condition:
                            trace.append(f"Condition: {r.condition}")

                    confirmed.append(Evidence(
                        warning=warning,
                        concrete_trace=trace,
                        leak_point=f"line {warning.line_number}",
                        suggested_fix=self._suggest_fix(warning),
                    ))
                else:
                    # Issue type not handled by Z3 slicer - keep warning
                    confirmed.append(Evidence(
                        warning=warning,
                        concrete_trace=[],
                        leak_point=f"line {warning.line_number}",
                        suggested_fix=self._suggest_fix(warning),
                    ))

            except Exception as e:
                # Z3 analysis failed - keep warning (conservative)
                logger.debug(f"Z3 analysis failed for {warning.function_name}: {e}")
                confirmed.append(Evidence(
                    warning=warning,
                    concrete_trace=[f"Z3 analysis skipped: {e}"],
                    leak_point=f"line {warning.line_number}",
                    suggested_fix=self._suggest_fix(warning),
                ))

        if filtered_count:
            logger.info(f"Z3 filtered {filtered_count} infeasible warnings")

        return confirmed

    def _export_analyzer_config(self, output_dir: Path) -> None:
        """Export annotations to analyzer-specific formats."""
        from src.analyzer.adapters import CodeQLAnalyzer, InferAnalyzer

        # CodeQL model extension
        codeql_model = output_dir / "codeql_model.yml"
        with open(codeql_model, "w") as f:
            f.write(self.annotations.to_codeql_model())

        # CodeQL custom query
        if isinstance(self.analyzer, CodeQLAnalyzer):
            query_file = output_dir / "memory_safety.ql"
            self.analyzer.generate_custom_query(self.annotations, query_file)
        else:
            # Generate CodeQL query anyway for reference
            codeql = CodeQLAnalyzer()
            query_file = output_dir / "memory_safety.ql"
            codeql.generate_custom_query(self.annotations, query_file)

        # Infer config (if using Infer)
        if isinstance(self.analyzer, InferAnalyzer):
            # Infer config is generated in project directory during analysis
            pass

        # Export annotations as JSON (generic format)
        ann_file = output_dir / "annotations.json"
        with open(ann_file, "w") as f:
            json.dump(self.annotations.to_json(), f, indent=2)

        # Export conflicts for debugging
        if self.conflicts:
            with open(output_dir / "validation_conflicts.txt", "w") as f:
                f.write("\n".join(self.conflicts))

        # Export LLM bug hints (for reference only, NOT merged with results)
        bug_hints = []
        for func_name, anns in self.annotations.annotations.items():
            for ann in anns:
                if ann.is_bug_annotation():
                    bug_hints.append({
                        "function": func_name,
                        "type": ann.annotation_type.name,
                        "variable": ann.target,
                        "line": ann.line_number,
                        "confidence": ann.confidence,
                        "reason": ann.reason,
                    })

        logger.info(f"Exported analyzer configs to {output_dir}")

    def _export_final_output(self, output_dir: Path, bugs: list[Evidence]) -> None:
        """Export final results."""
        # Bug report grouped by type
        bugs_by_type = {}
        for bug in bugs:
            t = bug.warning.issue_type.name
            if t not in bugs_by_type:
                bugs_by_type[t] = []
            bugs_by_type[t].append({
                "file": bug.warning.file_path,
                "line": bug.warning.line_number,
                "function": bug.warning.function_name,
                "message": bug.warning.message,
                "leak_point": bug.leak_point,
                "suggested_fix": bug.suggested_fix,
                "trace": bug.concrete_trace,
            })

        with open(output_dir / "memory_safety_bugs.json", "w") as f:
            json.dump(bugs_by_type, f, indent=2)

        # Human-readable summary
        summary_lines = ["# HINT Memory Safety Analysis Report", ""]
        summary_lines.append(f"**Total Issues Found**: {len(bugs)}")
        summary_lines.append(f"**Annotations Generated**: {len(self.annotations.annotations)}")
        summary_lines.append(f"**Conflicts Resolved**: {len(self.conflicts)}")
        summary_lines.append("")

        for issue_type, type_bugs in bugs_by_type.items():
            summary_lines.append(f"## {issue_type} ({len(type_bugs)} issues)")
            for i, bug in enumerate(type_bugs, 1):
                summary_lines.append(f"\n### Issue {i}")
                summary_lines.append(f"- **File**: {bug['file']}:{bug['line']}")
                summary_lines.append(f"- **Function**: {bug['function']}")
                summary_lines.append(f"- **Message**: {bug['message']}")
                summary_lines.append(f"- **Fix**: {bug['suggested_fix']}")
            summary_lines.append("")

        with open(output_dir / "report.md", "w") as f:
            f.write("\n".join(summary_lines))

        logger.info(f"Final output saved to {output_dir}")

    def _suggest_fix(self, warning: Warning) -> str:
        """Generate fix suggestion for a warning."""
        fixes = {
            MemoryIssueType.MEMORY_LEAK: "Free allocated memory before function exit",
            MemoryIssueType.DOUBLE_FREE: "Remove duplicate free() or add null guard",
            MemoryIssueType.USE_AFTER_FREE: "Use memory before freeing, not after",
            MemoryIssueType.NULL_DEREFERENCE: "Add null check before pointer use",
            MemoryIssueType.BUFFER_OVERFLOW: "Validate buffer bounds before access",
        }
        return fixes.get(warning.issue_type, "Review and fix memory safety issue")