"""HINT Pipeline - Memory Safety Analysis with LLM-Assisted Hints.

Pipeline flow:
1. Parse: Extract functions from source code (tree-sitter)
2. Generate: LLM generates memory safety hints (ALLOCATOR, DEALLOCATOR, etc.)
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
        max_iterations: int = 3,
        issue_types: list[MemoryIssueType] = None,
        use_merge: bool = False,  # Kept for compatibility
    ):
        """Initialize pipeline.

        Args:
            api_key: Unused (Vertex AI uses service account)
            model: LLM model name
            analyzer_type: "codeql" or "infer"
            max_iterations: Max CEGAR iterations (for future refinement)
            issue_types: Which bug types to detect (default: all)
            use_merge: Whether to merge code (kept for compatibility)
        """
        # Components
        self.parser = CodeParser()
        self.llm_client = LLMClient(model=model)
        self.hint_generator = HintGenerator(self.llm_client)
        self.hint_validator = HintValidator()
        self.warning_validator = None  # Initialized with known allocators
        self.analyzer = CodeQLAnalyzer()

        # Configuration
        self.max_iterations = max_iterations
        self.use_merge = use_merge
        self.issue_types = issue_types or [
            MemoryIssueType.MEMORY_LEAK,
            MemoryIssueType.DOUBLE_FREE,
            MemoryIssueType.USE_AFTER_FREE,
            MemoryIssueType.NULL_DEREFERENCE,
            MemoryIssueType.BUFFER_OVERFLOW,
        ]

        # State
        self.functions: dict[str, FunctionInfo] = {}
        self.hints = HintSet()
        self.conflicts: list[str] = []

    def analyze(self, project_path: Path, output_dir: Path = None) -> AnalysisResult:
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

        logger.info(f"Analyzing {project_path}")
        logger.info(f"Bug types: {[t.name for t in self.issue_types]}")

        # =====================================================================
        # Phase 1: Parse source code
        # =====================================================================
        logger.info("Phase 1: Parsing source code...")
        self.functions = self.parser.parse_project(project_path)
        logger.info(f"  Found {len(self.functions)} functions")

        if not self.functions:
            logger.warning("No functions found")
            return AnalysisResult(
                confirmed_bugs=[],
                hints=HintSet(),
                iterations=0,
                spurious_filtered=0,
            )

        # Check for cached hints
        hints_file = output_dir / "hints.json"
        if hints_file.exists():
            logger.info(f"  Loading cached hints from {hints_file}")
            self.hints = self._load_hints(hints_file)
        else:
            # =================================================================
            # Phase 2: LLM generates hints
            # =================================================================
            logger.info("Phase 2: Generating hints with LLM...")
            self.hints = self.hint_generator.generate_hints(self.functions)
            logger.info(f"  {self.hints.summary()}")

            # =================================================================
            # Phase 3: Z3 validates hints
            # =================================================================
            logger.info("Phase 3: Validating hints with Z3...")
            self.hints, conflicts = self.hint_validator.validate_hints(
                self.hints, self.functions
            )
            self.conflicts.extend(conflicts)

            if conflicts:
                logger.info(f"  Removed {len(conflicts)} invalid hints:")
                for c in conflicts[:5]:
                    logger.info(f"    - {c}")
                if len(conflicts) > 5:
                    logger.info(f"    ... and {len(conflicts) - 5} more")

            logger.info(f"  After validation: {self.hints.summary()}")

            # Save validated hints
            self._save_hints(hints_file)

        # =====================================================================
        # Phase 4: CodeQL analysis with hints
        # =====================================================================
        logger.info("Phase 4: Running CodeQL with hint-based models...")

        # Export hints as CodeQL model extensions
        self._export_codeql_models(output_dir)

        # Run CodeQL
        warnings = self.analyzer.analyze(
            project_path,
            self.hints,
            self.issue_types,
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

    def _export_codeql_models(self, output_dir: Path) -> None:
        """Export hints as CodeQL model extensions."""
        # YAML model file
        model_yaml = output_dir / "codeql_model.yml"
        model_yaml.write_text(self.hints.to_codeql_model())

        # Also create model pack for CodeQL
        if isinstance(self.analyzer, CodeQLAnalyzer):
            query_pack_dir = output_dir / "query-pack"
            self.analyzer._setup_model_pack(query_pack_dir, self.hints)
            logger.info(f"  Created CodeQL query pack at {query_pack_dir}")

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

        # Human-readable report
        report = self._generate_report(confirmed)
        report_file = output_dir / "report.md"
        report_file.write_text(report)

        logger.info(f"Results saved to {output_dir}")

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
            MemoryIssueType.NULL_DEREFERENCE: "Add null check before pointer use",
            MemoryIssueType.BUFFER_OVERFLOW: "Validate buffer bounds before access",
        }
        return fixes.get(warning.issue_type, "Review memory safety issue")
