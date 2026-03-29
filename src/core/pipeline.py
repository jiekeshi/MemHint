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
    Warning, Evidence, AnalysisResult, MemoryIssueType,
    CustomQuery, CustomQuerySet
)
from src.tree_sitter_parser import CodeParser
from src.llm_client import HintGenerator, LLMClient, CustomQueryGenerator
from src.symbolic.z3_solver import HintValidator, WarningValidator
from src.analyzer.adapters import CodeQLAnalyzer, InferAnalyzer

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
        
        # Select analyzer based on analyzer_type
        if analyzer_type == "infer":
            self.analyzer = InferAnalyzer(reuse_db=reuse_db)
        else:
            # Default to CodeQL
            self.analyzer = CodeQLAnalyzer(codeql_dir=codeql_dir, cpp_queries_dir=cpp_queries_dir, reuse_db=reuse_db)

        # Configuration
        self.analyzer_type = analyzer_type
        self.use_merge = use_merge
        self.skip_hint_injection = skip_hint_injection
        self.use_enhanced_queries = use_enhanced_queries
        self.issue_types = issue_types or [
            MemoryIssueType.MEMORY_LEAK,
            MemoryIssueType.DOUBLE_FREE,
            MemoryIssueType.USE_AFTER_FREE,
            MemoryIssueType.MEMORY_LEAK_FILTERED,
            MemoryIssueType.USE_AFTER_FREE_FILTERED,
            MemoryIssueType.DOUBLE_FREE_FILTERED,
        ]

        # State
        self.functions: dict[str, FunctionInfo] = {}
        self.macro_names: set[str] = set()  # Track which function names are from macros
        self.hints = HintSet()
        self.custom_queries = CustomQuerySet()  # Custom queries for special functions
        self.conflicts: list[str] = []
        self.validated_hints_count: int = 0  # Track number of hints that passed Z3 validation
        # Track function/macro replacements due to duplicate names (for later analysis/LLM reuse)
        self.function_replacements: list[dict] = []

    def _serialize_function_for_llm(self, name: str, is_macro: bool) -> dict | None:
        """Serialize a function into a JSON-friendly dict for LLM reuse."""
        func = self.functions.get(name)
        if not func:
            return None
        return {
            "name": func.name,
            "file_path": getattr(func, "file_path", ""),
            "start_line": getattr(func, "start_line", 0),
            "end_line": getattr(func, "end_line", 0),
            "arg_names": getattr(func, "arg_names", []),
            "arg_types": getattr(func, "arg_types", []),
            "return_type": getattr(func, "return_type", ""),
            "is_macro": is_macro,
            "code": getattr(func, "code", ""),
        }

    def _compute_llm_filter_classes(self, pointer_typedef_aliases: set[str]) -> dict[str, list[str]]:
        """Compute (kept/filtered) classes without calling the LLM."""
        self.hint_generator.set_pointer_typedef_aliases(pointer_typedef_aliases)

        kept: list[str] = []
        filtered_entry: list[str] = []
        filtered_non_pointer: list[str] = []

        for fn, f in self.functions.items():
            # Match HintGenerator.generate_hints() filtering behavior
            if fn in ("main", "_main", "wmain") or "test" in fn.lower():
                filtered_entry.append(fn)
                continue

            if fn in self.macro_names or self.hint_generator._has_pointer_io(f):
                kept.append(fn)
            else:
                filtered_non_pointer.append(fn)

        return {
            "kept": sorted(kept),
            "filtered_entry": sorted(filtered_entry),
            "filtered_non_pointer": sorted(filtered_non_pointer),
        }

    def _export_llm_filter_metadata(self, output_dir: Path, classes: dict[str, list[str]] | None = None) -> None:
        """Export LLM filter info (kept/filtered/replaced functions) to JSON for later reuse."""
        if classes is None:
            classes = getattr(self.hint_generator, "last_filter_classes", None)
        if not classes:
            return

        kept_names = classes.get("kept", []) or []
        filtered_entry = classes.get("filtered_entry", []) or []
        filtered_nonptr = classes.get("filtered_non_pointer", []) or []

        kept = []
        for name in kept_names:
            is_macro = name in self.macro_names
            meta = self._serialize_function_for_llm(name, is_macro)
            if meta:
                kept.append(meta)

        filtered_entry_detail = []
        for name in filtered_entry:
            is_macro = name in self.macro_names
            meta = self._serialize_function_for_llm(name, is_macro)
            if meta:
                filtered_entry_detail.append(meta)

        filtered_nonptr_detail = []
        for name in filtered_nonptr:
            is_macro = name in self.macro_names
            meta = self._serialize_function_for_llm(name, is_macro)
            if meta:
                filtered_nonptr_detail.append(meta)

        data = {
            "total_functions": len(self.functions),
            "macro_names": sorted(self.macro_names),
            "kept_for_llm": kept,
            "filtered_entry_or_test": filtered_entry_detail,
            "filtered_non_pointer": filtered_nonptr_detail,
            "replacements": self.function_replacements,
        }

        out_path = output_dir / "llm_filter_functions.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def analyze(
        self,
        project_path: Path,
        output_dir: Path = None,
        single_sources: list[Path] | None = None,
        source_root: Path | None = None,
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
        source_root = Path(source_root).resolve() if source_root else project_path
        output_dir = output_dir or Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        single_sources = [Path(p).resolve() for p in single_sources] if single_sources else None

        logger.info(f"Analyzing {project_path}")
        logger.info(f"Bug types: {[t.name for t in self.issue_types]}")

        # =====================================================================
        # Phase 1: Parse source code
        # =====================================================================
        logger.info("Phase 1: Parsing source code...")

        # We use the same parsing logic for both full and single-source modes.
        # The only difference is the set of files we feed into the parser.
        target_files: list[Path] = []
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
            target_files = single_sources
        else:
            logger.info(f"  Full-project mode: parsing all source files under source root {source_root}")
            # Reuse the same extension set as CodeParser.parse_project_with_macros
            exts = {
                ".c", ".cpp", ".cc", ".cxx",
                ".h", ".hpp", ".hxx", ".hh",
                ".inc", ".inl", ".def",
            }
            for fp in source_root.rglob("*"):
                if fp.suffix.lower() in exts:
                    target_files.append(fp)

        # Unified parsing for both modes: per-file parse_file_with_macros
        self.functions = {}
        self.macro_names = set()
        for src in target_files:
            logger.debug(f"    Parsing {src}")
            file_functions, file_macros = self.parser.parse_file_with_macros(src, preprocess=False)

            # Merge functions: if duplicates exist, keep the one with longer code (more context)
            for func_name, info in file_functions.items():
                if not getattr(info, "file_path", None):
                    info.file_path = str(src)
                if func_name in self.functions:
                    existing = self.functions[func_name]
                    if len(info.code) > len(existing.code):
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "    Function %s found in multiple files, replacing with longer definition from %s",
                                func_name,
                                src,
                            )
                        # Record replacement detail for later analysis / LLM reuse
                        self.function_replacements.append({
                            "name": func_name,
                            "kept": {
                                "name": info.name,
                                "file_path": info.file_path,
                                "start_line": info.start_line,
                                "end_line": info.end_line,
                                "arg_names": info.arg_names,
                                "arg_types": info.arg_types,
                                "return_type": info.return_type,
                                "is_macro": False,
                            },
                            "discarded": {
                                "name": existing.name,
                                "file_path": existing.file_path,
                                "start_line": existing.start_line,
                                "end_line": existing.end_line,
                                "arg_names": existing.arg_names,
                                "arg_types": existing.arg_types,
                                "return_type": existing.return_type,
                                "is_macro": False,
                            },
                            "reason": "longer_code",
                        })
                        self.functions[func_name] = info
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "    Function %s found in multiple files, keeping existing definition from %s",
                                func_name,
                                existing.file_path,
                            )
                        # Record that the new definition was discarded
                        self.function_replacements.append({
                            "name": func_name,
                            "kept": {
                                "name": existing.name,
                                "file_path": existing.file_path,
                                "start_line": existing.start_line,
                                "end_line": existing.end_line,
                                "arg_names": existing.arg_names,
                                "arg_types": existing.arg_types,
                                "return_type": existing.return_type,
                                "is_macro": False,
                            },
                            "discarded": {
                                "name": info.name,
                                "file_path": info.file_path,
                                "start_line": info.start_line,
                                "end_line": info.end_line,
                                "arg_names": info.arg_names,
                                "arg_types": info.arg_types,
                                "return_type": info.return_type,
                                "is_macro": False,
                            },
                            "reason": "duplicate_function_kept_existing",
                        })
                    continue
                self.functions[func_name] = info

            # Convert function-like macros to FunctionInfo and merge
            for macro_name, macro_info in file_macros.items():
                if not macro_info.is_function_like:
                    continue

                macro_code = macro_info.code
                if macro_info.expansion:
                    macro_code += f"\n// Expands to: {macro_info.expansion}"

                func_info = FunctionInfo(
                    name=macro_name,
                    code=macro_code,
                    file_path=macro_info.file_path or str(src),
                    start_line=macro_info.start_line,
                    end_line=macro_info.end_line,
                    arg_names=macro_info.arg_names,
                    arg_types=[""] * len(macro_info.arg_names),
                    return_type="",
                    callees=set(),
                    callers=set(),
                    return_expressions={macro_info.expansion} if macro_info.expansion else set(),
                )
                if macro_name in self.functions:
                    existing = self.functions[macro_name]
                    if len(func_info.code) > len(existing.code):
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "    Macro/Function %s found in multiple files, replacing with longer definition from %s",
                                macro_name,
                                src,
                            )
                        self.function_replacements.append({
                            "name": macro_name,
                            "kept": {
                                "name": func_info.name,
                                "file_path": func_info.file_path,
                                "start_line": func_info.start_line,
                                "end_line": func_info.end_line,
                                "arg_names": func_info.arg_names,
                                "arg_types": func_info.arg_types,
                                "return_type": func_info.return_type,
                                "is_macro": True,
                            },
                            "discarded": {
                                "name": existing.name,
                                "file_path": existing.file_path,
                                "start_line": existing.start_line,
                                "end_line": existing.end_line,
                                "arg_names": existing.arg_names,
                                "arg_types": existing.arg_types,
                                "return_type": existing.return_type,
                                "is_macro": True,
                            },
                            "reason": "longer_code",
                        })
                        self.functions[macro_name] = func_info
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "    Macro/Function %s found in multiple files, keeping existing definition from %s",
                                macro_name,
                                existing.file_path,
                            )
                        self.function_replacements.append({
                            "name": macro_name,
                            "kept": {
                                "name": existing.name,
                                "file_path": existing.file_path,
                                "start_line": existing.start_line,
                                "end_line": existing.end_line,
                                "arg_names": existing.arg_names,
                                "arg_types": existing.arg_types,
                                "return_type": existing.return_type,
                                "is_macro": True,
                            },
                            "discarded": {
                                "name": func_info.name,
                                "file_path": func_info.file_path,
                                "start_line": func_info.start_line,
                                "end_line": func_info.end_line,
                                "arg_names": func_info.arg_names,
                                "arg_types": func_info.arg_types,
                                "return_type": func_info.return_type,
                                "is_macro": True,
                            },
                            "reason": "duplicate_macro_kept_existing",
                        })
                    # Regardless, it is still a macro name
                    self.macro_names.add(macro_name)
                    continue

                self.functions[macro_name] = func_info
                self.macro_names.add(macro_name)

        # Best-effort call graph resolution (both modes)
        try:
            self.parser._resolve_calls(self.functions)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"  Warning: could not resolve calls: {e}")
        
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
                # logger.debug(f"    Scan target function: {func_name} (file: {file_hint})")
                pass

        if not self.functions:
            logger.warning("No functions found")
            return AnalysisResult(
                confirmed_bugs=[],
                hints=HintSet(),
                iterations=0,
                spurious_filtered=0,
            )

        # Export filter+replacement metadata right after parsing (so you can reuse it without running the full pipeline)
        try:
            pointer_typedef_aliases = getattr(self.parser, "pointer_typedef_aliases", set())  # type: ignore[attr-defined]
            classes = self._compute_llm_filter_classes(pointer_typedef_aliases)
            self._export_llm_filter_metadata(output_dir, classes=classes)
            logger.info(
                "Wrote LLM filter metadata JSON: kept=%d, filtered_entry/test=%d, filtered_non_pointer=%d",
                len(classes.get("kept", [])),
                len(classes.get("filtered_entry", [])),
                len(classes.get("filtered_non_pointer", [])),
            )
        except Exception as e:
            logger.warning(f"  Warning: failed to export LLM filter metadata after parsing: {e}")

        if self.skip_hint_injection:
            logger.info("Hint injection disabled: skipping hint generation/validation (phases 2-3).")
            self.hints = HintSet()
            self.custom_queries = CustomQuerySet()
            self.validated_hints_count = 0
            # Export empty cost file
            self._export_costs(output_dir)
        else:
            # Check for cached hints with fallback logic:
            # 1. First check current analyzer's output directory
            # 2. If Infer and not found, copy from CodeQL's equivalent directory
            current_hints_file = output_dir / "hints.json"
            hints_file = self._find_hints_file(output_dir)
            
            if hints_file and hints_file.exists():
                logger.info(f"  Loading cached hints from {hints_file}")
                self.hints = self._load_hints(hints_file)
                
                # If hints were loaded from CodeQL directory, copy them to current directory
                if hints_file != current_hints_file:
                    logger.info(f"  Copying hints from CodeQL directory to current directory: {current_hints_file}")
                    import shutil
                    shutil.copy2(hints_file, current_hints_file)
                
                # Count validated hints from cached hints (approximate - all cached hints are considered validated)
                self.validated_hints_count = sum(len(hints) for hints in self.hints.hints.values())
                
                # Try to load cached custom queries (analyzer-specific)
                # Custom queries are analyzer-specific, so use analyzer-specific filename
                analyzer_suffix = f"_{self.analyzer_type}" if self.analyzer_type != "codeql" else ""
                custom_queries_file = output_dir / f"custom_queries{analyzer_suffix}.json"
                if custom_queries_file.exists():
                    logger.info(f"  Loading cached custom queries from {custom_queries_file}")
                    self.custom_queries = self._load_custom_queries(custom_queries_file)
                    logger.info(f"  {self.custom_queries.summary()}")
                    if len(self.custom_queries) > 0:
                        logger.info(f"  Special functions: {', '.join(sorted(self.custom_queries.queries.keys()))}")
                else:
                    # Custom queries not cached, generate them from validated hints (analyzer-specific)
                    logger.info(f"  Custom queries not cached, generating from validated hints for {self.analyzer_type}...")
                    self.custom_queries = self._generate_custom_queries_iteratively(
                        self.functions,
                        self.hints,
                    )
                    # Save generated custom queries with analyzer-specific filename
                    self._save_custom_queries(custom_queries_file)
                
                # Export cost file (will show zeros since no LLM calls were made)
                self._export_costs(output_dir)
            else:
                # =================================================================
                # Phase 2-3: hint generation and validation
                # =================================================================

                all_conflicts = []

                # -----------------------------------------------------------------
                # Phase 2a: Filter LLM candidate functions (pointer / alloc/free related)
                # -----------------------------------------------------------------
                total_funcs = len(self.functions)
                num_macros = len(self.macro_names)
                pointer_typedef_aliases = getattr(self.parser, "pointer_typedef_aliases", set())  # type: ignore[attr-defined]
                self.hint_generator.set_pointer_typedef_aliases(pointer_typedef_aliases)
                if logger.isEnabledFor(logging.DEBUG) and pointer_typedef_aliases:
                    logger.debug(
                        "Phase 2a: Found %d pointer typedef aliases: %s",
                        len(pointer_typedef_aliases),
                        sorted(pointer_typedef_aliases),
                    )

                # Classify all parsed functions into "kept" and "filtered out" for LLM
                kept_functions: list[str] = []
                filtered_functions: list[str] = []
                for fn, f in self.functions.items():
                    if fn in self.macro_names or self.hint_generator._has_pointer_io(f):
                        kept_functions.append(fn)
                    else:
                        filtered_functions.append(fn)

                candidate_funcs = len(kept_functions)
                logger.info(
                    "Phase 2a: Filtering LLM candidates - %d/%d functions are pointer/alloc/free related (including %d macros)",
                    candidate_funcs,
                    total_funcs,
                    num_macros,
                )

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Phase 2a: Kept %d function(s) for LLM (including macros): %s",
                        len(kept_functions),
                        sorted(kept_functions),
                    )
                    logger.debug(
                        "Phase 2a: Filtered out %d function(s) (no pointer/alloc/free IO): %s",
                        len(filtered_functions),
                        sorted(filtered_functions),
                    )

                # Initial hint generation (actual per-function filtering is done inside HintGenerator)
                logger.info("Phase 2: Generating hints with LLM...")
                self.hints = self.hint_generator.generate_hints(
                    self.functions, 
                    macro_names=self.macro_names,
                    pointer_typedef_aliases=pointer_typedef_aliases,
                )
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
                        previous_conflicts=conflicts,  # Pass current iteration's conflicts for LLM feedback
                        macro_names=self.macro_names,
                        pointer_typedef_aliases=pointer_typedef_aliases,
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

                # Save validated hints to current output directory (even if loaded from CodeQL directory)
                current_hints_file = output_dir / "hints.json"
                self._save_hints(current_hints_file)
                
                # =================================================================
                # Phase 2b: Generate custom queries for special functions (after Z3 validation)
                # Note: Custom queries are analyzer-specific (CodeQL queries vs Infer models)
                # =================================================================
                logger.info(f"Phase 2b: Generating custom queries for {self.analyzer_type} (one by one)...")
                self.custom_queries = self._generate_custom_queries_iteratively(
                    self.functions,
                    self.hints,
                )
                logger.info(f"  {self.custom_queries.summary()}")
                if len(self.custom_queries) > 0:
                    logger.info(f"  Special functions: {', '.join(sorted(self.custom_queries.queries.keys()))}")
                
                # Save custom queries with analyzer-specific filename
                analyzer_suffix = f"_{self.analyzer_type}" if self.analyzer_type != "codeql" else ""
                custom_queries_file = output_dir / f"custom_queries{analyzer_suffix}.json"
                self._save_custom_queries(custom_queries_file)
                
                # Export cost information right after hint generation completes
                self._export_costs(output_dir)

        # =====================================================================
        # Phase 4: Static analysis with hints and custom queries
        # =====================================================================
        analyzer_name = "Infer" if self.analyzer_type == "infer" else "CodeQL"
        logger.info(f"Phase 4: Running {analyzer_name} with hint-based models and custom queries...")

        # Hints are SHARED between analyzers (same LLM-generated memory semantics)
        # Custom queries are analyzer-specific (CodeQL queries vs Infer models)
        # All functions (including special ones) get allocator/deallocator models from hints
        # Special functions additionally get analyzer-specific custom queries
        hints_for_analyzer = self.hints

        # Run analyzer with shared hints and analyzer-specific custom queries
        warnings = self.analyzer.analyze(
            project_path,
            hints_for_analyzer,  # Shared hints (same for CodeQL and Infer)
            self.issue_types,
            use_enhanced_queries=self.use_enhanced_queries,
            custom_queries=self.custom_queries if hasattr(self, 'custom_queries') else None,  # Analyzer-specific
        )
        logger.info(f"  {analyzer_name} found {len(warnings)} warnings")

        analyzer_lower = analyzer_name.lower()
        warning_dump_path = output_dir / f"{analyzer_lower}_warnings.json"

        with open(warning_dump_path, "w") as f:
            json.dump([w.to_dict() for w in warnings], f, indent=2)

        logger.info(f"  Saved {analyzer_name} warnings to {warning_dump_path}")
        
        # Get and save irrelevant warnings from Infer immediately (non-memory-safety issues)
        # This happens right after parsing, same as the main warnings file
        if self.analyzer_type == "infer" and hasattr(self.analyzer, 'get_irrelevant_warnings'):
            irrelevant_warnings = self.analyzer.get_irrelevant_warnings()
            if irrelevant_warnings:
                irrelevant_path = output_dir / f"{analyzer_lower}_irrelevant_warnings.json"
                with open(irrelevant_path, "w") as f:
                    json.dump([w.to_dict() for w in irrelevant_warnings], f, indent=2)
                logger.info(f"  Saved {len(irrelevant_warnings)} non-memory-safety warnings to {irrelevant_path} (saved immediately after parsing)")


        # =====================================================================
        # Phase 5: Z3 filters false positives
        # =====================================================================
        logger.info("Phase 5: Filtering warnings with Z3 path analysis...")

        # Initialize warning validator with known allocators       
        alloc_funcs = set(fn for fn, _ in self.hints.get_allocators())
        alloc_funcs.update({"malloc", "calloc", "realloc", "strdup"})
        free_funcs = set(fn for fn, _ in self.hints.get_deallocators())
        free_funcs.update({"free"})

        self.warning_validator = WarningValidator(alloc_funcs, free_funcs)
        confirmed_warnings, filtered_warnings ,unconfirmed_warnings= self.warning_validator.validate_warnings(
            warnings, self.functions, self.hints
        )

        logger.info(f"  Total warnings: {len(warnings)}, Confirmed: {len(confirmed_warnings)}, Filtered: {len(filtered_warnings)}, Unconfirmed: {len(unconfirmed_warnings)}")
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
        self._export_results(output_dir, confirmed_bugs, filtered_warnings, unconfirmed_warnings)

        return AnalysisResult(
            confirmed_bugs=confirmed_bugs,
            hints=self.hints,
            iterations=1,
            spurious_filtered=len(filtered_warnings),
        )

    def _find_hints_file(self, output_dir: Path) -> Path | None:
        """Find hints.json file with fallback logic.
        
        For Infer: checks its own directory first, then falls back to CodeQL's equivalent directory.
        For CodeQL: checks its own directory only.
        
        Args:
            output_dir: Current analyzer's output directory
            
        Returns:
            Path to hints.json if found, None otherwise
        """
        # First, check current analyzer's output directory
        hints_file = output_dir / "hints.json"
        if hints_file.exists():
            return hints_file
        
        # If using Infer and hints not found, try to find CodeQL's equivalent directory
        if self.analyzer_type == "infer":
            # Infer output dirs typically have "_infer" suffix
            # Try to find the CodeQL equivalent by removing "_infer" suffix
            output_str = str(output_dir)
            if "_infer" in output_str:
                # Replace "_infer" with empty string to get CodeQL equivalent
                codeql_output_str = output_str.replace("_infer", "")
                codeql_output_dir = Path(codeql_output_str)
                codeql_hints_file = codeql_output_dir / "hints.json"
                if codeql_hints_file.exists():
                    logger.info(f"  Hints not found in Infer directory ({output_dir}), checking CodeQL equivalent: {codeql_hints_file}")
                    return codeql_hints_file
                else:
                    logger.debug(f"  CodeQL equivalent directory not found: {codeql_output_dir}")
            
            # Also try looking in parent directory for sibling CodeQL directories
            parent_dir = output_dir.parent
            if parent_dir.exists():
                # Look for directories that match the pattern but without "_infer"
                base_name = output_dir.name
                if "_infer" in base_name:
                    codeql_base_name = base_name.replace("_infer", "")
                    codeql_output_dir = parent_dir / codeql_base_name
                    codeql_hints_file = codeql_output_dir / "hints.json"
                    if codeql_hints_file.exists():
                        logger.info(f"  Hints not found in Infer directory, checking CodeQL sibling: {codeql_hints_file}")
                        return codeql_hints_file
        
        return None
    
    def _load_hints(self, file_path: Path) -> HintSet:
        """Load hints from JSON file."""
        data = json.loads(file_path.read_text())
        return HintSet.from_json(data)

    def _save_hints(self, file_path: Path) -> None:
        """Save hints to JSON file."""
        file_path.write_text(json.dumps(self.hints.to_json(), indent=2))
    
    def _load_custom_queries(self, file_path: Path) -> CustomQuerySet:
        """Load custom queries from JSON file."""
        data = json.loads(file_path.read_text())
        return CustomQuerySet.from_json(data)
    
    def _save_custom_queries(self, file_path: Path) -> None:
        """Save custom queries to JSON file."""
        file_path.write_text(json.dumps(self.custom_queries.to_json(), indent=2))
        if len(self.custom_queries) > 0:
            logger.info(f"  Custom queries saved to {file_path}")

    def _export_results(
        self,
        output_dir: Path,
        confirmed: list[Evidence],
        filtered: list[Warning],
        unconfirmed: list[Warning],
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
                "warning_type": bug.warning.warning_type,
                "type": bug.warning.issue_type.name,
                "message": bug.warning.message,
                "allocation_site": bug.warning.allocation_site,
                "trace": bug.warning.trace,
                "suggested_fix": bug.suggested_fix,
            })

        bugs_file = output_dir / "memory_safety_bugs.json"
        bugs_file.write_text(json.dumps(bugs_data, indent=2))

        # Filtered warnings (for debugging)
        filtered_data = {}
        for w in filtered:
            t = w.issue_type.name
            if t not in filtered_data:
                filtered_data[t] = []
            filtered_data[w.issue_type.name].append({
                "file": w.file_path,
                "line": w.line_number,
                "function": w.function_name,
                "warning_type": w.warning_type,
                "message": w.message,
                "allocation_site": w.allocation_site,
                "trace": w.trace,
            })
        filtered_file = output_dir / "filtered_warnings.json"
        filtered_file.write_text(json.dumps(filtered_data, indent=2))

        # Unconfirmed warnings (for debugging)
        unconfirmed_data = {}
        for w in unconfirmed:
            t = w.issue_type.name
            if t not in unconfirmed_data:
                unconfirmed_data[t] = []
            unconfirmed_data[w.issue_type.name].append({
                "file": w.file_path,
                "line": w.line_number,
                "function": w.function_name,
                "warning_type": w.warning_type,
                "message": w.message,
                "allocation_site": w.allocation_site,
                "trace": w.trace,
            })
        unconfirmed_file = output_dir / "unconfirmed_warnings.json"
        unconfirmed_file.write_text(json.dumps(unconfirmed_data, indent=2))
        
        # Custom queries (if any) - save with analyzer-specific filename
        if len(self.custom_queries) > 0:
            analyzer_suffix = f"_{self.analyzer_type}" if self.analyzer_type != "codeql" else ""
            custom_queries_file = output_dir / f"custom_queries{analyzer_suffix}.json"
            custom_queries_file.write_text(json.dumps(self.custom_queries.to_json(), indent=2))
            logger.info(f"  Custom queries saved to {custom_queries_file}")

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
            MemoryIssueType.MEMORY_LEAK_FILTERED: "Free allocated memory before function exit",
            MemoryIssueType.USE_AFTER_FREE_FILTERED: "Use memory before freeing, not after",
            MemoryIssueType.DOUBLE_FREE_FILTERED: "Remove duplicate free() or add null guard",
        }
        return fixes.get(warning.issue_type, "Review memory safety issue")

    def _generate_custom_queries_iteratively(
        self,
        functions: dict[str, FunctionInfo],
        hints: HintSet,
    ) -> CustomQuerySet:
        """Generate custom queries for validated hints iteratively (one by one).
        
        This is analyzer-specific:
        - For CodeQL: generates CodeQL filter queries
        - For Infer: generates Infer-specific models/annotations (or skips if not needed)
        
        After Z3 validation confirms the hints, evaluate each function that has validated
        hints to determine if it needs a custom query. Process them one by one with logging.
        
        All evaluated functions are stored in CustomQuerySet with their decisions:
        - Special functions: includes generated filter predicates and reasoning
        - Non-special functions: includes reasoning for why no query was needed
        
        Args:
            functions: Dict of all functions
            hints: The validated hints (after Z3 validation) - SHARED between analyzers
            
        Returns:
            CustomQuerySet with evaluation results for all validated functions
        """
        custom_queries = CustomQuerySet()
        
        # For Infer, custom query generation is not needed (models are injected directly from hints)
        if self.analyzer_type == "infer":
            logger.info("  Infer uses direct model injection from hints; skipping custom query generation")
            return custom_queries
        
        # For CodeQL, generate custom filter queries
        if not self.hint_generator.llm:
            logger.warning("No LLM client available for custom query generation")
            return custom_queries
        
        # Create CodeQL-specific custom query generator
        cq_generator = CustomQueryGenerator(
            llm=self.hint_generator.llm,
            codeql_validator=None,  # Validation handled internally in llm_client
        )
        
        # Get functions with validated hints
        functions_with_hints = list(hints.hints.keys())
        total_functions_with_hints = len(functions_with_hints)
        
        logger.info(f"Processing {total_functions_with_hints} function(s) with validated hints...")
        
        # Process each function with validated hints one by one
        for idx, func_name in enumerate(functions_with_hints, 1):
            if func_name not in functions:
                logger.warning(f"  [{idx}/{total_functions_with_hints}] Function '{func_name}' not found in parsed functions, skipping")
                continue
            
            func = functions[func_name]
            func_hints = hints.hints[func_name]
            
            logger.info(f"  [{idx}/{total_functions_with_hints}] Evaluating '{func_name}' for custom CodeQL query ({len(func_hints)} hint(s))")
            
            # Evaluate and generate custom query for this function
            query = cq_generator.generate_custom_query_for_function(func, func_hints)
            if query:
                custom_queries.add(query)
                if query.is_special:
                    logger.debug(f"      ✓ Custom query generated for '{func_name}'")
                else:
                    logger.debug(f"      - No custom query needed for '{func_name}': {query.reason}")
        
        return custom_queries