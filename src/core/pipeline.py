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
import os
import shutil
import tempfile
import time
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
        api_key: str = None,
        model: str = "gemini-3-flash-preview",
        analyzer_type: str = "codeql",
        issue_types: list[MemoryIssueType] = None,
        use_merge: bool = False,  # Kept for compatibility
        codeql_dir: Path = None,
        cpp_queries_dir: Path = None,
        reuse_db: bool = True,
        skip_hint_injection: bool = False,
        use_enhanced_queries: bool = True,
        skip_z3: bool = False,
        use_llm_relabel: bool = False,
        use_custom_queries: bool = False,
        use_llm_verify_bugs: bool = False,
        llm_verify_model: str = "gemini-3.1-pro-preview",
        pipeline_mode: str = "full",
        hint_batch_size: int | None = None,
    ):
        """Initialize pipeline.
        
        Args:
            api_key: Optional Gemini API key for direct HTTP calls. If not provided,
                the LLM client falls back to Vertex AI using GOOGLE_APPLICATION_CREDENTIALS.
            model: LLM model name
            analyzer_type: "codeql" or "infer"
            issue_types: Which bug types to detect (default: all)
            use_merge: Whether to merge code (kept for compatibility)
            codeql_dir: Optional custom CodeQL directory path (default: ~/.codeql)
            cpp_queries_dir: Optional direct path to cpp-queries directory
            reuse_db: Whether to reuse existing CodeQL database (default: True)
            pipeline_mode: "full" to run the entire pipeline (parse → hints → analyzer →
                warning validation), or "hints-only" to stop after hint generation/validation.
        """
        # Components
        self.parser = CodeParser()
        self.llm_client = LLMClient(api_key=api_key, model=model)
        self.hint_generator = HintGenerator(self.llm_client)
        self.hint_validator = HintValidator()
        self.warning_validator = None  # Initialized with known allocators
        
        # Initialize analyzer based on analyzer_type
        if analyzer_type == "infer":
            self.analyzer = InferAnalyzer(reuse_db=reuse_db)
        else:
            self.analyzer = CodeQLAnalyzer(codeql_dir=codeql_dir, cpp_queries_dir=cpp_queries_dir, reuse_db=reuse_db)

        # Configuration
        self.analyzer_type = analyzer_type
        self.use_merge = use_merge
        self.skip_hint_injection = skip_hint_injection
        self.use_enhanced_queries = use_enhanced_queries
        self.skip_z3 = skip_z3
        self.use_llm_relabel = use_llm_relabel
        self.use_custom_queries = use_custom_queries
        self.use_llm_verify_bugs = use_llm_verify_bugs
        self.llm_verify_model = llm_verify_model or "gemini-3.1-pro-preview"
        self.pipeline_mode = pipeline_mode or "full"
        self.issue_types = issue_types or [
            MemoryIssueType.MEMORY_LEAK,
            MemoryIssueType.MEMORY_LEAK_FILTERED,
        ]

        # How many functions to send to the LLM in a single call when
        # generating hints. Can be configured via:
        #   1) Constructor argument hint_batch_size
        #   2) Environment variable HINT_BATCH_SIZE
        # Defaults to 1 (one function per LLM call).
        env_batch = os.getenv("HINT_BATCH_SIZE", "").strip()
        resolved_batch: int | None = None
        if hint_batch_size is not None:
            try:
                resolved_batch = int(hint_batch_size)
            except (TypeError, ValueError):
                resolved_batch = None
        elif env_batch:
            try:
                resolved_batch = int(env_batch)
            except ValueError:
                resolved_batch = None

        if resolved_batch is None or resolved_batch <= 0:
            resolved_batch = 1
        self.hint_batch_size: int = resolved_batch

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
        # Shared cache directory for hints / relabeled hints / custom queries:
        # by default we use the parent of the run-specific output_dir so that
        # different configurations (relabel/custom-query) can share them.
        cache_dir = output_dir.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        single_sources = [Path(p).resolve() for p in single_sources] if single_sources else None

        logger.info(f"Analyzing {project_path}")
        logger.info(f"Bug types: {[t.name for t in self.issue_types]}")

        # Phase timing tracker
        phase_timings = {}
        _current_phase = [None, 0.0]  # [name, start_time]
        def _start_phase(name: str):
            # End previous phase if any
            if _current_phase[0]:
                elapsed = time.time() - _current_phase[1]
                phase_timings[_current_phase[0]] = elapsed
                logger.info(f"[TIMING] {_current_phase[0]} finished in {elapsed:.1f}s ({elapsed/60:.1f}m)")
            _current_phase[0] = name
            _current_phase[1] = time.time()
            logger.info(f"[TIMING] {name} started")
        def _end_all_phases():
            if _current_phase[0]:
                elapsed = time.time() - _current_phase[1]
                phase_timings[_current_phase[0]] = elapsed
                logger.info(f"[TIMING] {_current_phase[0]} finished in {elapsed:.1f}s ({elapsed/60:.1f}m)")
                _current_phase[0] = None
            # Save timing summary
            logger.info("[TIMING] === Phase Timing Summary ===")
            for name, elapsed in phase_timings.items():
                logger.info(f"[TIMING]   {name}: {elapsed:.1f}s ({elapsed/60:.1f}m)")
            total = sum(phase_timings.values())
            logger.info(f"[TIMING]   TOTAL: {total:.1f}s ({total/60:.1f}m)")

        # =====================================================================
        # Phase 1: Parse source code
        # =====================================================================
        _start_phase("Phase1_Parsing")
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
            self._export_llm_filter_metadata(cache_dir, classes=classes)
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
            self._export_costs(cache_dir)
        else:
            # Check for cached hints with fallback logic:
            # 1. First check current analyzer's cache directory
            # 2. If Infer and not found, copy from CodeQL's equivalent directory
            current_hints_file = cache_dir / "hints.json"
            hints_file = self._find_hints_file(cache_dir)
            
            if hints_file and hints_file.exists():
                logger.info(f"  Loading cached hints from {hints_file}")
                self.hints = self._load_hints(hints_file)
                
                # If hints were loaded from other analyzer's directory, copy them to current directory
                if hints_file != current_hints_file:
                    other_analyzer = "CodeQL" if self.analyzer_type == "infer" else "Infer"
                    logger.info(f"  Hints not found in {self.analyzer_type} directory, copying from {other_analyzer} directory: {current_hints_file}")
                    import shutil
                    shutil.copy2(hints_file, current_hints_file)
                
                # Count validated hints from cached hints (approximate - all cached hints are considered validated)
                self.validated_hints_count = sum(len(hints) for hints in self.hints.hints.values())
                
                # Try to load cached custom queries (analyzer-specific) only if enabled
                if self.use_custom_queries:
                    # Custom queries can be copied from other analyzer
                    current_custom_queries_file = self._find_custom_queries_file(cache_dir)
                    
                    if current_custom_queries_file and current_custom_queries_file.exists():
                        logger.info(f"  Loading cached custom queries from {current_custom_queries_file}")
                        self.custom_queries = self._load_custom_queries(current_custom_queries_file)
                        logger.info(f"  {self.custom_queries.summary()}")
                        if len(self.custom_queries) > 0:
                            logger.info(f"  Special functions: {', '.join(sorted(self.custom_queries.queries.keys()))}")
                        
                        # If loaded from other analyzer, copy to current directory
                        analyzer_suffix = f"_{self.analyzer_type}" if self.analyzer_type != "codeql" else ""
                        target_custom_queries_file = cache_dir / f"custom_queries{analyzer_suffix}.json"
                        if current_custom_queries_file != target_custom_queries_file:
                            other_analyzer = "CodeQL" if self.analyzer_type == "infer" else "Infer"
                            logger.info(f"  Custom queries not found in {self.analyzer_type} directory, copying from {other_analyzer} directory: {target_custom_queries_file}")
                            import shutil
                            shutil.copy2(current_custom_queries_file, target_custom_queries_file)
                    else:
                        # Custom queries not cached, generate them from validated hints (analyzer-specific)
                        logger.info(f"  Custom queries not cached, generating from validated hints for {self.analyzer_type}...")
                        self.custom_queries = self._generate_custom_queries_iteratively(
                            self.functions,
                            self.hints,
                        )
                        # Save generated custom queries with analyzer-specific filename
                        analyzer_suffix = f"_{self.analyzer_type}" if self.analyzer_type != "codeql" else ""
                        custom_queries_file = cache_dir / f"custom_queries{analyzer_suffix}.json"
                        self._save_custom_queries(custom_queries_file)
                else:
                    # Custom queries disabled
                    self.custom_queries = CustomQuerySet()
                
                # Do NOT rewrite LLM cost file when reusing cached hints.
                # This preserves the original cost information from the run
                # that actually generated the hints.
            else:
                # =================================================================
                # Phase 2-3: hint generation and validation
                # =================================================================

                all_conflicts = []

                # Fresh-start mode: if HINT_FRESH_START is set (e.g. "1"/"true"),
                # ignore any cached hints/partial progress and start from scratch.
                fresh_env = os.getenv("HINT_FRESH_START", "").strip().lower()
                fresh_start = fresh_env in ("1", "true", "yes")

                partial_hints = None
                processed_functions: set[str] = set()
                processed_false_functions: set[str] = set()

                if not fresh_start:
                    logger.info("  Hint generation resume mode: fresh_start disabled; will reuse partial/raw progress if available.")
                    # Try to resume from partial hints if a previous run was interrupted.
                    # This avoids re-calling the LLM for functions that already have hints.
                    partial_hints_file = cache_dir / "hints_partial.json"
                    if partial_hints_file.exists():
                        try:
                            logger.info(f"  Loading partial hints from {partial_hints_file} to resume hint generation")
                            partial_hints = self._load_hints(partial_hints_file)
                            logger.info(
                                "  Resume state: partial hints cover %d function(s)",
                                len(partial_hints.hints) if getattr(partial_hints, "hints", None) else 0,
                            )
                        except Exception as e:
                            logger.warning(f"  Warning: failed to load partial hints from {partial_hints_file}: {e}")
                            partial_hints = None

                    # Also try to detect functions that were already processed (including those
                    # that produced no hints) from hints_all_raw.json so we can skip them.
                    all_raw_file = cache_dir / "hints_all_raw.json"
                    if all_raw_file.exists():
                        try:
                            raw = json.loads(all_raw_file.read_text())
                            raw_hints = raw.get("hints") or {}

                            # IMPORTANT:
                            # hints_all_raw.json stores per-function entries with a boolean "processed".
                            # We should only skip functions whose last-known state is processed=true.
                            # Functions with processed=false (e.g., LLM timeout/HTTP error) should be
                            # retried on rerun, which matches the user's expectation for resuming.
                            processed_true: set[str] = set()
                            processed_false: set[str] = set()

                            if isinstance(raw_hints, dict):
                                for fn, entries in raw_hints.items():
                                    is_true = False
                                    is_false = False

                                    # entries is typically a list[dict]; be defensive.
                                    if isinstance(entries, list):
                                        for e in entries:
                                            if not isinstance(e, dict):
                                                continue
                                            p = e.get("processed")
                                            if p is True:
                                                is_true = True
                                            elif p is False:
                                                is_false = True
                                    elif isinstance(entries, dict):
                                        # Unexpected but handle single dict entry
                                        p = entries.get("processed")
                                        if p is True:
                                            is_true = True
                                        elif p is False:
                                            is_false = True

                                    if is_true:
                                        processed_true.add(str(fn))
                                    elif is_false:
                                        processed_false.add(str(fn))

                            processed_functions = processed_true
                            processed_false_functions = processed_false
                            logger.info(
                                "  Loaded processed function states from %s: processed_true=%d processed_false=%d",
                                all_raw_file,
                                len(processed_true),
                                len(processed_false),
                            )
                        except Exception as e:
                            logger.warning(f"  Warning: failed to load processed functions from {all_raw_file}: {e}")
                            processed_functions = set()
                            processed_false_functions = set()
                else:
                    logger.info("  Hint generation resume mode: fresh_start enabled; ignoring partial/raw progress and re-running from scratch.")

                # Small helper to persist partial (raw) hints after each function.
                # This writes *unvalidated* hints so that progress is not lost if
                # the LLM or process is unstable.
                def _save_partial_hints(func_name: str, hint_set: HintSet) -> None:
                    # Incrementally persist both:
                    # 1) hints_partial.json  -> only functions with allocator/deallocator hints
                    # 2) hints_all_raw.json -> all functions, including those with no hints
                    try:
                        partial_file = cache_dir / "hints_partial.json"
                        partial_file.write_text(json.dumps(hint_set.to_json(), indent=2))
                    except Exception as e:
                        logger.warning(f"  Warning: failed to save partial hints after {func_name}: {e}")

                    # Also persist the raw per-function view if available from the generator.
                    try:
                        all_results = getattr(self.hint_generator, "last_all_results", None)
                        processed = getattr(self.hint_generator, "processed_functions", set())
                        failed = getattr(self.hint_generator, "failed_functions", set())
                        if all_results is not None:
                            # Only export the LLM-candidate functions into hints_all_raw.json.
                            # Users expect this file to reflect the set of functions that would
                            # actually be analyzed by the LLM (after pointer/macro filtering),
                            # not every parsed function in the entire project.
                            classes = getattr(self.hint_generator, "last_filter_classes", None) or {}
                            kept = classes.get("kept", []) if isinstance(classes, dict) else []
                            candidate_only: set[str] = set(kept or [])

                            raw_data: dict[str, dict] = {"hints": {}}
                            for fn, func_hints in all_results.items():
                                if candidate_only and fn not in candidate_only:
                                    continue
                                is_processed = not processed or fn in processed
                                if func_hints and is_processed:
                                    # Normal case: function has one or more allocator/deallocator hints.
                                    raw_data["hints"][fn] = [
                                        {
                                            "name": fn,
                                            "role": "Allocator" if h.hint_type.name == "ALLOCATOR" else "Deallocator",
                                            "target": h.target,
                                            "processed": True,
                                        }
                                        for h in func_hints
                                    ]
                                elif is_processed:
                                    raw_data["hints"][fn] = [
                                        {
                                            "name": fn,
                                            "role": "none",
                                            "target": "none",
                                            "processed": True,
                                        }
                                    ]
                                else:
                                    raw_data["hints"][fn] = [
                                        {
                                            "name": fn,
                                            "role": "none",
                                            "target": "none",
                                            "processed": False,
                                        }
                                    ]

                            raw_path = cache_dir / "hints_all_raw.json"
                            raw_path.write_text(json.dumps(raw_data, indent=2))
                    except Exception as e:
                        logger.warning(f"  Warning: failed to save raw per-function hints after {func_name}: {e}")

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
                _start_phase("Phase2_LLM_Hints")
                logger.info("Phase 2: Generating hints with LLM...")

                # Log how many functions will be re-run in resume mode.
                # We only count functions that:
                #   - were previously marked processed=false in hints_all_raw.json, AND
                #   - are still in the LLM candidate set (pointer/alloc/free related),
                # to avoid inflating the number with filtered-out functions.
                try:
                    if not fresh_start and processed_false_functions:
                        pointer_typedef_aliases = getattr(self.parser, "pointer_typedef_aliases", set())  # type: ignore[attr-defined]
                        classes = self._compute_llm_filter_classes(pointer_typedef_aliases)
                        kept = set(classes.get("kept", []) or [])
                        retry_candidates = kept & set(processed_false_functions)
                        logger.info(
                            "  Resume plan: will retry %d function(s) (processed=false last run AND still LLM-candidate); "
                            "skip processed_true=%d, reuse partial_hints=%d",
                            len(retry_candidates),
                            len(processed_functions),
                            len(partial_hints.hints) if (partial_hints and getattr(partial_hints, "hints", None)) else 0,
                        )
                except Exception as e:
                    logger.warning("  Warning: failed to compute resume retry count: %s", e)

                self.hints = self.hint_generator.generate_hints(
                    self.functions,
                    macro_names=self.macro_names,
                    pointer_typedef_aliases=pointer_typedef_aliases,
                    progress_callback=_save_partial_hints,
                    existing_hints=partial_hints,
                    processed_functions=processed_functions,
                    batch_size=self.hint_batch_size,
                )
                logger.info(f"  {self.hints.summary()}")

                # Export raw per-function hint results (including functions with no hints)
                # to a separate JSON file so users can inspect which functions were
                # analyzed and which ones have no allocator/deallocator semantics.
                try:
                    all_results = getattr(self.hint_generator, "last_all_results", None)
                    processed = getattr(self.hint_generator, "processed_functions", set())
                    failed = getattr(self.hint_generator, "failed_functions", set())
                    if all_results is not None:
                        # Only export the LLM-candidate functions into hints_all_raw.json (see _save_partial_hints).
                        classes = getattr(self.hint_generator, "last_filter_classes", None) or {}
                        kept = classes.get("kept", []) if isinstance(classes, dict) else []
                        candidate_only: set[str] = set(kept or [])

                        raw_data: dict[str, dict] = {"hints": {}}
                        for fn, func_hints in all_results.items():
                            if candidate_only and fn not in candidate_only:
                                continue
                            is_processed = not processed or fn in processed

                            if func_hints and is_processed:
                                raw_data["hints"][fn] = [
                                    {
                                        "name": fn,
                                        "role": "Allocator" if h.hint_type.name == "ALLOCATOR" else "Deallocator",
                                        "target": h.target,
                                        "processed": True,
                                    }
                                    for h in func_hints
                                ]
                            elif is_processed:
                                raw_data["hints"][fn] = [
                                    {
                                        "name": fn,
                                        "role": "none",
                                        "target": "none",
                                        "processed": True,
                                    }
                                ]
                            else:
                                raw_data["hints"][fn] = [
                                    {
                                        "name": fn,
                                        "role": "none",
                                        "target": "none",
                                        "processed": False,
                                    }
                                ]
                        raw_path = cache_dir / "hints_all_raw.json"
                        raw_path.write_text(json.dumps(raw_data, indent=2))
                        logger.info(f"  Saved raw per-function hint results (including both_not) to {raw_path}")
                except Exception as e:
                    logger.warning(f"  Warning: failed to export raw per-function hint results: {e}")

                # =================================================================
                # Phase 3: Z3 validates hints
                # =================================================================
                _start_phase("Phase3_Z3_Validation")
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

                    if self.use_llm_relabel:
                        logger.info(f"Phase 2: Regenerating hints for {len(conflict_functions)} conflict function(s) (use_llm_relabel=True)...")
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
                        logger.info(f"  After regeneration: {self.hints.summary()}")
                    else:
                        # USE_LLM_RELABEL=false: Skip LLM regeneration, keep only validated hints
                        logger.info(f"  Skipping LLM regeneration (use_llm_relabel=False). Keeping validated hints only.")
                        logger.info(f"  Final hints after validation: {self.hints.summary()}")
                else:
                    # No conflicts, we're done
                    logger.info(f"  No conflicts found! Hints are consistent.")
                    logger.info(f"  Final hints: {self.hints.summary()}")

                self.conflicts.extend(all_conflicts)

                # Save validated hints to current output directory (even if loaded from CodeQL directory)
                current_hints_file = cache_dir / "hints.json"
                self._save_hints(current_hints_file)
                
                # =================================================================
                # Phase 2b: Generate custom queries for special functions (after Z3 validation)
                # Note: Custom queries are analyzer-specific (CodeQL queries vs Infer models)
                # =================================================================
                if self.use_custom_queries:
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
                    custom_queries_file = cache_dir / f"custom_queries{analyzer_suffix}.json"
                    self._save_custom_queries(custom_queries_file)
                else:
                    # Custom queries disabled
                    logger.info(f"Phase 2b: Skipping custom query generation (use_custom_queries=False)")
                    self.custom_queries = CustomQuerySet()
                
                # Update validated hints count and export cost information
                # right after hint generation/validation completes.
                self.validated_hints_count = sum(
                    len(func_hints) for func_hints in self.hints.hints.values()
                )
                self._export_costs(cache_dir)

        # If the user requested only the hint-generation pipeline, stop here.
        if self.pipeline_mode == "hints-only":
            logger.info("Pipeline mode 'hints-only': skipping analyzer and warning validation phases.")
            return AnalysisResult(
                confirmed_bugs=[],
                hints=self.hints,
                iterations=1,
                spurious_filtered=0,
            )

        # =====================================================================
        # Skip Phase 4 & 5 if results already exist; run Phase 6 (LLM verify) directly
        # =====================================================================
        bugs_file = output_dir / "memory_safety_bugs.json"
        if bugs_file.is_file():
            logger.info("memory_safety_bugs.json already exists: skipping Phase 4 and 5.")
            llm_verify_summary = None
            if self.use_llm_verify_bugs:
                logger.info("Phase 6: Running LLM verify on existing bugs...")
                try:
                    from src.verify_bugs_llm import run_verify
                    verify_result = run_verify(
                        output_dir, project_path, self.llm_verify_model
                    )
                    if verify_result:
                        out_file = output_dir / "llm_verify_bugs.json"
                        out_file.write_text(
                            json.dumps(verify_result, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        llm_verify_summary = verify_result.get("summary", {})
                        logger.info(
                            "  LLM verify done: %s (TP=%s, FP=%s, ERROR=%s, kept=%s)",
                            out_file,
                            llm_verify_summary.get("tp", 0),
                            llm_verify_summary.get("fp", 0),
                            llm_verify_summary.get("error", 0),
                            llm_verify_summary.get("should_keep", 0),
                        )
                except Exception as e:
                    logger.warning("LLM verify failed: %s", e)
            return AnalysisResult(
                confirmed_bugs=[],
                hints=self.hints,
                iterations=1,
                spurious_filtered=0,
                skipped_phases_4_5=True,
                llm_verify_summary=llm_verify_summary,
            )

        # =====================================================================
        # Phase 4: Static analyzer analysis with hints and custom queries
        # =====================================================================
        analyzer_name = "Infer" if self.analyzer_type == "infer" else "CodeQL"
        _start_phase("Phase4_SAST_Analysis")
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

        # Filter warnings to only include files within source_root.
        # The CodeQL DB may be built from compile_commands.json that includes
        # files outside source_root (e.g. host tools captured by bear), but
        # hints and Z3 analysis only cover source_root files.
        if source_root != project_path:
            def _within_source_root(w) -> bool:
                fp = w.file_path or ""
                abs_fp = Path(fp) if Path(fp).is_absolute() else project_path / fp
                try:
                    abs_fp.relative_to(source_root)
                    return True
                except ValueError:
                    return False
            before = len(warnings)
            warnings = [w for w in warnings if _within_source_root(w)]
            if len(warnings) < before:
                logger.info(
                    f"  Source-root filter: kept {len(warnings)}/{before} warnings "
                    f"(removed {before - len(warnings)} outside {source_root})"
                )

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
        _start_phase("Phase5_Z3_Filtering")
        if self.skip_z3:
            logger.info("Phase 5: Z3 filtering SKIPPED (--skip-z3). Treating all warnings as confirmed.")
            confirmed_warnings = warnings
            filtered_warnings = []
            unconfirmed_warnings = []
        else:
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
                z3_validated=not self.skip_z3,
            )
            for w in confirmed_warnings
        ]

        # =====================================================================
        # Export results
        # =====================================================================
        self._export_results(output_dir, confirmed_bugs, filtered_warnings, unconfirmed_warnings)

        # =====================================================================
        # Phase 6: LLM verify bugs (does this function really have this bug?)
        # =====================================================================
        _start_phase("Phase6_LLM_Verify")
        if self.use_llm_verify_bugs and (output_dir / "memory_safety_bugs.json").is_file():
            try:
                from src.verify_bugs_llm import run_verify
                verify_result = run_verify(
                    output_dir, project_path, self.llm_verify_model
                )
                if verify_result:
                    out_file = output_dir / "llm_verify_bugs.json"
                    out_file.write_text(
                        json.dumps(verify_result, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    s = verify_result.get("summary", {})
                    logger.info(
                        "LLM verify: %s (TP=%s, FP=%s, ERROR=%s, kept=%s)",
                        out_file,
                        s.get("tp", 0),
                        s.get("fp", 0),
                        s.get("error", 0),
                        s.get("should_keep", 0),
                    )
            except Exception as e:
                logger.warning("LLM verify failed: %s", e)

        # End timing and save summary
        _end_all_phases()

        # Collect LLM costs per phase
        hint_cost = 0.0
        hint_tokens = 0
        verify_cost = 0.0
        verify_tokens = 0

        # Phase 2 (Hint generation) cost
        if hasattr(self, 'hint_generator') and self.hint_generator:
            cost_summary = self.hint_generator.get_cost_summary()
            hint_cost = cost_summary.get("total_cost", 0.0)
            hint_tokens = cost_summary.get("total_tokens", 0)

        # Phase 6 (LLM verification) cost — read from llm_verify_bugs.json if exists
        verify_file = output_dir / "llm_verify_bugs.json"
        if verify_file.exists():
            try:
                verify_data = json.loads(verify_file.read_text())
                verify_summary = verify_data.get("summary", {})
                verify_cost = verify_summary.get("actual_cost_usd", 0.0)
                verify_tokens = (
                    verify_summary.get("actual_prompt_tokens", 0)
                    + verify_summary.get("actual_completion_tokens", 0)
                )
            except Exception:
                pass

        # Save combined pipeline stats
        pipeline_stats = {
            "phase_timings_seconds": phase_timings,
            "llm_costs": {
                "phase2_hint_generation": {
                    "cost_usd": round(hint_cost, 4),
                    "tokens": hint_tokens,
                },
                "phase6_bug_verification": {
                    "cost_usd": round(verify_cost, 4),
                    "tokens": verify_tokens,
                },
                "total_cost_usd": round(hint_cost + verify_cost, 4),
                "total_tokens": hint_tokens + verify_tokens,
            },
        }
        stats_path = output_dir / "pipeline_stats.json"
        stats_path.write_text(json.dumps(pipeline_stats, indent=2))
        logger.info(f"[TIMING] Pipeline stats saved to {stats_path}")
        logger.info(f"[COST] Hint generation: ${hint_cost:.4f} ({hint_tokens} tokens)")
        logger.info(f"[COST] Bug verification: ${verify_cost:.4f} ({verify_tokens} tokens)")
        logger.info(f"[COST] Total LLM cost: ${hint_cost + verify_cost:.4f}")

        return AnalysisResult(
            confirmed_bugs=confirmed_bugs,
            hints=self.hints,
            iterations=1,
            spurious_filtered=len(filtered_warnings),
        )

    def _find_hints_file(self, output_dir: Path) -> Path | None:
        """Find hints.json file with fallback logic.
        
        Both CodeQL and Infer can copy hints from each other:
        - Checks current analyzer's directory first
        - Then checks the other analyzer's equivalent directory
        
        Args:
            output_dir: Current analyzer's output directory
            
        Returns:
            Path to hints.json if found, None otherwise
        """
        # First, check current analyzer's output directory
        hints_file = output_dir / "hints.json"
        if hints_file.exists():
            return hints_file
        
        # Try to find the other analyzer's equivalent directory
        output_str = str(output_dir)
        other_analyzer_dir = None
        other_analyzer_name = None
        
        if self.analyzer_type == "infer":
            # Infer: look for CodeQL directory (remove "_infer" suffix)
            if "_infer" in output_str:
                codeql_output_str = output_str.replace("_infer", "")
                other_analyzer_dir = Path(codeql_output_str)
                other_analyzer_name = "CodeQL"
        else:
            # CodeQL: look for Infer directory (add "_infer" suffix)
            if not output_str.endswith("_infer"):
                infer_output_str = output_str + "_infer"
                other_analyzer_dir = Path(infer_output_str)
                other_analyzer_name = "Infer"
        
        if other_analyzer_dir:
            other_hints_file = other_analyzer_dir / "hints.json"
            if other_hints_file.exists():
                logger.info(f"  Hints not found in {self.analyzer_type} directory ({output_dir}), checking {other_analyzer_name} equivalent: {other_hints_file}")
                return other_hints_file
        
        # Also try looking in parent directory for sibling directories
        parent_dir = output_dir.parent
        if parent_dir.exists():
            base_name = output_dir.name
            if self.analyzer_type == "infer" and "_infer" in base_name:
                # Infer: look for CodeQL sibling (remove "_infer")
                codeql_base_name = base_name.replace("_infer", "")
                codeql_output_dir = parent_dir / codeql_base_name
                codeql_hints_file = codeql_output_dir / "hints.json"
                if codeql_hints_file.exists():
                    logger.info(f"  Hints not found in Infer directory, checking CodeQL sibling: {codeql_hints_file}")
                    return codeql_hints_file
            elif self.analyzer_type == "codeql" and not base_name.endswith("_infer"):
                # CodeQL: look for Infer sibling (add "_infer")
                infer_base_name = base_name + "_infer"
                infer_output_dir = parent_dir / infer_base_name
                infer_hints_file = infer_output_dir / "hints.json"
                if infer_hints_file.exists():
                    logger.info(f"  Hints not found in CodeQL directory, checking Infer sibling: {infer_hints_file}")
                    return infer_hints_file
        
        return None

    def _find_custom_queries_file(self, output_dir: Path) -> Path | None:
        """Find custom queries file with fallback logic.
        
        Both CodeQL and Infer can copy custom queries from each other:
        - Checks current analyzer's directory first
        - Then checks the other analyzer's equivalent directory
        
        Args:
            output_dir: Current analyzer's output directory
            
        Returns:
            Path to custom queries file if found, None otherwise
        """
        # Build analyzer-specific filename
        analyzer_suffix = f"_{self.analyzer_type}" if self.analyzer_type != "codeql" else ""
        custom_queries_file = output_dir / f"custom_queries{analyzer_suffix}.json"
        
        # First, check current analyzer's file
        if custom_queries_file.exists():
            return custom_queries_file
        
        # Try to find the other analyzer's equivalent directory
        output_str = str(output_dir)
        other_analyzer_dir = None
        other_analyzer_name = None
        other_suffix = ""
        
        if self.analyzer_type == "infer":
            # Infer: look for CodeQL directory (remove "_infer" suffix)
            if "_infer" in output_str:
                codeql_output_str = output_str.replace("_infer", "")
                other_analyzer_dir = Path(codeql_output_str)
                other_analyzer_name = "CodeQL"
                other_suffix = ""  # CodeQL uses no suffix
        else:
            # CodeQL: look for Infer directory (add "_infer" suffix)
            if not output_str.endswith("_infer"):
                infer_output_str = output_str + "_infer"
                other_analyzer_dir = Path(infer_output_str)
                other_analyzer_name = "Infer"
                other_suffix = "_infer"  # Infer uses "_infer" suffix
        
        if other_analyzer_dir:
            other_custom_queries_file = other_analyzer_dir / f"custom_queries{other_suffix}.json"
            if other_custom_queries_file.exists():
                logger.info(f"  Custom queries not found in {self.analyzer_type} directory ({output_dir}), checking {other_analyzer_name} equivalent: {other_custom_queries_file}")
                return other_custom_queries_file
        
        # Also try looking in parent directory for sibling directories
        parent_dir = output_dir.parent
        if parent_dir.exists():
            base_name = output_dir.name
            if self.analyzer_type == "infer" and "_infer" in base_name:
                # Infer: look for CodeQL sibling (remove "_infer")
                codeql_base_name = base_name.replace("_infer", "")
                codeql_output_dir = parent_dir / codeql_base_name
                codeql_custom_queries_file = codeql_output_dir / "custom_queries.json"
                if codeql_custom_queries_file.exists():
                    logger.info(f"  Custom queries not found in Infer directory, checking CodeQL sibling: {codeql_custom_queries_file}")
                    return codeql_custom_queries_file
            elif self.analyzer_type == "codeql" and not base_name.endswith("_infer"):
                # CodeQL: look for Infer sibling (add "_infer")
                infer_base_name = base_name + "_infer"
                infer_output_dir = parent_dir / infer_base_name
                infer_custom_queries_file = infer_output_dir / "custom_queries_infer.json"
                if infer_custom_queries_file.exists():
                    logger.info(f"  Custom queries not found in CodeQL directory, checking Infer sibling: {infer_custom_queries_file}")
                    return infer_custom_queries_file
        
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
        # Confirmed bugs - group by type, then by function
        bugs_data = {}
        all_functions = set()  # Track all unique functions
        
        for bug in confirmed:
            t = bug.warning.issue_type.name
            if t not in bugs_data:
                bugs_data[t] = {}
            
            func_name = bug.warning.function_name
            all_functions.add(func_name)
            
            # Group by function name
            if func_name not in bugs_data[t]:
                bugs_data[t][func_name] = {
                    "file": bug.warning.file_path,
                    "function": func_name,
                    "bug_count": 0,
                    "bugs": []
                }
            
            # Add bug details to the function's bug list
            bugs_data[t][func_name]["bugs"].append({
                "line": bug.warning.line_number,
                "warning_type": bug.warning.warning_type,
                "type": bug.warning.issue_type.name,
                "message": bug.warning.message,
                "allocation_site": bug.warning.allocation_site,
                "trace": bug.warning.trace,
                "suggested_fix": bug.suggested_fix,
            })
            bugs_data[t][func_name]["bug_count"] += 1
        
        # Convert to original format but with merged functions
        output_data = {
            "total_functions": len(all_functions)
        }
        
        # Add each bug type as a list of merged function entries
        for bug_type, func_dict in bugs_data.items():
            output_data[bug_type] = list(func_dict.values())

        bugs_file = output_dir / "memory_safety_bugs.json"
        bugs_file.write_text(json.dumps(output_data, indent=2))

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
            "estimated_prompt_tokens": cost_summary.get("estimated_prompt_tokens", 0),
            "estimated_calls": cost_summary.get("estimated_calls", 0),
            "estimated_input_cost": cost_summary.get("estimated_input_cost", 0.0),
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
            MemoryIssueType.MEMORY_LEAK_FILTERED: "Free allocated memory before function exit",
        }
        return fixes.get(warning.issue_type, "Review memory safety issue")

    def _generate_custom_queries_iteratively(
        self,
        functions: dict[str, FunctionInfo],
        hints: HintSet,
    ) -> CustomQuerySet:
        """Generate custom CodeQL queries for validated hints iteratively (one by one).
        
        After Z3 validation confirms the hints, evaluate each function that has validated
        hints to determine if it needs a custom query. Process them one by one with logging.
        
        All evaluated functions are stored in CustomQuerySet with their decisions:
        - Special functions: includes generated filter predicates and reasoning
        - Non-special functions: includes reasoning for why no query was needed
        
        Args:
            functions: Dict of all functions
            hints: The validated hints (after Z3 validation)
            
        Returns:
            CustomQuerySet with evaluation results for all validated functions
        """
        custom_queries = CustomQuerySet()
        
        # If no LLM client is available, return an empty set.
        if not self.hint_generator.llm:
            logger.warning("No LLM client available for custom query generation")
            return custom_queries
        
        # Create a dedicated CustomQueryGenerator for this iteration.
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