"""CodeQL and Infer Analyzers with Enhanced Queries.

This module provides:
1. CodeQL database creation and management
2. Custom memory model injection
3. Enhanced memory safety queries (hardcoded improvements over standard queries)
4. Infer analyzer integration
"""

import json
import logging
import os
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
    # Enhanced queries
    "cpp/memory-never-freed-enhanced": MemoryIssueType.MEMORY_LEAK,
    "cpp/memory-may-not-be-freed-enhanced": MemoryIssueType.MEMORY_LEAK,
    # Filtered queries
    "cpp/memory-never-freed-enhanced-filtered": MemoryIssueType.MEMORY_LEAK_FILTERED,
    "cpp/memory-may-not-be-freed-enhanced-filtered": MemoryIssueType.MEMORY_LEAK_FILTERED,
}

# Standard memory queries (memory leak only, per paper)
MEMORY_QUERIES = [
    "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
    "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
]


# =============================================================================
# Enhanced Queries (Hardcoded)
# =============================================================================

ENHANCED_MEMORY_NEVER_FREED = (
    (Path(__file__).parent / "queries" / "ENHANCED_MEMORY_NEVER_FREED.ql")
    .read_text(encoding="utf-8")
)

ENHANCED_MEMORY_MAY_NOT_BE_FREED = (
    (Path(__file__).parent / "queries" / "ENHANCED_MEMORY_MAY_NOT_BE_FREED.ql")
    .read_text(encoding="utf-8")
)

ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE = (
    (Path(__file__).parent / "queries" / "ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE.ql")
    .read_text(encoding="utf-8")
)

ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE = (
    (Path(__file__).parent / "queries" / "ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE.ql")
    .read_text(encoding="utf-8")
)

ENHANCED_MEMORY_NEVER_FREED_FILTERED = ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE
ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED = ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE

# Map query names to enhanced versions
ENHANCED_QUERIES = {
    "MemoryNeverFreed": ENHANCED_MEMORY_NEVER_FREED,
    "MemoryMayNotBeFreed": ENHANCED_MEMORY_MAY_NOT_BE_FREED,
}

# Base (built-in) filtered query text; may be extended with LLM-generated filters at runtime.
ENHANCED_QUERIES_FILTERED = {
    "MemoryNeverFreed": ENHANCED_MEMORY_NEVER_FREED_FILTERED,
    "MemoryMayNotBeFreed": ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED,
}


class CodeQLAnalyzer:
    """CodeQL analyzer with model injection and enhanced queries."""

    def __init__(
        self,
        binary: str = "codeql",
        timeout: int = 18000,
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
        # Dynamic enhanced filtered queries; may be overridden
        # with LLM-generated filters at analysis time.
        self._memory_never_freed_filtered_query: str | None = None
        self._memory_may_not_be_freed_filtered_query: str | None = None

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
                self._memory_never_freed_filtered_query = self._build_memory_never_freed_filtered_with_custom_filters(
                    custom_queries
                )
                self._memory_may_not_be_freed_filtered_query = self._build_memory_may_not_be_freed_filtered_with_custom_filters(
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
              if query_name == "MemoryNeverFreed":
                  qtext = self._memory_never_freed_filtered_query
                  label = "MemoryNeverFreed"
              elif query_name == "MemoryMayNotBeFreed":
                  qtext = self._memory_may_not_be_freed_filtered_query
                  label = "MemoryMayNotBeFreed"
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

        # Prefer an explicit build_command from config so CodeQL can trace
        # compilation itself. Only fall back to compile_commands.json if no
        # build command is available (--compilation-database is not supported
        # in CodeQL 2.x for C/C++; --build-mode=none is also unavailable).
        build_cmd = None
        if config and "build_command" in config:
            bc = config["build_command"]
            build_cmd = bc.get("command") if isinstance(bc, dict) else bc

        if build_cmd:
            logger.info(f"Using build command for CodeQL database creation: {build_cmd}")
            cmd = [
                self.binary, "database", "create", str(db_path),
                f"--source-root={project_path}",
                "--language=cpp", "--overwrite",
                "--command", build_cmd
            ]
        else:
            build_cmd = self._detect_build_command(project_path)
            if not build_cmd:
                logger.error("Could not determine build command")
                return False

            logger.info(f"Using detected build command for CodeQL database creation: {build_cmd}")
            cmd = [
                self.binary, "database", "create", str(db_path),
                f"--source-root={project_path}",
                "--language=cpp", "--overwrite",
                "--command", build_cmd
            ]

        logger.info(f"Running CodeQL database create: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            timeout=self.timeout,
            capture_output=True,
            text=True,
            cwd=str(project_path),
        )

        # Always log build stdout/stderr from CodeQL database create so user can see make output.
        if result.stdout:
            logger.info(
                "CodeQL build stdout:\n%s",
                result.stdout,
            )
        if result.stderr:
            logger.info(
                "CodeQL build stderr:\n%s",
                result.stderr,
            )

        if result.returncode != 0:
            # Check if a partial database was created (e.g. one file failed
            # but most compiled successfully).  If TRAP data exists we can
            # still finalize and analyze the database.
            trap_dir = db_path / "trap"
            if trap_dir.exists() and any(trap_dir.rglob("*.trap.tar.zst")):
                logger.warning(
                    f"Build exited with code {result.returncode} but partial "
                    f"database exists at {db_path} — attempting to finalize."
                )
            else:
                logger.error(
                    f"Database creation failed (exit {result.returncode})"
                )
                return False

        return self._finalize_database(db_path)

    def _load_build_config(self, project_path: Path) -> Optional[dict]:
        config_file = Path(__file__).parent.parent.parent / "proj_build_command.json"
        if not config_file.exists():
            return None
        try:
            with open(config_file) as f:
                all_cfg = json.load(f)

            # Prefer an explicit project key provided via environment, so that
            # callers (e.g. run.sh cases like linux_v7_0-rc1_drivers_peci) can
            # share / override build settings by "project name" instead of just
            # the leaf directory.
            env_key = os.getenv("HINT_PROJECT_NAME") or os.getenv("HINT_BUILD_CONFIG_KEY")
            if env_key and env_key in all_cfg:
                logger.info(
                    "Loaded build config using env key %s for project %s",
                    env_key,
                    project_path,
                )
                return all_cfg.get(env_key)

            # Fallback: use the leaf directory name for backward compatibility.
            return all_cfg.get(project_path.name)
        except Exception:
            return None

    def _run_commands(self, cwd: Path, commands) -> None:
        if isinstance(commands, str):
            commands = [{"command": c.strip(), "can_error": True} for c in commands.split("&&")]
        elif isinstance(commands, list):
            commands = [
                {
                    "command": c.get("command", c) if isinstance(c, dict) else c,
                    "can_error": c.get("can_error", True) if isinstance(c, dict) else True,
                }
                for c in commands
            ]

        for cmd_info in commands:
            cmd = cmd_info.get("command")
            if not cmd:
                continue

            can_error = cmd_info.get("can_error", True)
            logger.info(f"Running pre-build command in {cwd}: {cmd} (can_error={can_error})")
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd),
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )

            # Log stdout/stderr so user can see actual build output.
            if result.stdout:
                logger.info(
                    "Pre-build command stdout:\n%s",
                    result.stdout,
                )
            if result.stderr:
                logger.info(
                    "Pre-build command stderr:\n%s",
                    result.stderr,
                )

            if result.returncode != 0:
                msg = (
                    f"Pre-build command failed with exit code {result.returncode}: {cmd}"
                )
                if can_error:
                    logger.warning(msg)
                else:
                    logger.error(msg)

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

        The allocationFunctionModel columns are:
          [namespace, type, subtypes, name, sizeArg, sizeMult, reallocPtrArg, requiresDealloc]
        For wrapper allocators that return heap memory, leave the numeric columns as ""
        (unknown size). requiresDealloc=True tells CodeQL to flag leaked return values.

        Args:
            funcs: List of (function_name, arg_index) tuples. arg_index is
                   preserved from the summary but not emitted into the YAML
                   — CodeQL allocators implicitly return via return value.
        """
        lines = [
            "extensions:",
            "  - addsTo:",
            "      pack: codeql/cpp-all",
            "      extensible: allocationFunctionModel",
            "    data:"
        ]
        for name, _idx in funcs:
            lines.append(f'      - ["", "", False, "{name}", "", "", "", True]')
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


# =============================================================================
# Infer Analyzer (Microsoft Infer / Facebook Infer)
# =============================================================================

# Mapping from Infer issue types to MemoryIssueType
# Based on Infer documentation: https://fbinfer.com/docs/all-categories/
INFER_ISSUE_MAP = {
    # Memory Leak / Resource Leak (Resource leak category)
    "MEMORY_LEAK": MemoryIssueType.MEMORY_LEAK,
    "MEMORY_LEAK_C": MemoryIssueType.MEMORY_LEAK,
    "MEMORY_LEAK_CPP": MemoryIssueType.MEMORY_LEAK,
    "BIABDUCTION_MEMORY_LEAK": MemoryIssueType.MEMORY_LEAK,
    "PULSE_RESOURCE_LEAK": MemoryIssueType.MEMORY_LEAK,
    "RESOURCE_LEAK": MemoryIssueType.MEMORY_LEAK,
    "BIABDUCTION_RETAIN_CYCLE": MemoryIssueType.MEMORY_LEAK,
    "RETAIN_CYCLE": MemoryIssueType.MEMORY_LEAK,
    "RETAIN_CYCLE_NO_WEAK_INFO": MemoryIssueType.MEMORY_LEAK,
    "CAPTURED_STRONG_SELF": MemoryIssueType.MEMORY_LEAK,
    "MIXED_SELF_WEAKSELF": MemoryIssueType.MEMORY_LEAK,
    "CHECKERS_FRAGMENT_RETAINS_VIEW": MemoryIssueType.MEMORY_LEAK,
    "PULSE_UNAWAITED_AWAITABLE": MemoryIssueType.MEMORY_LEAK,
    
}

# Default Infer flags used across all projects (no need to configure per-project)
INFER_DEFAULT_FLAGS: list[str] = ["--keep-going", "--pulse-only", "--debug-level", "2"]


def _normalize_hint_names(raw: list, skip: tuple = ("main", "_main", "")) -> list[str]:
    """
    Normalize a hints list that may contain either plain strings or (name, idx)
    tuples/lists into a flat list of function name strings, filtering out
    entries whose name is in `skip`.

    This helper is used in InferAnalyzer._run_infer to safely extract function
    names for --pulse-model-alloc-pattern / --pulse-model-free-pattern without
    crashing when hints.get_allocators() returns plain strings instead of tuples.
    """
    names: list[str] = []
    for item in (raw or []):
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            name = str(item[0])
        else:
            name = str(item)
        if name not in skip:
            names.append(name)
    return names


class InferAnalyzer:
    """Infer analyzer with hint support via --pulse-model-* flags.

    The _inject_models method that wrote fake .models files has been removed.
    Infer does not read those files; the only supported mechanism for teaching
    Infer about custom allocators/deallocators at runtime is through the CLI
    flags --pulse-model-alloc-pattern and --pulse-model-free-pattern, which
    are built and passed inside _run_infer.
    """

    def __init__(
        self,
        binary: str = "infer",
        timeout: int | None = None,
        infer_dir: Path | None = None,
        reuse_db: bool = True,
    ):
        """Initialize Infer analyzer.
        
        Args:
            binary: Path to infer binary (default: "infer")
            timeout: Timeout for infer commands in seconds
            infer_dir: Optional custom Infer installation directory
            reuse_db: Whether to reuse existing Infer results (default: True)
        """
        self.binary = binary
        self.timeout = timeout
        self.infer_dir = infer_dir
        self.reuse_db = reuse_db
        self._irrelevant_warnings: list[Warning] = []  # Warnings not related to memory safety

    def analyze(
        self,
        project_path: Path,
        hints: HintSet | None = None,
        issue_types: list[MemoryIssueType] | None = None,
        use_enhanced_queries: bool = True,
        custom_queries: CustomQuerySet | None = None,
    ) -> list[Warning]:
        """
        Run Infer analysis.

        Hints (custom allocators/deallocators) are passed directly to Infer via
        --pulse-model-alloc-pattern and --pulse-model-free-pattern flags inside
        _run_infer.  There is no file-based injection step.

        When use_enhanced_queries is True, the enhanced Pulse checker
        (PulseEnhancedModels.ml) is compiled into the local Infer build,
        providing error-path leak detection, loop-leak detection,
        conditional double-free, and use-after-conditional-free checks
        parallel to CodeQL's enhanced .ql queries.

        Args:
            project_path: Project to analyze
            hints: Allocator/deallocator hints
            issue_types: Which issue types to map/return
            use_enhanced_queries: If True, use Infer build with enhanced checkers
            custom_queries: Not used by Infer (kept for compatibility)
        """
        output_dir = Path(tempfile.mkdtemp())
        results_path = output_dir / "infer_results.json"

        try:
            # Run Infer analysis; hints are forwarded as pulse-model-* flags
            logger.info("Running Infer analysis...")
            success = self._run_infer(project_path, results_path, hints)

            if not success:
                return []

            warnings = self._parse_infer_results(results_path, project_path, issue_types)
            return warnings

        except Exception as e:
            logger.error(f"Infer analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _load_build_config(self, project_path: Path) -> Optional[dict]:
        """Load Infer-specific build configuration from proj_build_command_infer.json."""
        config_file = Path(__file__).parent.parent.parent / "proj_build_command_infer.json"
        if not config_file.exists():
            logger.debug(f"Infer build config file not found: {config_file}")
            return None
        try:
            with open(config_file) as f:
                return json.load(f).get(project_path.name)
        except Exception as e:
            logger.warning(f"Failed to load Infer build config: {e}")
            return None

    def _run_commands(self, cwd: Path, commands) -> None:
        """Run prepare_for_build commands."""
        if isinstance(commands, str):
            commands = [{"command": c.strip(), "can_error": True} for c in commands.split("&&")]
        elif isinstance(commands, list):
            commands = [
                {"command": c.get("command", c) if isinstance(c, dict) else c,
                 "can_error": c.get("can_error", True) if isinstance(c, dict) else True}
                for c in commands
            ]

        command_list = [cmd_info["command"] for cmd_info in commands if cmd_info["command"]]
        if command_list:
            logger.info(f"Pre-build commands to execute ({len(command_list)}):")
            for i, cmd in enumerate(command_list, 1):
                logger.info(f"  [{i}] {cmd}")

        for cmd_info in commands:
            if cmd_info["command"]:
                logger.info(f"Executing pre-build command: {cmd_info['command']}")
                result = subprocess.run(cmd_info["command"], shell=True, cwd=str(cwd),
                                       timeout=self.timeout, capture_output=True, text=True)
                if result.returncode != 0:
                    if cmd_info.get("can_error", True):
                        logger.warning(f"Pre-build command failed (allowed): {cmd_info['command']}")
                        if result.stderr:
                            logger.debug(f"Error output: {result.stderr[:200]}")
                    else:
                        logger.error(f"Pre-build command failed (not allowed): {cmd_info['command']}")
                        if result.stderr:
                            logger.error(f"Error output: {result.stderr[:500]}")
                else:
                    logger.info(f"Pre-build command succeeded: {cmd_info['command']}")

    def _write_inferconfig(self, project_path: Path, config: Optional[dict], pulse_flags: list[str]) -> None:
        """
        (Re)generate a minimal .inferconfig JSON file for the project containing
        only pulse-model-malloc/free patterns. The actual Infer CLI invocation
        is always done as `infer run <default flags> -- <build_cmd>`.
        """
        try:
            infer_cfg: dict[str, object] = {}

            # Extract pulse-model patterns from CLI-style flags
            for flag in pulse_flags or []:
                if flag.startswith("--pulse-model-alloc-pattern="):
                    infer_cfg["pulse-model-alloc-pattern"] = flag.split("=", 1)[1]
                elif flag.startswith("--pulse-model-free-pattern="):
                    infer_cfg["pulse-model-free-pattern"] = flag.split("=", 1)[1]

            inferconfig_path = project_path / ".inferconfig"
            inferconfig_path.write_text(json.dumps(infer_cfg, indent=2))
            logger.info(f"Wrote .inferconfig to {inferconfig_path}")
        except Exception as e:
            logger.warning(f"Failed to write .inferconfig: {e}")

    def _run_infer(self, project_path: Path, results_path: Path, hints: HintSet | None = None) -> bool:
        """Run Infer analysis on the project.

        Custom allocators/deallocators from hints are passed via
        --pulse-model-alloc-pattern / --pulse-model-free-pattern.
        hints.get_allocators() may return plain strings ("my_malloc") or
        (name, idx) tuples; both are handled by _normalize_hint_names.

        Args:
            project_path: Path to project
            results_path: Path to save JSON results
            hints: Optional hints for custom allocator/deallocator patterns

        Returns:
            True if analysis succeeded, False otherwise
        """
        # Always clean previous Infer results to ensure fresh analysis
        infer_out_dir = project_path / "infer-out"
        if infer_out_dir.exists():
            logger.info(f"Removing existing infer-out directory: {infer_out_dir}")
            shutil.rmtree(infer_out_dir, ignore_errors=True)

        # Remove any existing .inferconfig so we can regenerate a fresh one
        inferconfig_path = project_path / ".inferconfig"
        if inferconfig_path.exists():
            logger.info(f"Removing existing .inferconfig: {inferconfig_path}")
            try:
                inferconfig_path.unlink()
            except Exception:
                # Best-effort cleanup; continue even if deletion fails
                pass

        config = self._load_build_config(project_path)

        if config and "prepare_for_build" in config:
            prepare_commands = config["prepare_for_build"]
            if prepare_commands:
                logger.info("Running prepare_for_build commands for Infer...")
                self._run_commands(project_path, prepare_commands)

        # ---------------------------------------------------------------
        # Build pulse-model flags from hints.
        #
        # FIX: the original code did `for name, _ in allocators` which
        # crashes when get_allocators() returns plain strings instead of
        # tuples.  _normalize_hint_names() handles both formats safely.
        # ---------------------------------------------------------------
        pulse_flags: list[str] = []
        if hints:
            import re

            def escape_regex_special(name: str) -> str:
                special_chars = r'.^$*+?{}[]\|()'
                return "".join(
                    ("\\" + c if c in special_chars else c) for c in name
                )

            alloc_names = _normalize_hint_names(hints.get_allocators() or [])
            dealloc_names = _normalize_hint_names(hints.get_deallocators() or [])

            # Build patterns in the same style as:
            #   "pulse-model-deep-release-pattern": "dir1::...::deep_wait\\|not_captured::...::not_captured_deep_wait"
            # i.e. functions separated by "\|" inside a single regex string.
            if alloc_names:
                pattern = "\\|".join(escape_regex_special(n) for n in alloc_names)
                pulse_flags.append(f"--pulse-model-alloc-pattern={pattern}")
                logger.info(f"pulse-model-alloc-pattern for: {', '.join(alloc_names)} -> {pattern}")

            if dealloc_names:
                pattern = "\\|".join(escape_regex_special(n) for n in dealloc_names)
                pulse_flags.append(f"--pulse-model-free-pattern={pattern}")
                logger.info(f"pulse-model-free-pattern for: {', '.join(dealloc_names)} -> {pattern}")

        # After computing pulse flags (and loading config earlier), regenerate .inferconfig
        # so that it reflects the current configuration and hint patterns.
        # NOTE: pulse-model patterns are stored in .inferconfig only, and are no longer
        # passed on the Infer CLI (so that `infer run -- <build>` is sufficient).
        self._write_inferconfig(project_path, config, pulse_flags)

        compile_commands = project_path / "compile_commands.json"
        if compile_commands.exists():
            logger.info("Using compile_commands.json for Infer")
            capture_cmd = [
                self.binary, "capture",
                "--compilation-database", str(compile_commands),
            ]

            logger.info(f"Running Infer capture command: {' '.join(capture_cmd)}")
            result = subprocess.run(
                capture_cmd,
                cwd=str(project_path),
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Infer capture failed: {result.stderr[:500] if result.stderr else 'No error message'}")
                logger.debug(f"Capture stdout: {result.stdout[:500] if result.stdout else 'No output'}")
            else:
                logger.info("Infer capture completed successfully")
                if infer_out_dir.exists():
                    files_count = len(list(infer_out_dir.rglob("*")))
                    logger.debug(f"Infer-out directory contains {files_count} files/directories")
                else:
                    logger.warning(f"Infer-out directory not found at {infer_out_dir} after capture")

            analyze_cmd = [self.binary, "analyze"] + INFER_DEFAULT_FLAGS
            logger.info(f"Running Infer analyze command: {' '.join(analyze_cmd)}")
            result = subprocess.run(
                analyze_cmd,
                cwd=str(project_path),
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Infer analyze failed: {result.stderr[:500] if result.stderr else 'No error message'}")
                logger.debug(f"Analyze stdout: {result.stdout[:500] if result.stdout else 'No output'}")
            else:
                logger.info("Infer analyze completed successfully")
        else:
            build_cmd = None
            infer_flags = list(INFER_DEFAULT_FLAGS)

            if config and "build_command" in config:
                bc = config["build_command"]
                if isinstance(bc, dict):
                    build_cmd = bc.get("command")
                else:
                    build_cmd = bc

            if not build_cmd:
                build_cmd = self._detect_build_command(project_path)

            if not build_cmd:
                logger.error("Could not determine build command for Infer")
                return False

            logger.info(f"Running Infer with build command: {build_cmd}")
            if infer_flags:
                logger.info(f"Using Infer flags: {infer_flags}")
            if pulse_flags:
                logger.info("Using pulse-model patterns via .inferconfig")

            build_cmd_parts = build_cmd.split() if isinstance(build_cmd, str) else build_cmd
            # FreeRDP_3_23_0: build_command is the full infer command, run it as-is
            if project_path.name == "FreeRDP_3_23_0" and build_cmd and (build_cmd if isinstance(build_cmd, str) else " ".join(build_cmd)).strip().startswith("infer run "):
                run_cmd_parts = build_cmd.split() if isinstance(build_cmd, str) else build_cmd
            else:
                run_cmd_parts = (
                    [self.binary, "run"]
                    + infer_flags
                    + ["--"]
                    + build_cmd_parts
                )

            logger.info(f"Running Infer command: {' '.join(run_cmd_parts)}")
            logger.info(f"Working directory: {project_path}")

            result = subprocess.run(
                run_cmd_parts,
                cwd=str(project_path),
                timeout=self.timeout,
                capture_output=True,
                text=True,
                shell=False,
                env=os.environ.copy(),
            )

            if not infer_out_dir.exists():
                logger.error("Infer-out directory not created - analysis may have failed")
                if result.returncode != 0:
                    logger.error(f"Infer run failed with return code {result.returncode}")
                    logger.debug(f"Stderr: {result.stderr[:1000] if result.stderr else 'No stderr'}")
                return False
            else:
                logger.info("Infer run completed - infer-out directory created")
                files_count = len(list(infer_out_dir.rglob("*")))
                logger.debug(f"Infer-out directory contains {files_count} files/directories")

        report_json = infer_out_dir / "report.json"
        if report_json.exists():
            logger.info(f"Found report.json at {report_json}")
            shutil.copy(report_json, results_path)
            try:
                json_data = json.loads(results_path.read_text())
                issue_count = len(json_data) if isinstance(json_data, list) else len(json_data.get("bugs", []))
                logger.info(f"Infer analysis found {issue_count} issues")
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in report.json: {e}")
                return False

        logger.info(f"report.json not found in {infer_out_dir}, generating JSON report...")
        report_cmd = [self.binary, "report", "--json"]

        logger.info(f"Running Infer report command: {' '.join(report_cmd)}")
        result = subprocess.run(
            report_cmd,
            cwd=str(project_path),
            timeout=self.timeout,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.warning(f"Infer report failed: {result.stderr[:500] if result.stderr else 'No error message'}")
            return False

        if result.stdout:
            try:
                json_data = json.loads(result.stdout)
                results_path.write_text(result.stdout)
                issue_count = len(json_data) if isinstance(json_data, list) else len(json_data.get("bugs", []))
                logger.info(f"Infer report generated: {issue_count} issues found")
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from infer report: {e}")
                logger.debug(f"JSON output (first 500 chars): {result.stdout[:500]}")
                return False
        else:
            logger.warning("Infer report produced no output")
            return False

    def _detect_build_command(self, project_path: Path) -> Optional[str]:
        """Detect build command for the project."""
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
            files = " ".join(str(f.relative_to(project_path)) for f in sources[:10])
            return f"clang -I. -c {files}"

        return None

    def _parse_infer_results(self, results_path: Path, project_path: Path, issue_types: list[MemoryIssueType] | None) -> list[Warning]:
        """Parse Infer JSON results into Warning objects.
        
        Args:
            results_path: Path to Infer JSON report
            project_path: Project root path
            issue_types: Optional filter for issue types
            
        Returns:
            List of Warning objects
        """
        if not results_path.exists():
            return []
        
        warnings: list[Warning] = []
        try:
            json_text = results_path.read_text()
            if not json_text.strip():
                logger.warning(f"Infer results file is empty: {results_path}")
                return []
            
            data = json.loads(json_text)
            
            if isinstance(data, dict):
                bugs = data.get("bugs", [])
                if not bugs and "bug_type" in data:
                    bugs = [data]
            elif isinstance(data, list):
                bugs = data
            else:
                logger.warning(f"Unexpected Infer JSON format: {type(data)}")
                return []
            
            if not bugs:
                logger.info("Infer found no issues")
                return []
            
            logger.info(f"Parsing {len(bugs)} Infer bug reports")
            
            unmapped_types = set()
            filtered_count = 0
            
            for bug in bugs:
                bug_type = bug.get("bug_type", "").upper()
                
                issue_type = INFER_ISSUE_MAP.get(bug_type)
                if not issue_type:
                    for key, val in INFER_ISSUE_MAP.items():
                        if key in bug_type:
                            issue_type = val
                            break
                
                is_memory_safety_related = issue_type is not None
                
                if not is_memory_safety_related:
                    unmapped_types.add(bug_type)
                    issue_type = MemoryIssueType.OTHER
                    logger.debug(f"Non-memory-safety bug type '{bug_type}' mapped to OTHER")
                
                if issue_types and issue_type not in issue_types:
                    filtered_count += 1
                    logger.debug(f"Filtered out bug type '{bug_type}' (mapped to {issue_type.name}) - not in requested types")
                    continue
                
                file_path = bug.get("file", "")
                line = bug.get("line", 0)
                function_name = bug.get("procedure", "")
                
                message = bug.get("qualifier", "") or bug.get("description", "") or bug.get("bug_type_hum", "") or bug_type
                
                bug_trace = bug.get("bug_trace", [])
                trace = []
                allocation_site = ""
                
                if bug_trace:
                    primary_loc = bug_trace[-1] if bug_trace else {}
                    if not file_path:
                        file_path = primary_loc.get("filename", "")
                    if not line:
                        line = primary_loc.get("line_number", 0)
                    
                    for trace_elem in bug_trace:
                        trace_file = trace_elem.get("filename", "")
                        trace_line = trace_elem.get("line_number", 0)
                        if trace_file and trace_line:
                            trace.append(f"{trace_file}:{trace_line}")
                    
                    if bug_trace:
                        first_loc = bug_trace[0]
                        alloc_file = first_loc.get("filename", "")
                        alloc_line = first_loc.get("line_number", 0)
                        if alloc_file and alloc_line:
                            allocation_site = f"{alloc_file}:{alloc_line}"
                
                if not trace and file_path and line:
                    trace.append(f"{file_path}:{line}")
                    allocation_site = f"{file_path}:{line}"
                
                warning = Warning(
                    file_path=file_path,
                    line_number=line,
                    function_name=function_name,
                    warning_type=bug_type,
                    message=message,
                    issue_type=issue_type,
                    allocation_site=allocation_site,
                    trace=trace,
                )
                
                warnings.append(warning)
            
            memory_safety_count = len([w for w in warnings if w.issue_type != MemoryIssueType.OTHER])
            other_count = len([w for w in warnings if w.issue_type == MemoryIssueType.OTHER])
            if unmapped_types:
                logger.info(f"Found {len(unmapped_types)} non-memory-safety bug type(s): {', '.join(sorted(unmapped_types))} (mapped to OTHER)")
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} bug(s) based on issue_types filter")
            if warnings:
                logger.info(f"Successfully parsed {len(warnings)} warning(s) from Infer results: {memory_safety_count} memory-safety, {other_count} other")
        
        except Exception as e:
            logger.error(f"Failed to parse Infer results: {e}")
            import traceback
            traceback.print_exc()

        return warnings