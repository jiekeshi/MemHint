"""CodeQL Analyzer with Enhanced Queries.

This module provides:
1. CodeQL database creation and management
2. Custom memory model injection
3. Enhanced memory safety queries (hardcoded improvements over standard queries)
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
    "double-free": MemoryIssueType.DOUBLE_FREE,
    "cpp/double-free": MemoryIssueType.DOUBLE_FREE,
    "use-after-free": MemoryIssueType.USE_AFTER_FREE,
    "cpp/use-after-free": MemoryIssueType.USE_AFTER_FREE,
    # Enhanced queries
    "cpp/memory-never-freed-enhanced": MemoryIssueType.MEMORY_LEAK,
    "cpp/memory-may-not-be-freed-enhanced": MemoryIssueType.MEMORY_LEAK,
    "cpp/double-free-enhanced": MemoryIssueType.DOUBLE_FREE,
    "cpp/use-after-free-enhanced": MemoryIssueType.USE_AFTER_FREE,
    # Filtered queries
    "cpp/memory-never-freed-enhanced-filtered": MemoryIssueType.MEMORY_LEAK_FILTERED,
    "cpp/memory-may-not-be-freed-enhanced-filtered": MemoryIssueType.MEMORY_LEAK_FILTERED,
    "cpp/double-free-enhanced-filtered": MemoryIssueType.DOUBLE_FREE_FILTERED,
    "cpp/use-after-free-enhanced-filtered": MemoryIssueType.USE_AFTER_FREE_FILTERED,
    
}

# Standard memory queries
MEMORY_QUERIES = [
    "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
    "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
    "codeql/cpp-queries:Critical/DoubleFree.ql",
    "codeql/cpp-queries:Critical/UseAfterFree.ql",
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

ENHANCED_DOUBLE_FREE =  (
    (Path(__file__).parent / "queries" / "ENHANCED_DOUBLE_FREE.ql")
    .read_text(encoding="utf-8")
)
ENHANCED_USE_AFTER_FREE = (
    (Path(__file__).parent / "queries" / "ENHANCED_USE_AFTER_FREE.ql")
    .read_text(encoding="utf-8")
)


ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE = (
    (Path(__file__).parent / "queries" / "ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE.ql")
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


ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE = (
    (Path(__file__).parent / "queries" / "ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE.ql")
    .read_text(encoding="utf-8")
)
ENHANCED_DOUBLE_FREE_FILTERED = ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE
ENHANCED_MEMORY_NEVER_FREED_FILTERED = ENHANCED_MEMORY_NEVER_FREED_FILTERED_TEMPLATE
ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED = ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED_TEMPLATE
ENHANCED_USE_AFTER_FREE_FILTERED = ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE

# Map query names to enhanced versions
ENHANCED_QUERIES = {
    "MemoryNeverFreed": ENHANCED_MEMORY_NEVER_FREED,
    "MemoryMayNotBeFreed": ENHANCED_MEMORY_MAY_NOT_BE_FREED,
    "DoubleFree": ENHANCED_DOUBLE_FREE,
    "UseAfterFree": ENHANCED_USE_AFTER_FREE,
}

# Base (built-in) filtered query text; may be extended with LLM-generated filters at runtime.
ENHANCED_QUERIES_FILTERED = {
    "MemoryNeverFreed": ENHANCED_MEMORY_NEVER_FREED_FILTERED,
    "MemoryMayNotBeFreed": ENHANCED_MEMORY_MAY_NOT_BE_FREED_FILTERED,
    "DoubleFree": ENHANCED_DOUBLE_FREE_FILTERED,
    "UseAfterFree": ENHANCED_USE_AFTER_FREE_FILTERED,
}


class CodeQLAnalyzer:
    """CodeQL analyzer with model injection and enhanced queries."""

    def __init__(
        self,
        binary: str = "codeql",
        timeout: int = 600,
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
        # Dynamic enhanced filtered query for DoubleFree; may be overridden
        # with LLM-generated filters at analysis time.
        self._double_free_filtered_query: str | None = None
        self._memory_never_freed_filtered_query: str | None = None
        self._memory_may_not_be_freed_filtered_query: str | None = None
        self._use_after_free_filtered_query: str | None = None

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
                self._double_free_filtered_query = self._build_double_free_filtered_with_custom_filters(
                    custom_queries
                )
                self._memory_never_freed_filtered_query = self._build_memory_never_freed_filtered_with_custom_filters(
                    custom_queries
                )
                self._memory_may_not_be_freed_filtered_query = self._build_memory_may_not_be_freed_filtered_with_custom_filters(
                        custom_queries
                )
                
                self._use_after_free_filtered_query = self._build_use_after_free_filtered_with_custom_filters(
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

    def _build_double_free_filtered_with_custom_filters(
        self,
        custom_queries: CustomQuerySet,
    ) -> str:
        """
        Build the final enhanced+filtered DoubleFree query text by combining:
          - the hard-coded ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE, and
          - all LLM-generated *double_free_filter* snippets from CustomQuerySet.

        This matches the JSON structure:

          "queries": {
            "<func>": {
              "double_free_filter": {
              },
              "use_after_free_filter": { ... },
              "memory_never_freed_filter": { ... },
              "memory_may_not_be_freed_filter": { ... },
              ...
            }
          }
        """
        predicates_snippets: list[str] = []
        use_exprs: list[str] = []
        required_imports: set[str] = set()

        cq_json = custom_queries.to_json()
        for func_name, qinfo in (cq_json.get("queries") or {}).items():
            if not isinstance(qinfo, dict):
                continue
            df_block = qinfo.get("double_free_filter") or {}
            if not isinstance(df_block, dict):
                continue

            predicates_code = (df_block.get("predicates_code", "") or "").strip()
            use_expr = (df_block.get("use_expr", "") or "").strip()
            
            # Collect required imports (use set to avoid duplicates)
            imports = df_block.get("required_imports", [])
            if isinstance(imports, list):
                for imp in imports:
                    if isinstance(imp, str) and imp.strip():
                        required_imports.add(imp.strip())
            
            # Skip if validation failed or block is empty
            validated = df_block.get("validated")
            if validated is False:
                continue  # Skip invalid blocks
            if not predicates_code and not use_expr:
                continue  # Skip empty blocks

            if predicates_code:
                predicates_snippets.append(
                    f"// ---- LLM double-free predicates for {func_name} ----\n{predicates_code}\n"
                )
            if use_expr:
                use_exprs.append(use_expr)

        # Only build query if we have validated filters
        if use_exprs or predicates_snippets:
            preds_part = "\n\n".join(predicates_snippets) if predicates_snippets else ""
            # Build dfFiltered aggregator from all use_exprs
            body = " or\n  ".join(use_exprs) if use_exprs else "false"
            df_agg = (
                "predicate dfFiltered(DeallocationExpr srcDealloc, "
                "DataFlow::Node sinkNode, Expr sinkFreedExpr) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = ENHANCED_DOUBLE_FREE_FILTERED_TEMPLATE
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + df_agg
            return merged

        # No validated filters: return None to skip creating the filtered query file
        return None

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

    def _build_use_after_free_filtered_with_custom_filters(
        self,
        custom_queries: CustomQuerySet,
    ) -> str:
        """
        Build the final enhanced+filtered UseAfterFree query text by combining:
          - ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE, and
          - all LLM-generated *use_after_free_filter* snippets from CustomQuerySet.
        """
        predicates_snippets: list[str] = []
        use_exprs: list[str] = []
        required_imports: set[str] = set()

        cq_json = custom_queries.to_json()
        for func_name, qinfo in (cq_json.get("queries") or {}).items():
            if not isinstance(qinfo, dict):
                continue
            uaf_block = qinfo.get("use_after_free_filter") or {}
            if not isinstance(uaf_block, dict):
                continue

            predicates_code = (uaf_block.get("predicates_code", "") or "").strip()
            use_expr = (uaf_block.get("use_expr", "") or "").strip()
            
            # Collect required imports (use set to avoid duplicates)
            imports = uaf_block.get("required_imports", [])
            if isinstance(imports, list):
                for imp in imports:
                    if isinstance(imp, str) and imp.strip():
                        required_imports.add(imp.strip())
            
            # Skip if validation failed or block is empty
            validated = uaf_block.get("validated")
            if validated is False:
                continue  # Skip invalid blocks
            if not predicates_code and not use_expr:
                continue  # Skip empty blocks

            if predicates_code:
                predicates_snippets.append(
                    f"// ---- LLM use-after-free predicates for {func_name} ----\n{predicates_code}\n"
                )
            if use_expr:
                use_exprs.append(use_expr)

        # Only build query if we have validated filters
        if use_exprs or predicates_snippets:
            preds_part = "\n\n".join(predicates_snippets) if predicates_snippets else ""
            body = " or\n  ".join(use_exprs) if use_exprs else "false"
            uaf_agg = (
                "predicate uafFiltered(DeallocationExpr dealloc, DataFlow::Node sinkNode) {\n"
                f"  {body}\n"
                "}\n"
            )
            merged = ENHANCED_USE_AFTER_FREE_FILTERED_TEMPLATE
            # Insert additional imports if any
            merged = self._insert_imports_into_template(merged, required_imports)
            if preds_part:
                merged += "\n\n" + preds_part
            merged += "\n\n" + uaf_agg
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
              if query_name == "DoubleFree":
                  qtext = self._double_free_filtered_query
                  label = "DoubleFree"
              elif query_name == "MemoryNeverFreed":
                  qtext = self._memory_never_freed_filtered_query
                  label = "MemoryNeverFreed"
              elif query_name == "MemoryMayNotBeFreed":
                  qtext = self._memory_may_not_be_freed_filtered_query
                  label = "MemoryMayNotBeFreed"
              elif query_name == "UseAfterFree":
                  qtext = self._use_after_free_filtered_query
                  label = "UseAfterFree"
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

        compile_commands = project_path / "compile_commands.json"
        if compile_commands.exists():
            logger.info(f"Using compile_commands.json for database creation: {compile_commands}")
            cmd = [
                self.binary, "database", "create", str(db_path),
                f"--source-root={project_path}",
                "--language=cpp", "--overwrite",
                f"--compilation-database={compile_commands}"
            ]
        else:
            build_cmd = None
            if config and "build_command" in config:
                bc = config["build_command"]
                build_cmd = bc.get("command") if isinstance(bc, dict) else bc
            if not build_cmd:
                build_cmd = self._detect_build_command(project_path)
            if not build_cmd:
                logger.error("Could not determine build command")
                return False

            logger.info(f"Using build command for CodeQL database creation: {build_cmd}")
            cmd = [
                self.binary, "database", "create", str(db_path),
                f"--source-root={project_path}",
                "--language=cpp", "--overwrite",
                "--command", build_cmd
            ]

        logger.info(f"Running CodeQL database create: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True, cwd=str(project_path))
        if result.returncode != 0:
            logger.error(f"Database creation failed: {result.stderr.decode(errors='replace')[:500]}")
            return False

        return self._finalize_database(db_path)

    def _load_build_config(self, project_path: Path) -> Optional[dict]:
        config_file = Path(__file__).parent.parent.parent / "proj_build_command.json"
        if not config_file.exists():
            return None
        try:
            with open(config_file) as f:
                return json.load(f).get(project_path.name)
        except Exception:
            return None

    def _run_commands(self, cwd: Path, commands) -> None:
        if isinstance(commands, str):
            commands = [{"command": c.strip(), "can_error": True} for c in commands.split("&&")]
        elif isinstance(commands, list):
            commands = [
                {"command": c.get("command", c) if isinstance(c, dict) else c,
                 "can_error": c.get("can_error", True) if isinstance(c, dict) else True}
                for c in commands
            ]

        for cmd_info in commands:
            if cmd_info["command"]:
                subprocess.run(cmd_info["command"], shell=True, cwd=str(cwd),
                               timeout=self.timeout, capture_output=True)

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

        Args:
            funcs: List of (function_name, arg_index) where:
                - arg_index = -1 means return value ("ReturnValue")
                - arg_index >= 0 means output parameter at that index
        """
        lines = [
            "extensions:",
            "  - addsTo:",
            "      pack: codeql/cpp-all",
            "      extensible: allocationFunctionModel",
            "    data:"
        ]
        for name, idx in funcs:
            # CodeQL uses "ReturnValue" for return, or index string for out-parameter
            output = "ReturnValue" if idx == -1 else str(idx)
            lines.append(f'      - ["", "", False, "{name}", "{output}", "", "", True]')
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
    
    # Use After Free (Memory error category)
    "USE_AFTER_FREE": MemoryIssueType.USE_AFTER_FREE,
    "USE_AFTER_FREE_LATENT": MemoryIssueType.USE_AFTER_FREE,
    "USE_AFTER_DELETE": MemoryIssueType.USE_AFTER_FREE,
    "USE_AFTER_DELETE_LATENT": MemoryIssueType.USE_AFTER_FREE,
    "USE_AFTER_LIFETIME": MemoryIssueType.USE_AFTER_FREE,
    "USE_AFTER_LIFETIME_LATENT": MemoryIssueType.USE_AFTER_FREE,
    "PULSE_REFERENCE_STABILITY": MemoryIssueType.USE_AFTER_FREE,
    "VECTOR_INVALIDATION": MemoryIssueType.USE_AFTER_FREE,
    "VECTOR_INVALIDATION_LATENT": MemoryIssueType.USE_AFTER_FREE,
    
    # Note: Infer does not have a DOUBLE_FREE issue type
    # Double-free bugs may be detected indirectly through other issue types
}


class InferAnalyzer:
    """Infer analyzer with model injection and hint support.
    
    Infer (Facebook Infer / Microsoft Infer) is a static analysis tool
    that detects memory safety issues in C/C++ code.
    """

    def __init__(
        self,
        binary: str = "infer",
        timeout: int = 600,
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
        self._injected_files: list[Path] = []
        self._irrelevant_warnings: list[Warning] = []  # Warnings not related to memory safety

    def analyze(
        self,
        project_path: Path,
        hints: HintSet | None = None,
        issue_types: list[MemoryIssueType] | None = None,
        use_enhanced_queries: bool = True,  # Kept for compatibility, not used by Infer
        custom_queries: CustomQuerySet | None = None,  # Kept for compatibility, not used by Infer
    ) -> list[Warning]:
        """
        Run Infer analysis.

        Args:
            project_path: Project to analyze
            hints: Allocator/deallocator hints
            issue_types: Which issue types to map/return
            use_enhanced_queries: Not used by Infer (kept for compatibility)
            custom_queries: Not used by Infer (kept for compatibility)
        """
        output_dir = Path(tempfile.mkdtemp())
        results_path = output_dir / "infer_results.json"

        try:
            # Step 1: Inject models from hints
            if hints and getattr(hints, "hints", None):
                logger.info("Step 1: Injecting custom memory models for Infer...")
                self._inject_models(hints, project_path)

            # Step 2: Run Infer analysis
            logger.info("Step 2: Running Infer analysis...")
            success = self._run_infer(project_path, results_path, hints)

            if not success:
                return []

            # Step 3: Parse Infer results
            warnings = self._parse_infer_results(results_path, project_path, issue_types)
            # _parse_infer_results already stores irrelevant warnings in self._irrelevant_warnings
            return warnings

        except Exception as e:
            logger.error(f"Infer analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            self._cleanup_models()

    def _inject_models(self, hints: HintSet, project_path: Path) -> None:
        """Inject custom memory models into Infer's models directory.
        
        Infer uses .models files in a models subdirectory or .infer directory.
        We create models for allocators and deallocators.
        """
        # Infer looks for models in models/ subdirectory or .infer/models/
        models_dir = project_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        allocators = hints.get_allocators() or []
        deallocators = hints.get_deallocators() or []
        
        # Normalize allocators: convert to list of (name, idx) tuples
        alloc_list: list[tuple[str, int]] = []
        for a in allocators:
            if isinstance(a, (list, tuple)) and len(a) == 2:
                alloc_list.append((str(a[0]), int(a[1])))
            else:
                alloc_list.append((str(a), -1))  # -1 means return value
        
        # Normalize deallocators: convert to list of (name, idx) tuples
        dealloc_list: list[tuple[str, int]] = []
        for d in deallocators:
            if isinstance(d, (list, tuple)) and len(d) == 2:
                dealloc_list.append((str(d[0]), int(d[1])))
            else:
                # Default: assume first argument is freed
                dealloc_list.append((str(d), 0) if isinstance(d, str) else (str(d[0]), 0))
        
        # Write allocator models (Infer format: function_name followed by properties)
        if alloc_list:
            alloc_file = models_dir / "hint_allocators.models"
            with open(alloc_file, "w") as f:
                for name, idx in alloc_list:
                    if idx == -1:
                        # Return value allocator
                        f.write(f"{name}\n")
                        f.write("  alloc: true\n")
                        f.write("  alloc_return: true\n")
                    else:
                        # Output parameter allocator
                        f.write(f"{name}\n")
                        f.write("  alloc: true\n")
                        f.write(f"  alloc_index: {idx}\n")
            self._injected_files.append(alloc_file)
            logger.info(f"Injected {len(alloc_list)} allocator models to {alloc_file}")
        
        # Write deallocator models
        if dealloc_list:
            dealloc_file = models_dir / "hint_deallocators.models"
            with open(dealloc_file, "w") as f:
                for name, idx in dealloc_list:
                    f.write(f"{name}\n")
                    f.write("  free: true\n")
                    f.write(f"  free_index: {idx}\n")
            self._injected_files.append(dealloc_file)
            logger.info(f"Injected {len(dealloc_list)} deallocator models to {dealloc_file}")

    def _load_build_config(self, project_path: Path) -> Optional[dict]:
        """Load Infer-specific build configuration from proj_build_command_infer.json."""
        config_file = Path(__file__).parent.parent.parent / "proj_build_command_infer.json"
        if not config_file.exists():
            return None
        try:
            with open(config_file) as f:
                return json.load(f).get(project_path.name)
        except Exception:
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

        for cmd_info in commands:
            if cmd_info["command"]:
                subprocess.run(cmd_info["command"], shell=True, cwd=str(cwd),
                               timeout=self.timeout, capture_output=True)

    def _run_infer(self, project_path: Path, results_path: Path, hints: HintSet | None = None) -> bool:
        """Run Infer analysis on the project.
        
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
        
        # Load Infer-specific build configuration
        config = self._load_build_config(project_path)
        
        # Run prepare_for_build commands if configured
        if config and "prepare_for_build" in config:
            prepare_commands = config["prepare_for_build"]
            if prepare_commands:
                logger.info("Running prepare_for_build commands for Infer...")
                self._run_commands(project_path, prepare_commands)
        
        # Build pulse-model patterns from hints for custom allocators/deallocators
        # These flags tell Infer to recognize custom allocation/deallocation functions
        pulse_flags = []
        if hints:
            allocators = hints.get_allocators() or []
            deallocators = hints.get_deallocators() or []
            
            # Extract function names (ignore arg_index for patterns)
            alloc_names = [name for name, _ in allocators if name not in ("main", "_main", "")]
            dealloc_names = [name for name, _ in deallocators if name not in ("main", "_main", "")]
            
            # Build regex patterns: ^function_name$ for exact match
            # Escape special regex characters in function names (but not underscores, letters, digits)
            import re
            def escape_regex_special(name: str) -> str:
                """Escape only special regex characters that could cause issues."""
                # Characters that need escaping in regex: . ^ $ * + ? { } [ ] \ | ( )
                # We'll escape them, but keep normal function name characters as-is
                special_chars = r'.^$*+?{}[]\|()'
                escaped = ""
                for char in name:
                    if char in special_chars:
                        escaped += "\\" + char
                    else:
                        escaped += char
                return escaped
            
            # Combine multiple functions with | inside parentheses
            # Format: --pulse-model-malloc-pattern=^(func1|func2|func3)$
            # Note: No quotes needed - subprocess.run() with list handles arguments correctly
            if alloc_names:
                escaped_names = [escape_regex_special(name) for name in alloc_names]
                alloc_pattern = f"^({'|'.join(escaped_names)})$"
                pulse_flags.append(f"--pulse-model-malloc-pattern={alloc_pattern}")
                logger.info(f"Adding pulse-model-malloc-pattern for: {', '.join(alloc_names)}")
            
            if dealloc_names:
                escaped_names = [escape_regex_special(name) for name in dealloc_names]
                dealloc_pattern = f"^({'|'.join(escaped_names)})$"
                pulse_flags.append(f"--pulse-model-free-pattern={dealloc_pattern}")
                logger.info(f"Adding pulse-model-free-pattern for: {', '.join(dealloc_names)}")
        
        # Use infer run -- [command] which combines capture and analyze
        # This is the recommended approach per Infer documentation
        # See: https://fbinfer.com/docs/hello-world
        compile_commands = project_path / "compile_commands.json"
        if compile_commands.exists():
            logger.info("Using compile_commands.json for Infer")
            # For compile_commands.json, use capture with --compilation-database
            # then run analyze separately (infer run doesn't support --compilation-database directly)
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
                # Check if infer-out directory was created and has files
                if infer_out_dir.exists():
                    files_count = len(list(infer_out_dir.rglob("*")))
                    logger.debug(f"Infer-out directory contains {files_count} files/directories")
                else:
                    logger.warning(f"Infer-out directory not found at {infer_out_dir} after capture")
            
            # Run analyze separately with pulse-model flags
            analyze_cmd = [self.binary, "analyze"] + pulse_flags
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
            # Get build command from config or detect it
            build_cmd = None
            infer_flags = []
            
            if config and "build_command" in config:
                bc = config["build_command"]
                if isinstance(bc, dict):
                    build_cmd = bc.get("command")
                    # Support infer_flags in build_command config
                    infer_flags = bc.get("infer_flags", [])
                    if isinstance(infer_flags, str):
                        infer_flags = [infer_flags]
                else:
                    build_cmd = bc
            
            if not build_cmd:
                build_cmd = self._detect_build_command(project_path)
            
            if not build_cmd:
                logger.error("Could not determine build command for Infer")
                return False
            
            # Use infer run -- [command] which is equivalent to capture + analyze
            # This is the recommended approach per Infer documentation
            # Support Infer-specific flags like --keep-going, --pulse-only, etc.
            # Pulse-model flags are added from hints (already built above)
            logger.info(f"Running Infer with build command: {build_cmd}")
            if infer_flags:
                logger.info(f"Using Infer flags: {infer_flags}")
            if pulse_flags:
                logger.info(f"Using pulse-model flags from hints")
            
            # Build the command - split build_cmd to match how Infer expects it
            # Infer's -- separator expects the build command as separate arguments
            build_cmd_parts = build_cmd.split() if isinstance(build_cmd, str) else build_cmd
            run_cmd_parts = [
                self.binary, "run",
            ] + infer_flags + pulse_flags + [
                "--"
            ] + build_cmd_parts
            
            # Log the command as it would appear in terminal
            logger.info(f"Running Infer command: {' '.join(run_cmd_parts)}")
            logger.info(f"Working directory: {project_path}")
            logger.debug(f"Full command list: {run_cmd_parts}")
            
            # Execute Infer with environment variables passed through
            # This matches how the command is run manually in terminal
            result = subprocess.run(
                run_cmd_parts,
                cwd=str(project_path),
                timeout=self.timeout,
                capture_output=True,
                text=True,
                shell=False,  # Use list format, not shell
                env=os.environ.copy(),  # Pass through environment variables (PATH, etc.)
            )
            
            # Infer writes compilation output to stderr, which is normal
            # Check if infer-out was created to determine success
            if not infer_out_dir.exists():
                logger.error("Infer-out directory not created - analysis may have failed")
                if result.returncode != 0:
                    logger.error(f"Infer run failed with return code {result.returncode}")
                    logger.debug(f"Stderr: {result.stderr[:1000] if result.stderr else 'No stderr'}")
                return False
            else:
                logger.info("Infer run completed - infer-out directory created")
                # Check if infer-out directory has files
                files_count = len(list(infer_out_dir.rglob("*")))
                logger.debug(f"Infer-out directory contains {files_count} files/directories")
        
        # Check for report.json in the project's infer-out directory
        # This is where Infer writes the report by default
        report_json = infer_out_dir / "report.json"
        if report_json.exists():
            logger.info(f"Found report.json at {report_json}")
            shutil.copy(report_json, results_path)
            # Validate JSON format
            try:
                json_text = results_path.read_text()
                json_data = json.loads(json_text)
                issue_count = len(json_data) if isinstance(json_data, list) else len(json_data.get("bugs", [])) if isinstance(json_data, dict) else 0
                logger.info(f"Infer analysis found {issue_count} issues")
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in report.json: {e}")
                return False
        
        # If report.json doesn't exist, try to generate it using infer report --json
        logger.info(f"report.json not found in {infer_out_dir}, generating JSON report...")
        report_cmd = [
            self.binary, "report",
            "--json",
        ]
        
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
        
        # Write stdout to results file
        if result.stdout:
            try:
                # Validate JSON format
                json_data = json.loads(result.stdout)
                results_path.write_text(result.stdout)
                issue_count = len(json_data) if isinstance(json_data, list) else len(json_data.get("bugs", [])) if isinstance(json_data, dict) else 0
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
        # Check for Makefile
        if (project_path / "Makefile").exists() and which("make"):
            return "make"
        
        # Check for CMake
        if (project_path / "CMakeLists.txt").exists() and which("cmake"):
            build_dir = project_path / "build"
            build_dir.mkdir(exist_ok=True)
            subprocess.run(["cmake", ".."], cwd=str(build_dir), capture_output=True, timeout=120)
            if (build_dir / "Makefile").exists():
                return "make -C build"
        
        # Fallback: try to compile some source files
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
            
            # Infer JSON format: array of bug reports
            # Handle both array format and object with "bugs" key
            if isinstance(data, dict):
                # Some Infer versions output {"bugs": [...]}
                bugs = data.get("bugs", [])
                if not bugs and "bug_type" in data:
                    # Single bug object
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
            irrelevant_warnings_list = []
            
            for bug in bugs:
                bug_type = bug.get("bug_type", "").upper()
                
                # Map Infer bug types to our issue types
                issue_type = INFER_ISSUE_MAP.get(bug_type)
                if not issue_type:
                    # Try substring matching
                    for key, val in INFER_ISSUE_MAP.items():
                        if key in bug_type:
                            issue_type = val
                            break
                
                # Check if this is a memory safety related issue
                is_memory_safety_related = issue_type is not None
                
                # If not memory safety related, we'll store it separately
                if not is_memory_safety_related:
                    unmapped_types.add(bug_type)
                    # Use OTHER for non-memory-safety issues
                    issue_type = MemoryIssueType.OTHER
                    logger.debug(f"Non-memory-safety bug type '{bug_type}' will be saved to separate file")
                else:
                    # Filter by issue_types if specified (only for memory safety related)
                    if issue_types and issue_type not in issue_types:
                        filtered_count += 1
                        logger.debug(f"Filtered out bug type '{bug_type}' (mapped to {issue_type.name}) - not in requested types")
                        continue
                
                # Extract location information directly from bug object
                file_path = bug.get("file", "")
                line = bug.get("line", 0)
                function_name = bug.get("procedure", "")
                
                # Extract message
                message = bug.get("qualifier", "") or bug.get("description", "") or bug.get("bug_type_hum", "") or bug_type
                
                # Build trace from bug_trace array
                bug_trace = bug.get("bug_trace", [])
                trace = []
                allocation_site = ""
                
                if bug_trace:
                    # Primary location is the last entry in trace (where the bug occurs)
                    primary_loc = bug_trace[-1] if bug_trace else {}
                    if not file_path:
                        file_path = primary_loc.get("filename", "")
                    if not line:
                        line = primary_loc.get("line_number", 0)
                    
                    # Build trace from all trace elements
                    for trace_elem in bug_trace:
                        trace_file = trace_elem.get("filename", "")
                        trace_line = trace_elem.get("line_number", 0)
                        if trace_file and trace_line:
                            trace.append(f"{trace_file}:{trace_line}")
                    
                    # Allocation site is typically the first entry in trace
                    if bug_trace:
                        first_loc = bug_trace[0]
                        alloc_file = first_loc.get("filename", "")
                        alloc_line = first_loc.get("line_number", 0)
                        if alloc_file and alloc_line:
                            allocation_site = f"{alloc_file}:{alloc_line}"
                
                # If no trace, use direct file/line from bug object
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
                
                # Store in appropriate list
                if is_memory_safety_related:
                    warnings.append(warning)
                else:
                    irrelevant_warnings_list.append(warning)
            
            # Store irrelevant warnings in instance variable
            self._irrelevant_warnings = irrelevant_warnings_list
            
            # Log summary
            if unmapped_types:
                logger.info(f"Found {len(unmapped_types)} non-memory-safety bug type(s): {', '.join(sorted(unmapped_types))} (saved to separate file)")
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} bug(s) based on issue_types filter")
            if warnings:
                logger.info(f"Successfully parsed {len(warnings)} memory-safety warning(s) from Infer results")
            if irrelevant_warnings_list:
                logger.info(f"Found {len(irrelevant_warnings_list)} non-memory-safety warning(s) (will be saved separately)")
        
        except Exception as e:
            logger.error(f"Failed to parse Infer results: {e}")
            import traceback
            traceback.print_exc()
        
        return warnings

    def get_irrelevant_warnings(self) -> list[Warning]:
        """Get warnings that are not related to memory safety (leak, UAF, double-free).
        
        Returns:
            List of Warning objects for non-memory-safety issues
        """
        return self._irrelevant_warnings.copy()
    
    def _cleanup_models(self) -> None:
        """Clean up injected model files."""
        for f in self._injected_files:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass
        self._injected_files.clear()
