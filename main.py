"""
HINT: LLM-Assisted Memory-Semantics Annotation for Static Analysis

Pipeline flow:
1. Parse source code (tree-sitter)
2. LLM generates memory safety hints (ALLOCATOR, DEALLOCATOR)
3. Z3 validates hints consistency
4. CodeQL scans with hint-based model extensions
5. Z3 filters false positives by path feasibility

Supports detection of:
1. Memory leaks (allocation without free on some feasible path)
2. Double-free (two frees on same pointer on some feasible path)
3. Use-after-free (use of pointer after free on some feasible path)

Usage:
    python main.py --project /path/to/code
    python main.py --project /path/to/code --issues leak double-free
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.core.models import MemoryIssueType
from src.core.pipeline import Pipeline


BUG_TYPE_MAP = {
    "leak": MemoryIssueType.MEMORY_LEAK,
    "memory-leak": MemoryIssueType.MEMORY_LEAK,
    "double-free": MemoryIssueType.DOUBLE_FREE,
    "uaf": MemoryIssueType.USE_AFTER_FREE,
    "use-after-free": MemoryIssueType.USE_AFTER_FREE,
}


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_bug_types(type_strs: list[str]) -> list[MemoryIssueType]:
    """Parse bug type strings to enum values."""
    if not type_strs:
        return None

    types = []
    for s in type_strs:
        s_lower = s.lower()
        if s_lower in BUG_TYPE_MAP:
            types.append(BUG_TYPE_MAP[s_lower])
        else:
            print(f"Warning: Unknown bug type '{s}', skipping")
    return types if types else None


def main():
    parser = argparse.ArgumentParser(
        description="HINT: LLM-Assisted Memory-Semantics Annotation for Static Analysis"
    )

    parser.add_argument(
        "--project", "-p", required=True,
        help="Project path to analyze (used for building analyzer database)"
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--analyzer", choices=["codeql", "infer"], default="codeql",
        help="Static analyzer to use (default: codeql)"
    )
    parser.add_argument(
        "--issues", "-i", nargs="+", metavar="TYPE",
        help="Bug types to detect: leak, double-free, uaf (default: all)"
    )
    parser.add_argument(
        "--model", default="gemini-2.5-pro",
        help="LLM model (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--single-source",
        nargs="+",
        help="Optional: path(s) to one or more C/C++ source files to feed to the LLM/hint phases "
             "instead of parsing the entire project (for small tests). Can specify multiple files.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Optional: root directory to scan for functions/macros for LLM hints. "
            "Defaults to the same as --project. Useful when the build root is larger "
            "than the code region you want to feed to the LLM."
        ),
    )
    parser.add_argument(
        "--codeql-dir",
        type=Path,
        help="Optional: custom CodeQL directory path (default: ~/.codeql)"
    )
    parser.add_argument(
        "--cpp-queries-dir",
        type=Path,
        help="Optional: direct path to cpp-queries directory (e.g., /path/to/codeql/qlpacks/codeql/cpp-queries)"
    )
    parser.add_argument(
        "--no-reuse-db",
        action="store_true",
        help="Force recreation of CodeQL database (default: reuse existing database if available)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--skip-hints",
        action="store_true",
        help="Skip LLM hint generation/validation and run CodeQL without hint injection"
    )
    parser.add_argument(
        "--no-enhanced-queries",
        action="store_true",
        help="Disable enhanced (hardcoded) CodeQL queries and use standard CodeQL queries instead",
    )

    args = parser.parse_args()

    # Setup output and logging
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"hint_{timestamp}.log"

    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level, str(log_file))
    logger = logging.getLogger(__name__)

    # Validate project / source paths
    project_path = Path(args.project)
    if not project_path.exists():
        logger.error(f"Project not found: {project_path}")
        sys.exit(1)

    source_root: Path | None = None
    if args.source_root:
        source_root = Path(args.source_root)
        if not source_root.exists():
            logger.error(f"Source root not found: {source_root}")
            sys.exit(1)

    single_sources = None
    if args.single_source:
        single_sources = [Path(p) for p in args.single_source]
        for single_source in single_sources:
            if not single_source.exists():
                logger.error(f"Single source file not found: {single_source}")
                sys.exit(1)

    # Parse bug types
    bug_types = parse_bug_types(args.issues)
    if bug_types:
        logger.info(f"Detecting: {[t.name for t in bug_types]}")
    else:
        logger.info("Detecting: all memory safety bugs")

    # Run pipeline
    pipeline = Pipeline(
        model=args.model,
        analyzer_type=args.analyzer,
        issue_types=bug_types,
        codeql_dir=args.codeql_dir,
        cpp_queries_dir=args.cpp_queries_dir,
        reuse_db=not args.no_reuse_db,
        skip_hint_injection=args.skip_hints,
        use_enhanced_queries=not args.no_enhanced_queries,
    )

    result = pipeline.analyze(
        project_path,
        output_dir,
        single_sources=single_sources,
        source_root=source_root,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(result.summary())
    print(f"\nResults saved to: {output_dir}")
    print(f"  - hints.json              (LLM-generated hints)")
    print(f"  - memory_safety_bugs.json (confirmed bugs)")
    print(f"  - filtered_warnings.json  (Z3-filtered false positives)")
    print(f"  - report.md               (human-readable report)")
    print("=" * 60)


if __name__ == "__main__":
    main()