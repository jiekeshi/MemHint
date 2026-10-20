"""
MemHint: Finding Memory Leaks in C/C++ Programs via Neuro-Symbolic Augmented Static Analysis

Runs the full 3-stage pipeline:
  Stage 1 - Summary Generation (LLM + Z3 validation)
  Stage 2 - Summary-Augmented Analysis (CodeQL or Infer)
  Stage 3 - Warning Validation (Z3 filtering + LLM validation)

Usage:
    python main.py --project /path/to/code
    python main.py --project /path/to/code --analyzer infer
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.core.pipeline import Pipeline
from src.analyzer.adapters import AnalyzerError


def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="MemHint: Finding Memory Leaks in C/C++ Programs via Neuro-Symbolic Augmented Static Analysis"
    )
    parser.add_argument("--project", "-p", required=True, help="Project path to analyze")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--analyzer", choices=["codeql", "infer"], default="codeql", help="Static analyzer (default: codeql)")
    parser.add_argument("--source-root", type=Path, help="Root directory to scan for functions (defaults to --project)")
    parser.add_argument("--cpp-queries-dir", type=Path, help="Path to CodeQL cpp-queries directory")
    parser.add_argument("--start-from", type=int, choices=[1, 3, 4, 5, 6], default=1,
                        help="Start at this phase, reusing artifacts already in --output: "
                             "3 revalidates the phase 2 summaries in hints_unvalidated.json, "
                             "4 reuses hints.json, 5 also reuses <analyzer>_warnings.json, "
                             "6 also reuses memory_safety_bugs.json (default: 1, run everything)")
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="Phase 2 summary-generation model (default: the one the paper uses)")
    parser.add_argument("--llm-verify-model", default="gemini-3.1-pro-preview",
                        help="Phase 6 warning-validation model (default: the one the paper uses)")
    parser.add_argument("--reuse-db", action="store_true",
                        help="Reuse the analyzer database from a previous run. It does not depend "
                             "on the summaries, so rebuilding it to try a different set is waste")
    parser.add_argument("--no-llm-verify", action="store_true",
                        help="Stop after phase 5 instead of running the phase 6 LLM validation")

    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(str(output_dir / f"memhint_{timestamp}.log"))
    logger = logging.getLogger(__name__)

    # Validate paths
    project_path = Path(args.project)
    if not project_path.exists():
        logger.error(f"Project not found: {project_path}")
        sys.exit(1)

    source_root = None
    if args.source_root:
        source_root = Path(args.source_root)
        if not source_root.exists():
            logger.error(f"Source root not found: {source_root}")
            sys.exit(1)

    logger.info(f"Analyzing {project_path} with {args.analyzer}")

    # Run full pipeline
    pipeline = Pipeline(
        model=args.model,
        analyzer_type=args.analyzer,
        cpp_queries_dir=args.cpp_queries_dir,
        reuse_db=args.reuse_db,
        use_enhanced_queries=True,
        use_llm_verify_bugs=not args.no_llm_verify,
        llm_verify_model=args.llm_verify_model,
        pipeline_mode="full",
        start_from=args.start_from,
    )

    try:
        result = pipeline.analyze(
            project_path,
            output_dir,
            source_root=source_root,
        )
    except AnalyzerError as e:
        logger.error(f"{args.analyzer} did not complete: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print(result.summary())
    print(f"\nResults saved to: {output_dir}")
    print(f"  - hints.json              (validated function summaries)")
    print(f"  - memory_safety_bugs.json (detected memory leaks)")
    print(f"  - llm_verify_bugs.json    (LLM validation: TP vs FP)")
    print(f"  - report.md               (human-readable report)")
    print("=" * 60)


if __name__ == "__main__":
    main()
