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
        model="gemini-3-flash-preview",
        analyzer_type=args.analyzer,
        cpp_queries_dir=args.cpp_queries_dir,
        reuse_db=False,
        use_enhanced_queries=True,
        use_llm_verify_bugs=True,
        llm_verify_model="gemini-3.1-pro-preview",
        pipeline_mode="full",
    )

    result = pipeline.analyze(
        project_path,
        output_dir,
        source_root=source_root,
    )

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
