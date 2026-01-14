"""
Paper -- "Hint: Guiding Static Analysis with LLM-Assisted Memory-Semantics Annotation"

Uses CEGAR (Counterexample-Guided Abstraction Refinement) approach:
1. LLM generates memory safety annotations
2. Z3 constraint solver validates annotations
3. CodeQL performs static analysis with validated annotations

Supports detection of:
- Memory leaks
- Double-free
- Use-after-free
- Null pointer dereference
- Buffer overflow

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


ISSUE_TYPE_MAP = {
    "leak": MemoryIssueType.MEMORY_LEAK,
    "memory-leak": MemoryIssueType.MEMORY_LEAK,
    "double-free": MemoryIssueType.DOUBLE_FREE,
    "uaf": MemoryIssueType.USE_AFTER_FREE,
    "use-after-free": MemoryIssueType.USE_AFTER_FREE,
    "null": MemoryIssueType.NULL_DEREFERENCE,
    "null-deref": MemoryIssueType.NULL_DEREFERENCE,
    "overflow": MemoryIssueType.BUFFER_OVERFLOW,
    "buffer-overflow": MemoryIssueType.BUFFER_OVERFLOW,
}


def setup_logging(level: str = "INFO", log_file: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_issue_types(issue_strs: list[str]) -> list[MemoryIssueType]:
    """Parse issue type strings to enum values."""
    if not issue_strs:
        return None

    types = []
    for s in issue_strs:
        s_lower = s.lower()
        if s_lower in ISSUE_TYPE_MAP:
            types.append(ISSUE_TYPE_MAP[s_lower])
        else:
            print(f"Warning: Unknown issue type '{s}', skipping")
    return types if types else None


def main():
    parser = argparse.ArgumentParser(description="Hint: Guiding Static Analysis with LLM-Assisted Memory-Semantics Annotation")

    parser.add_argument("--project", "-p", required=True, help="Project path to analyze")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--analyzer", choices=["codeql", "infer"], default="codeql",
                        help="Static analyzer to use (default: codeql)")
    parser.add_argument("--issues", "-i", nargs="+", metavar="TYPE",
                        help="Issue types to detect (default: all)")
    parser.add_argument("--model", default="gemini-2.5-pro", help="LLM model (default: gemini-2.5-pro)")
    parser.add_argument("--api-key", help="API key for LLM access")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max CEGAR iterations (default: 5)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"hint_{timestamp}.log"

    setup_logging("INFO", str(log_file))
    logger = logging.getLogger(__name__)

    project_path = Path(args.project)
    if not project_path.exists():
        logger.error(f"Project not found: {project_path}")
        sys.exit(1)

    issue_types = parse_issue_types(args.issues)
    if issue_types:
        logger.info(f"Detecting: {[t.name for t in issue_types]}")
    else:
        logger.info("Detecting: all memory safety issues")

    pipeline = Pipeline(
        api_key=args.api_key,
        model=args.model,
        analyzer_type=args.analyzer,
        max_iterations=args.max_iterations,
        issue_types=issue_types,
    )

    result = pipeline.analyze(project_path, output_dir)

    print("\n" + "=" * 50)
    print(result.summary())
    print(f"Results saved to: {output_dir}")
    print(f"  - memory_safety_bugs.json")
    print(f"  - final_annotations.json")
    print(f"  - validation_conflicts.txt")
    print(f"  - report.md")


if __name__ == "__main__":
    main()