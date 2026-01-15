"""Static analyzer adapters for CodeQL.
"""

import json
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from shutil import which

from src.core.models import Warning, HintSet, MemoryIssueType


logger = logging.getLogger(__name__)

CODEQL_ISSUE_MAP = {
    # Memory leak
    "memory-never-freed": MemoryIssueType.MEMORY_LEAK,
    "memory-may-not-be-freed": MemoryIssueType.MEMORY_LEAK,
    "cpp/memory-never-freed": MemoryIssueType.MEMORY_LEAK,
    "cpp/memory-may-not-be-freed": MemoryIssueType.MEMORY_LEAK,
    # Double free
    "double-free": MemoryIssueType.DOUBLE_FREE,
    "cpp/double-free": MemoryIssueType.DOUBLE_FREE,
    # Use after free
    "use-after-free": MemoryIssueType.USE_AFTER_FREE,
    "cpp/use-after-free": MemoryIssueType.USE_AFTER_FREE,
    # Allocation/deallocation mismatch
    "new-array-delete-mismatch": MemoryIssueType.ALLOC_DEALLOC_MISMATCH,
    "cpp/new-array-delete-mismatch": MemoryIssueType.ALLOC_DEALLOC_MISMATCH,
    "new-free-mismatch": MemoryIssueType.ALLOC_DEALLOC_MISMATCH,
    "cpp/new-free-mismatch": MemoryIssueType.ALLOC_DEALLOC_MISMATCH,
    "new-delete-array-mismatch": MemoryIssueType.ALLOC_DEALLOC_MISMATCH,
    "cpp/new-delete-array-mismatch": MemoryIssueType.ALLOC_DEALLOC_MISMATCH,
}


class CodeQLAnalyzer:
    """CodeQL analyzer with C/C++ model extension support."""

    def __init__(self, binary: str = "codeql", timeout: int = 600):
        self.binary = binary
        self.timeout = timeout

    def analyze(
        self, project_path: Path, hints: HintSet = None,
        issue_types: list[MemoryIssueType] = None
    ) -> list[Warning]:
        """Run CodeQL analysis with model extensions."""
        output_dir = Path(tempfile.mkdtemp())
        db_path = output_dir / "codeql-db"
        results_path = output_dir / "results.sarif"
        model_pack_dir = output_dir / "model-pack"

        try:
            # Step 1: Create database
            logger.info("Step 1: Creating CodeQL database...")
            if not self._create_database(project_path, db_path):
                logger.error("Failed to create database")
                return []

            # Step 2: Generate library pack with custom models
            has_custom_models = False
            if hints and len(hints.hints) > 0:
                logger.info("Step 2: Generating model pack...")
                has_custom_models = self._setup_model_pack(model_pack_dir, hints)

            # Step 3: Run queries
            logger.info("Step 3: Running queries with custom models...")
            success = self._run_memory_queries_with_models(db_path, model_pack_dir, results_path)

            if success:
                return self._parse_sarif(results_path)
            return []

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    def _create_database(self, project_path: Path, db_path: Path) -> bool:
        """Create CodeQL database."""
        logger.info(f"Project: {project_path}")

        compile_commands = project_path / "compile_commands.json"
        if compile_commands.exists():
            logger.info("Using compile_commands.json")
            cmd = [
                self.binary, "database", "create",
                str(db_path),
                f"--source-root={project_path}",
                "--language=cpp",
                "--overwrite",
                f"--compilation-database={compile_commands}",
            ]
        else:
            build_cmd = self._get_build_command(project_path)
            if not build_cmd:
                logger.error("Could not determine build command")
                return False

            logger.info(f"Build command: {build_cmd}")
            cmd = [
                self.binary, "database", "create",
                str(db_path),
                f"--source-root={project_path}",
                "--language=cpp",
                "--overwrite",
                f"--command={build_cmd}",
            ]

        result = subprocess.run(
            cmd, timeout=self.timeout, capture_output=True, cwd=str(project_path)
        )

        if result.returncode != 0:
            logger.error(f"Database creation failed:\n{result.stderr.decode()[:1000]}")
            return False

        logger.info("Database created successfully")
        return True

    def _get_build_command(self, project_path: Path) -> str:
        """Determine appropriate build command."""
        if (project_path / "Makefile").exists() and which("make"):
            return "make"

        lint_dir = project_path / "lint"
        if lint_dir.exists() and (lint_dir / "CMakeLists.txt").exists() and which("cmake"):
            build_dir = lint_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            if not (build_dir / "Makefile").exists():
                subprocess.run(["cmake", ".."], cwd=str(build_dir), capture_output=True, timeout=120)
            if (build_dir / "Makefile").exists():
                return "make -C lint/build"

        if (project_path / "CMakeLists.txt").exists() and which("cmake"):
            build_dir = project_path / "build"
            build_dir.mkdir(exist_ok=True)
            if not (build_dir / "Makefile").exists():
                subprocess.run(["cmake", ".."], cwd=str(build_dir), capture_output=True, timeout=120)
            if (build_dir / "Makefile").exists():
                return "make -C build"

        c_files = list(project_path.rglob("*.c"))
        cpp_files = list(project_path.rglob("*.cpp"))
        if c_files or cpp_files:
            files = [str(f.relative_to(project_path)) for f in (c_files + cpp_files)[:30]]
            compiler = "clang++" if cpp_files else "clang"
            return f"{compiler} -I. -c -fsyntax-only {' '.join(files)}"

        return None

    def _setup_model_pack(self, model_pack_dir: Path, hints: HintSet) -> bool:
        """Create a CodeQL library pack with custom allocation/deallocation models.

        Returns True if models were created successfully.
        """
        model_pack_dir.mkdir(parents=True, exist_ok=True)

        # Collect function hints
        alloc_funcs = [f for f in hints.get_allocators() if f not in ("main", "_main")]
        free_funcs = hints.get_deallocators()  # List of (func_name, arg_index)

        logger.info(f"Allocators ({len(alloc_funcs)}): {alloc_funcs[:10]}...")
        logger.info(f"Deallocators ({len(free_funcs)}): {free_funcs[:10]}...")

        if not alloc_funcs and not free_funcs:
            logger.warning("No custom allocators or deallocators found")
            return False

        # Create qlpack.yml - LIBRARY pack that extends codeql/cpp-all
        qlpack_content = """name: hint/memory-models
version: 0.0.1
library: true
extensionTargets:
  codeql/cpp-all: "*"
"""
        (model_pack_dir / "qlpack.yml").write_text(qlpack_content)

        # Generate the .qll file
        self._generate_model_library(model_pack_dir, alloc_funcs, free_funcs)

        # Install dependencies
        logger.info("Installing model pack dependencies...")
        result = subprocess.run(
            [self.binary, "pack", "install", str(model_pack_dir)],
            capture_output=True, timeout=300
        )

        stdout = result.stdout.decode()
        stderr = result.stderr.decode()

        if result.returncode != 0:
            logger.error(f"Pack install failed!")
            logger.error(f"stdout: {stdout}")
            logger.error(f"stderr: {stderr}")
            return False

        logger.info(f"Pack install successful")
        logger.debug(f"stdout: {stdout}")

        return True

    def _generate_model_library(
        self, model_pack_dir: Path,
        alloc_funcs: list[str],
        free_funcs: list[tuple[str, int]],
    ) -> None:
        """Generate CodeQL library extending AllocationFunction and DeallocationFunction.

        Args:
            model_pack_dir: Directory to write the .qll file
            alloc_funcs: List of allocator function names
            free_funcs: List of (func_name, freed_arg_index) tuples
        """
        # Build allocator names for CodeQL IN clause
        if alloc_funcs:
            alloc_names = ", ".join(f'"{f}"' for f in alloc_funcs)
        else:
            alloc_names = '"__no_custom_allocator__"'

        # Group deallocators by freed arg index
        dealloc_by_arg: dict[int, list[str]] = {}
        for func_name, arg_idx in free_funcs:
            if arg_idx not in dealloc_by_arg:
                dealloc_by_arg[arg_idx] = []
            dealloc_by_arg[arg_idx].append(func_name)

        # Generate deallocation classes - one per arg index
        dealloc_classes = []
        for arg_idx, funcs in sorted(dealloc_by_arg.items()):
            func_names = ", ".join(f'"{f}"' for f in funcs)
            class_name = f"HintDeallocationFunctionArg{arg_idx}"
            dealloc_classes.append(f'''
/**
 * Custom deallocation functions (freed arg at index {arg_idx}).
 */
class {class_name} extends DeallocationFunction {{
  {class_name}() {{
    this.getName() in [{func_names}]
  }}

  override int getFreedArg() {{ result = {arg_idx} }}
}}''')

        if dealloc_classes:
            dealloc_section = "\n".join(dealloc_classes)
        else:
            dealloc_section = '''
/**
 * No custom deallocation functions.
 */
class HintDeallocationFunction extends DeallocationFunction {
  HintDeallocationFunction() {
    this.getName() in ["__no_custom_deallocator__"]
  }

  override int getFreedArg() { result = 0 }
}'''

        # Generate .qll content
        qll_content = f'''/**
 * HINT Custom Memory Models
 *
 * Extends CodeQL's built-in AllocationFunction and DeallocationFunction
 * with custom functions identified by LLM annotations.
 */

import cpp
private import semmle.code.cpp.models.interfaces.Allocation
private import semmle.code.cpp.models.interfaces.Deallocation

/**
 * Custom allocation functions from LLM annotations.
 */
class HintAllocationFunction extends AllocationFunction {{
  HintAllocationFunction() {{
    this.getName() in [{alloc_names}]
  }}

  override predicate requiresDealloc() {{ any() }}
}}
{dealloc_section}
'''
        qll_path = model_pack_dir / "HintMemoryModels.qll"
        qll_path.write_text(qll_content)
        logger.info(f"Generated: {qll_path}")
        logger.info(f"  - Allocators: {len(alloc_funcs)}")
        logger.info(f"  - Deallocators: {len(free_funcs)} ({len(dealloc_by_arg)} unique arg indices)")

        # Log the content for debugging
        logger.debug(f"QLL content:\n{qll_content}")

    def _run_memory_queries_with_models(
        self, db_path: Path, model_pack_dir: Path, results_path: Path
    ) -> bool:
        """Run built-in memory queries with custom model pack."""
        memory_queries = [
            # Core memory safety queries - these benefit from Hint models
            "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
            "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
            "codeql/cpp-queries:Critical/DoubleFree.ql",
            "codeql/cpp-queries:Critical/UseAfterFree.ql",
            # Allocation/deallocation mismatch queries
            "codeql/cpp-queries:Critical/NewArrayDeleteMismatch.ql",
            "codeql/cpp-queries:Critical/NewFreeMismatch.ql",
            "codeql/cpp-queries:Critical/NewDeleteArrayMismatch.ql",
        ]

        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "--format=sarif-latest",
            f"--output={results_path}",
            f"--additional-packs={model_pack_dir}",
            "--download",
        ] + memory_queries

        logger.info(f"Running CodeQL analyze...")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)

        stdout = result.stdout.decode()
        stderr = result.stderr.decode()

        logger.info(f"CodeQL analyze return code: {result.returncode}")
        if stdout:
            logger.debug(f"stdout:\n{stdout[:2000]}")
        if stderr:
            logger.info(f"stderr:\n{stderr[:2000]}")

        return results_path.exists()

    def _parse_sarif(self, sarif_path: Path) -> list[Warning]:
        """Parse SARIF results."""
        if not sarif_path.exists():
            logger.warning(f"SARIF not found: {sarif_path}")
            return []

        warnings = []
        try:
            data = json.loads(sarif_path.read_text())

            for run in data.get("runs", []):
                for result in run.get("results", []):
                    rule_id = result.get("ruleId", "").lower()

                    issue_type = MemoryIssueType.MEMORY_LEAK
                    for key, val in CODEQL_ISSUE_MAP.items():
                        if key in rule_id:
                            issue_type = val
                            break

                    locs = result.get("locations", [])
                    if not locs:
                        continue

                    loc = locs[0].get("physicalLocation", {})

                    warnings.append(Warning(
                        file_path=loc.get("artifactLocation", {}).get("uri", ""),
                        line_number=loc.get("region", {}).get("startLine", 0),
                        function_name="",
                        warning_type=rule_id,
                        message=result.get("message", {}).get("text", ""),
                        issue_type=issue_type,
                        trace=[],
                    ))

            logger.info(f"Parsed {len(warnings)} warnings")

        except Exception as e:
            logger.error(f"SARIF parse error: {e}")

        return warnings