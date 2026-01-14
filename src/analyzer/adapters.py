"""Static analyzer adapters for CodeQL and Facebook Infer.

关键理解：
- DoubleFree.ql, UseAfterFree.ql 使用 DeallocationExpr 来识别 free 操作
- DeallocationExpr 是通过 DeallocationFunction 派生的
- 如果 FunctionCall.getTarget() instanceof DeallocationFunction，那这个 call 就是 DeallocationExpr
- 所以扩展 DeallocationFunction 就能让这些 queries 识别自定义 deallocators

- MemoryNeverFreed.ql, MemoryMayNotBeFreed.ql 使用 AllocationExpr
- AllocationExpr 是通过 AllocationFunction 派生的
- 扩展 AllocationFunction 就能让这些 queries 识别自定义 allocators

- MissingNullTest.ql 检查 null 返回值，可能需要不同的机制
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
    # Memory leak queries
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
    # Null dereference
    "null-dereference": MemoryIssueType.NULL_DEREFERENCE,
    "missing-null-test": MemoryIssueType.NULL_DEREFERENCE,
    "cpp/missing-null-test": MemoryIssueType.NULL_DEREFERENCE,
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
            if has_custom_models:
                logger.info("Step 3: Running queries with custom models...")
                success = self._run_memory_queries_with_models(db_path, model_pack_dir, results_path)
            else:
                logger.info("Step 3: Running queries without custom models...")
                success = self._run_memory_queries(db_path, results_path)

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
        free_funcs = [fn for fn, _ in hints.get_deallocators()]
        nullable_funcs = hints.get_nullable_functions()

        logger.info(f"Allocators ({len(alloc_funcs)}): {alloc_funcs[:10]}...")
        logger.info(f"Deallocators ({len(free_funcs)}): {free_funcs[:10]}...")
        logger.info(f"Nullable ({len(nullable_funcs)}): {nullable_funcs[:10]}...")

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
        self._generate_model_library(model_pack_dir, alloc_funcs, free_funcs, nullable_funcs)

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
        free_funcs: list[str],
        nullable_funcs: list[str] = None
    ) -> None:
        """Generate CodeQL library extending AllocationFunction and DeallocationFunction.

        Key insight:
        - Queries like DoubleFree.ql check: fc instanceof DeallocationExpr
        - DeallocationExpr is satisfied when fc.getTarget() instanceof DeallocationFunction
        - So extending DeallocationFunction makes our custom deallocators recognized

        Same logic applies to AllocationFunction -> AllocationExpr for memory leak queries.

        For MissingNullTest.ql:
        - It checks AllocationExpr and verifies null check before dereference
        - Since AllocationExpr is derived from AllocationFunction, our custom allocators
          will also be checked for missing null tests
        - The predicate requiresDealloc() indicates memory that could fail (return NULL)
        """
        if nullable_funcs is None:
            nullable_funcs = []

        # Build the function name lists for CodeQL IN clause
        # Need at least one item to avoid syntax error
        if alloc_funcs:
            alloc_names = ", ".join(f'"{f}"' for f in alloc_funcs)
        else:
            alloc_names = '"__no_custom_allocator__"'

        if free_funcs:
            free_names = ", ".join(f'"{f}"' for f in free_funcs)
        else:
            free_names = '"__no_custom_deallocator__"'

        # Generate .qll content
        qll_content = f'''/**
 * HINT Custom Memory Models
 *
 * Extends CodeQL's built-in AllocationFunction and DeallocationFunction
 * with custom functions identified by LLM annotations.
 *
 * This enables built-in queries like:
 * - MemoryNeverFreed.ql (uses AllocationExpr)
 * - MemoryMayNotBeFreed.ql (uses AllocationExpr)
 * - DoubleFree.ql (uses DeallocationExpr)
 * - UseAfterFree.ql (uses DeallocationExpr)
 * - MissingNullTest.ql (uses AllocationExpr - checks null before dereference)
 * to recognize our custom allocators and deallocators.
 */

import cpp
private import semmle.code.cpp.models.interfaces.Allocation
private import semmle.code.cpp.models.interfaces.Deallocation

/**
 * Custom allocation functions from LLM annotations.
 *
 * When a FunctionCall's target is HintAllocationFunction,
 * the call becomes an AllocationExpr, which:
 * - Memory leak queries detect (MemoryNeverFreed, MemoryMayNotBeFreed)
 * - MissingNullTest detects (checks if return value is null-checked before use)
 */
class HintAllocationFunction extends AllocationFunction {{
  HintAllocationFunction() {{
    this.getName() in [{alloc_names}]
  }}

  // This allocation requires deallocation - memory must be freed
  override predicate requiresDealloc() {{ any() }}
}}

/**
 * Custom deallocation functions from LLM annotations.
 *
 * When a FunctionCall's target is HintDeallocationFunction,
 * the call becomes a DeallocationExpr, which double-free
 * and use-after-free queries detect.
 */
class HintDeallocationFunction extends DeallocationFunction {{
  HintDeallocationFunction() {{
    this.getName() in [{free_names}]
  }}

  // The first argument (index 0) is the pointer being freed
  override int getFreedArg() {{ result = 0 }}
}}
'''
        qll_path = model_pack_dir / "HintMemoryModels.qll"
        qll_path.write_text(qll_content)
        logger.info(f"Generated: {qll_path}")

        # Log the content for debugging
        logger.debug(f"QLL content:\n{qll_content}")

    def _run_memory_queries_with_models(
        self, db_path: Path, model_pack_dir: Path, results_path: Path
    ) -> bool:
        """Run built-in memory queries with custom model pack."""
        memory_queries = [
            "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
            "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
            "codeql/cpp-queries:Critical/DoubleFree.ql",
            "codeql/cpp-queries:Critical/UseAfterFree.ql",
            "codeql/cpp-queries:Critical/MissingNullTest.ql",
            "codeql/cpp-queries:Critical/OverflowCalculated.ql",
            "codeql/cpp-queries:Critical/OverflowDestination.ql",
            "codeql/cpp-queries:Critical/OverflowStatic.ql",
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

        if result.returncode != 0:
            logger.warning("Query execution returned non-zero, trying fallback...")
            return self._run_memory_queries(db_path, results_path)

        return results_path.exists()

    def _run_memory_queries(self, db_path: Path, results_path: Path) -> bool:
        """Run built-in memory queries without custom models."""
        memory_queries = [
            "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
            "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
            "codeql/cpp-queries:Critical/DoubleFree.ql",
            "codeql/cpp-queries:Critical/UseAfterFree.ql",
            "codeql/cpp-queries:Critical/MissingNullTest.ql",
            "codeql/cpp-queries:Critical/OverflowCalculated.ql",
            "codeql/cpp-queries:Critical/OverflowDestination.ql",
            "codeql/cpp-queries:Critical/OverflowStatic.ql",
        ]

        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "--format=sarif-latest",
            f"--output={results_path}",
            "--download",
        ] + memory_queries

        logger.info(f"Running CodeQL analyze (no custom models)...")
        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)

        if result.returncode != 0:
            logger.warning(f"Queries failed: {result.stderr.decode()[:500]}")

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


class InferAnalyzer:
    """Facebook Infer static analyzer adapter."""

    def __init__(self, binary: str = "infer", timeout: int = 600):
        self.binary = binary
        self.timeout = timeout
        self.issue_type_map = {
            "NULL_DEREFERENCE": MemoryIssueType.NULL_DEREFERENCE,
            "NULLPTR_DEREFERENCE": MemoryIssueType.NULL_DEREFERENCE,
            "MEMORY_LEAK": MemoryIssueType.MEMORY_LEAK,
            "RESOURCE_LEAK": MemoryIssueType.MEMORY_LEAK,
            "USE_AFTER_FREE": MemoryIssueType.USE_AFTER_FREE,
            "USE_AFTER_DELETE": MemoryIssueType.USE_AFTER_FREE,
            "DOUBLE_FREE": MemoryIssueType.DOUBLE_FREE,
        }

    def analyze(self, project_path: Path, hints: HintSet = None, issue_types: list[MemoryIssueType] = None) -> list[Warning]:
        output_dir = Path(tempfile.mkdtemp())
        infer_out = output_dir / "infer-out"

        if hints and len(hints.hints) > 0:
            self._generate_inferconfig(project_path, hints)

        try:
            warnings = self._run_infer(project_path, infer_out)
            if issue_types:
                warnings = [w for w in warnings if w.issue_type in issue_types]
            return warnings
        except Exception as e:
            logger.error(f"Infer failed: {e}")
            return []
        finally:
            inferconfig = project_path / ".inferconfig"
            if inferconfig.exists():
                inferconfig.unlink(missing_ok=True)

    def _run_infer(self, project_path: Path, infer_out: Path) -> list[Warning]:
        if (project_path / "Makefile").exists():
            subprocess.run(["make", "clean"], cwd=project_path, capture_output=True)
            cmd = [self.binary, "run", "-o", str(infer_out), "--", "make", "-j4"]
        elif (project_path / "compile_commands.json").exists():
            cmd = [self.binary, "run", "-o", str(infer_out), "--compilation-database", str(project_path / "compile_commands.json")]
        else:
            c_files = list(project_path.glob("**/*.c")) + list(project_path.glob("**/*.cpp"))
            if not c_files:
                return []
            for f in c_files[:50]:
                subprocess.run([self.binary, "capture", "-o", str(infer_out), "--", "clang", "-c", str(f)], capture_output=True, timeout=60)
            cmd = [self.binary, "analyze", "-o", str(infer_out)]

        subprocess.run(cmd, cwd=project_path, capture_output=True, timeout=self.timeout)
        return self._parse_results(infer_out)

    def _parse_results(self, infer_out: Path) -> list[Warning]:
        report_file = infer_out / "report.json"
        if not report_file.exists():
            return []

        warnings = []
        try:
            for item in json.loads(report_file.read_text()):
                bug_type = item.get("bug_type", "")
                issue_type = self.issue_type_map.get(bug_type, MemoryIssueType.MEMORY_LEAK)
                warnings.append(Warning(
                    file_path=item.get("file", ""),
                    line_number=item.get("line", 0),
                    function_name=item.get("procedure", ""),
                    warning_type=bug_type,
                    message=item.get("qualifier", ""),
                    issue_type=issue_type,
                    trace=[],
                ))
        except Exception as e:
            logger.error(f"Infer parse error: {e}")
        return warnings

    def _generate_inferconfig(self, project_path: Path, hints: HintSet) -> None:
        config = {"pulse": True, "biabduction": True}
        (project_path / ".inferconfig").write_text(json.dumps(config, indent=2))


def create_analyzer(analyzer_type: str = "codeql", **kwargs):
    if analyzer_type == "infer":
        return InferAnalyzer(**kwargs)
    return CodeQLAnalyzer(**kwargs)