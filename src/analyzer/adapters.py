"""Static analyzer adapters for CodeQL and Facebook Infer."""

import json
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from shutil import which

from src.core.models import Warning, AnnotationSet, MemoryIssueType, AnnotationType

logger = logging.getLogger(__name__)

CODEQL_ISSUE_MAP = {
    "memory-never-freed": MemoryIssueType.MEMORY_LEAK,
    "memory-may-not-be-freed": MemoryIssueType.MEMORY_LEAK,
    "double-free": MemoryIssueType.DOUBLE_FREE,
    "use-after-free": MemoryIssueType.USE_AFTER_FREE,
    "null-dereference": MemoryIssueType.NULL_DEREFERENCE,
    "hint-memory": MemoryIssueType.MEMORY_LEAK,
}


class CodeQLAnalyzer:
    """CodeQL analyzer with C/C++ model extension support.

    This analyzer uses CodeQL's model extension mechanism to define custom
    allocation and deallocation functions, then runs the built-in memory
    leak detection queries (MemoryNeverFreed.ql and MemoryMayNotBeFreed.ql).
    """

    def __init__(self, binary: str = "codeql", timeout: int = 600):
        self.binary = binary
        self.timeout = timeout

    def analyze(
        self, project_path: Path, annotations: AnnotationSet = None,
        issue_types: list[MemoryIssueType] = None
    ) -> list[Warning]:
        """Run CodeQL analysis with model extensions for custom allocators/deallocators."""
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

            # Step 2: Generate model pack with custom allocator/deallocator models
            if annotations and len(annotations.annotations) > 0:
                logger.info("Step 2: Generating model pack with custom models...")
                self._setup_model_pack(model_pack_dir, annotations)

                # Step 3: Run built-in memory queries with model pack
                logger.info("Step 3: Running memory leak queries with custom models...")
                if self._run_memory_queries_with_models(db_path, model_pack_dir, results_path):
                    return self._parse_sarif(results_path)

            # Fallback: Run built-in queries without custom models
            logger.info("Running built-in memory queries (no custom models)...")
            if self._run_memory_queries(db_path, results_path):
                return self._parse_sarif(results_path)

            return []

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # Cleanup
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    def _create_database(self, project_path: Path, db_path: Path) -> bool:
        """Create CodeQL database."""
        logger.info(f"Project: {project_path}")

        # Check for compile_commands.json first (preferred method)
        compile_commands = project_path / "compile_commands.json"
        if compile_commands.exists():
            logger.info("Using compile_commands.json for database creation")
            cmd = [
                self.binary, "database", "create",
                str(db_path),
                f"--source-root={project_path}",
                "--language=cpp",
                "--overwrite",
                f"--compilation-database={compile_commands}",
            ]
        else:
            # Determine build command
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
        """Determine appropriate build command for project.

        When analyzing original project (not merged), try to detect build system.
        When analyzing merged file, use direct compilation.
        """
        # Check for Makefile
        if (project_path / "Makefile").exists():
            if which("make") is None:
                logger.warning("Makefile found but 'make' command not available, skipping make detection")
            else:
                return "make"

        # Check for CMakeLists.txt in lint subdirectory (following README pattern)
        lint_dir = project_path / "lint"
        if lint_dir.exists() and (lint_dir / "CMakeLists.txt").exists():
            # Check if cmake is available before trying to use it
            if which("cmake") is None:
                logger.warning("CMakeLists.txt found in lint/ but 'cmake' command not available, skipping CMake detection")
            else:
                build_dir = lint_dir / "build"
                if build_dir.exists():
                    # Try using existing build directory
                    if (build_dir / "Makefile").exists():
                        return "make -C lint/build"
                else:
                    build_dir.mkdir(parents=True)
                    # Try configuring from lint/build directory
                    result = subprocess.run(
                        ["cmake", ".."], cwd=str(build_dir), capture_output=True, timeout=120
                    )
                    if result.returncode == 0:
                        return "make -C lint/build"

        # Check for CMakeLists.txt in project root
        if (project_path / "CMakeLists.txt").exists():
            # Check if cmake is available before trying to use it
            if which("cmake") is None:
                logger.warning("CMakeLists.txt found but 'cmake' command not available, skipping CMake detection")
            else:
                build_dir = project_path / "build"
                if build_dir.exists():
                    # Try using existing build directory
                    if (build_dir / "Makefile").exists():
                        return "make -C build"
                else:
                    build_dir.mkdir()
                    # Try configuring
                    result = subprocess.run(
                        ["cmake", ".."], cwd=str(build_dir), capture_output=True, timeout=120
                    )
                    if result.returncode == 0:
                        return "make -C build"

                # Try in-source build
                result = subprocess.run(
                    ["cmake", "."], cwd=str(project_path), capture_output=True, timeout=120
                )
                if result.returncode == 0:
                    return "make"

        # Fallback: Direct compilation (for merged files or simple projects)
        # Note: compile_commands.json is handled separately in _create_database()
        c_files = list(project_path.rglob("*.c"))
        cpp_files = list(project_path.rglob("*.cpp"))

        if c_files or cpp_files:
            files = [str(f.relative_to(project_path)) for f in (c_files + cpp_files)[:30]]
            # Determine compiler based on file types
            if cpp_files and not c_files:
                compiler = "clang++"
            elif c_files and not cpp_files:
                compiler = "clang"
            else:
                compiler = "clang++"  # Default to clang++ for mixed C/C++
            return f"{compiler} -I. -c -fsyntax-only {' '.join(files)}"

        return None

    def _setup_model_pack(self, model_pack_dir: Path, annotations: AnnotationSet) -> None:
        """Create a CodeQL model pack with custom allocation/deallocation/nullable models.

        This creates a library pack that extends AllocationFunction, DeallocationFunction,
        and adds nullable function models to recognize custom functions from annotations.
        """
        model_pack_dir.mkdir(parents=True, exist_ok=True)

        # Collect function annotations
        alloc_funcs = []
        free_funcs = []
        nullable_funcs = []  # Functions that may return NULL

        for func_name, anns in annotations.annotations.items():
            for ann in anns:
                if ann.annotation_type in (AnnotationType.ALLOC_SOURCE, AnnotationType.ARRAY_ALLOC,):
                    if func_name not in ("main", "_main") and func_name not in alloc_funcs:
                        alloc_funcs.append(func_name)
                        # Allocators can return NULL, so they're also nullable
                        if func_name not in nullable_funcs:
                            nullable_funcs.append(func_name)
                elif ann.annotation_type == AnnotationType.FREE_SINK:
                    if func_name not in free_funcs:
                        free_funcs.append(func_name)
                elif ann.annotation_type in (AnnotationType.NULLABLE_RETURN, AnnotationType.MUST_CHECK_NULL, AnnotationType.OWNERSHIP_RETURN):
                    if func_name not in nullable_funcs:
                        nullable_funcs.append(func_name)

        # Create qlpack.yml for the model pack
        qlpack_content = """name: hint/memory-models
version: 0.0.1
library: true
dependencies:
  codeql/cpp-all: "*"
dataExtensions:
  - models/**/*.yml
"""
        (model_pack_dir / "qlpack.yml").write_text(qlpack_content)

        # Create models directory
        models_dir = model_pack_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Generate the model extension YAML file
        # Note: For memory leak queries, we need to extend AllocationFunction and DeallocationFunction
        # This is done via a library (.qll) file that gets imported by the queries
        self._generate_model_extension_library(model_pack_dir, alloc_funcs, free_funcs, nullable_funcs)

        # Also generate a data extension file for dataflow models (if needed by some queries)
        self._generate_data_extension(models_dir, alloc_funcs, free_funcs, nullable_funcs)

        # Install dependencies
        logger.info("Installing model pack dependencies...")
        result = subprocess.run(
            [self.binary, "pack", "install", str(model_pack_dir)],
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            logger.warning(f"Pack install warning: {result.stderr.decode()[:500]}")

        logger.info(f"Model pack created with {len(alloc_funcs)} allocators, {len(free_funcs)} deallocators, {len(nullable_funcs)} nullable functions")

    def _generate_model_extension_library(
        self, model_pack_dir: Path, alloc_funcs: list[str], free_funcs: list[str], nullable_funcs: list[str] = None
    ) -> None:
        """Generate a CodeQL library (.qll) file that extends AllocationFunction, DeallocationFunction,
        and models nullable functions.

        This allows the built-in MemoryNeverFreed.ql, MemoryMayNotBeFreed.ql, and MissingNullTest.ql queries
        to recognize our custom allocators, deallocators, and nullable functions.
        """
        if nullable_funcs is None:
            nullable_funcs = []

        # Generate the list of function names for CodeQL
        alloc_names = ", ".join(f'"{f}"' for f in alloc_funcs) if alloc_funcs else '"__hint_no_custom_alloc__"'
        free_names = ", ".join(f'"{f}"' for f in free_funcs) if free_funcs else '"__hint_no_custom_free__"'
        nullable_names = ", ".join(f'"{f}"' for f in nullable_funcs) if nullable_funcs else '"__hint_no_custom_nullable__"'

        qll_content = f'''/**
 * @name HINT Custom Memory Models
 * @description Extends CodeQL's built-in allocation, deallocation, and nullable models
 *              with custom functions identified by LLM annotations.
 */

import cpp
private import semmle.code.cpp.models.interfaces.Allocation
private import semmle.code.cpp.models.interfaces.Deallocation

/**
 * Custom allocation functions identified by HINT/LLM annotations.
 * These functions allocate memory that must be freed.
 */
class HintAllocationFunction extends AllocationFunction {{
  HintAllocationFunction() {{
    this.getName() in [{alloc_names}]
  }}

  override predicate requiresDealloc() {{ any() }}
}}

/**
 * Custom deallocation functions identified by HINT/LLM annotations.
 * These functions free memory allocated by allocation functions.
 */
class HintDeallocationFunction extends DeallocationFunction {{
  HintDeallocationFunction() {{
    this.getName() in [{free_names}]
  }}

  override int getFreedArg() {{ result = 0 }}
}}

/**
 * Custom nullable functions identified by HINT/LLM annotations.
 * These functions may return NULL and their return values should be checked.
 * Used by MissingNullTest.ql to detect missing null checks.
 */
class HintNullableFunction extends Function {{
  HintNullableFunction() {{
    this.getName() in [{nullable_names}]
  }}

  /** Holds if the return value of this function may be NULL. */
  predicate mayReturnNull() {{ any() }}
}}

/**
 * Predicate to identify calls to nullable functions.
 * This can be used by queries to find unchecked return values.
 */
predicate isNullableFunctionCall(FunctionCall call) {{
  call.getTarget() instanceof HintNullableFunction
  or
  call.getTarget() instanceof HintAllocationFunction
  or
  // Standard library functions that may return NULL
  call.getTarget().getName() in [
    "malloc", "calloc", "realloc", "aligned_alloc", "memalign",
    "fopen", "fgets", "gets", "getenv", "bsearch",
    "strchr", "strrchr", "strstr", "strpbrk", "memchr",
    "tmpfile", "freopen", "strdup", "strndup"
  ]
}}
'''
        (model_pack_dir / "HintMemoryModels.qll").write_text(qll_content)
        logger.info(f"Generated HintMemoryModels.qll with {len(alloc_funcs)} allocators, {len(free_funcs)} deallocators, {len(nullable_funcs)} nullable functions")

    def _generate_data_extension(
        self, models_dir: Path, alloc_funcs: list[str], free_funcs: list[str], nullable_funcs: list[str] = None
    ) -> None:
        """Generate data extension YAML for dataflow models.

        This provides additional dataflow information that some queries may use.
        """
        if nullable_funcs is None:
            nullable_funcs = []

        extensions = []

        # Model allocators as sources (return value is a fresh allocation)
        if alloc_funcs:
            source_data = []
            for func in alloc_funcs:
                # Format: [namespace, type, subtypes, name, signature, ext, output, kind, provenance]
                source_data.append(["", "", False, func, "", "", "ReturnValue", "allocation", "manual"])

            extensions.append({
                "addsTo": {
                    "pack": "codeql/cpp-all",
                    "extensible": "sourceModel"
                },
                "data": source_data
            })

        # Model deallocators as sinks (argument is freed)
        if free_funcs:
            sink_data = []
            for func in free_funcs:
                # Format: [namespace, type, subtypes, name, signature, ext, input, kind, provenance]
                sink_data.append(["", "", False, func, "", "", "Argument[0]", "deallocation", "manual"])

            extensions.append({
                "addsTo": {
                    "pack": "codeql/cpp-all",
                    "extensible": "sinkModel"
                },
                "data": sink_data
            })

        # Model nullable functions as sources that may produce NULL
        # This helps queries like MissingNullTest detect unchecked return values
        if nullable_funcs:
            nullable_source_data = []
            for func in nullable_funcs:
                # Format: [namespace, type, subtypes, name, signature, ext, output, kind, provenance]
                # Using "local" kind to mark these as potential sources of null values
                nullable_source_data.append(["", "", False, func, "", "", "ReturnValue", "nullable", "manual"])

            extensions.append({
                "addsTo": {
                    "pack": "codeql/cpp-all",
                    "extensible": "sourceModel"
                },
                "data": nullable_source_data
            })

        if extensions:
            import yaml
            yaml_content = {"extensions": extensions}
            (models_dir / "hint-memory-models.model.yml").write_text(
                yaml.dump(yaml_content, default_flow_style=False, sort_keys=False)
            )
            logger.info(f"Generated hint-memory-models.model.yml with {len(alloc_funcs)} allocators, {len(free_funcs)} deallocators, {len(nullable_funcs)} nullable functions")

    def _run_memory_queries_with_models(
        self, db_path: Path, model_pack_dir: Path, results_path: Path
    ) -> bool:
        """Run built-in memory leak queries with custom model pack."""
        # The memory queries we want to run - correct path format: pack:path
        memory_queries = [
            "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
            "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
            "codeql/cpp-queries:Critical/DoubleFree.ql",
            "codeql/cpp-queries:Critical/UseAfterFree.ql",
            "codeql/cpp-queries:Critical/MissingNullTest.ql",
        ]

        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "--format=sarif-latest",
            f"--output={results_path}",
            f"--additional-packs={model_pack_dir}",
            "--download",  # Auto-download missing packs
        ] + memory_queries

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)

        if result.returncode != 0:
            logger.warning(f"Memory queries with models failed: {result.stderr.decode()[:500]}")
            # Try alternative approach: run query suite
            return self._run_memory_suite_with_models(db_path, model_pack_dir, results_path)

        return True

    def _run_memory_suite_with_models(
        self, db_path: Path, model_pack_dir: Path, results_path: Path
    ) -> bool:
        """Alternative: Run security-and-quality suite which includes memory queries."""
        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls",
            "--format=sarif-latest",
            f"--output={results_path}",
            f"--additional-packs={model_pack_dir}",
            "--download",
        ]

        logger.info(f"Running suite: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)

        if result.returncode != 0:
            logger.warning(f"Suite failed: {result.stderr.decode()[:500]}")
            return False
        return True

    def _run_memory_queries(self, db_path: Path, results_path: Path) -> bool:
        """Run built-in memory leak queries without custom models."""
        # Try running specific memory queries first - correct path format
        memory_queries = [
            "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
            "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
            "codeql/cpp-queries:Critical/DoubleFree.ql",
            "codeql/cpp-queries:Critical/UseAfterFree.ql",
            "codeql/cpp-queries:Critical/MissingNullTest.ql",
        ]

        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "--format=sarif-latest",
            f"--output={results_path}",
            "--download",
        ] + memory_queries

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)

        if result.returncode != 0:
            logger.warning(f"Specific memory queries failed, trying suite...")
            # Fallback to built-in suite
            return self._run_builtin(db_path, results_path)

        return True

    def _run_builtin(self, db_path: Path, results_path: Path) -> bool:
        """Run built-in query suite."""
        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls",
            "--format=sarif-latest",
            f"--output={results_path}",
            "--download",
        ]

        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)
        return result.returncode == 0

    def _parse_sarif(self, sarif_path: Path) -> list[Warning]:
        """Parse SARIF results."""
        if not sarif_path.exists():
            return []

        warnings = []
        try:
            data = json.loads(sarif_path.read_text())

            for run in data.get("runs", []):
                for result in run.get("results", []):
                    rule_id = result.get("ruleId", "").lower()

                    # Determine issue type
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
        except Exception as e:
            logger.warning(f"SARIF parse error: {e}")

        return warnings


class InferAnalyzer:
    """Facebook Infer static analyzer adapter.

    Infer is a static analysis tool that detects memory safety issues
    including null pointer dereferences, memory leaks, and resource leaks.

    Infer uses .inferconfig for configuration and supports custom models
    via .inferlibmodels files.
    """

    def __init__(self, binary: str = "infer", timeout: int = 600):
        self.binary = binary
        self.timeout = timeout

        # Infer checkers for different issue types
        self.issue_type_map = {
            "NULL_DEREFERENCE": MemoryIssueType.NULL_DEREFERENCE,
            "NULLPTR_DEREFERENCE": MemoryIssueType.NULL_DEREFERENCE,
            "MEMORY_LEAK": MemoryIssueType.MEMORY_LEAK,
            "RESOURCE_LEAK": MemoryIssueType.MEMORY_LEAK,
            "USE_AFTER_FREE": MemoryIssueType.USE_AFTER_FREE,
            "USE_AFTER_DELETE": MemoryIssueType.USE_AFTER_FREE,
            "DOUBLE_FREE": MemoryIssueType.DOUBLE_FREE,
            "USE_AFTER_LIFETIME": MemoryIssueType.USE_AFTER_FREE,
            "DANGLING_POINTER_DEREFERENCE": MemoryIssueType.USE_AFTER_FREE,
            "UNINITIALIZED_VALUE": MemoryIssueType.UNINITIALIZED_READ,
        }

    def analyze(
        self, project_path: Path, annotations: AnnotationSet = None,
        issue_types: list[MemoryIssueType] = None
    ) -> list[Warning]:
        """Run Infer analysis."""
        output_dir = Path(tempfile.mkdtemp())
        infer_out = output_dir / "infer-out"

        # Generate .inferconfig with custom models if we have annotations
        if annotations and len(annotations.annotations) > 0:
            self._generate_inferconfig(project_path, annotations)

        try:
            # Step 1: Run infer capture + analyze
            warnings = self._run_infer(project_path, infer_out)

            # Step 2: Filter by issue types if specified
            if issue_types:
                warnings = [w for w in warnings if w.issue_type in issue_types]

            return warnings

        except FileNotFoundError:
            logger.error(f"Infer binary not found: {self.binary}")
            return []
        except Exception as e:
            logger.error(f"Infer analysis failed: {e}")
            return []
        finally:
            # Cleanup .inferconfig if we created it
            inferconfig = project_path / ".inferconfig"
            if inferconfig.exists():
                try:
                    inferconfig.unlink()
                except:
                    pass

    def _run_infer(self, project_path: Path, infer_out: Path) -> list[Warning]:
        """Run Infer on project."""
        # Detect build system and run appropriate command
        warnings = []

        # Check for different build systems
        if (project_path / "Makefile").exists():
            warnings = self._run_with_make(project_path, infer_out)
        elif (project_path / "CMakeLists.txt").exists():
            warnings = self._run_with_cmake(project_path, infer_out)
        elif (project_path / "compile_commands.json").exists():
            warnings = self._run_with_compilation_db(project_path, infer_out)
        else:
            # Try direct capture on C files
            warnings = self._run_direct(project_path, infer_out)

        return warnings

    def _run_with_make(self, project_path: Path, infer_out: Path) -> list[Warning]:
        """Run Infer with make build system."""
        # Clean first
        subprocess.run(["make", "clean"], cwd=project_path, capture_output=True)

        cmd = [
            self.binary, "run",
            "-o", str(infer_out),
            "--", "make", "-j4"
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=project_path,
            capture_output=True,
            timeout=self.timeout
        )

        if result.returncode != 0:
            logger.warning(f"Infer returned {result.returncode}")
            stderr = result.stderr.decode()[:500]
            if stderr:
                logger.debug(f"stderr: {stderr}")

        return self._parse_results(infer_out)

    def _run_with_cmake(self, project_path: Path, infer_out: Path) -> list[Warning]:
        """Run Infer with CMake build system."""
        build_dir = project_path / "build"
        build_dir.mkdir(exist_ok=True)

        # Generate compile_commands.json
        subprocess.run(
            ["cmake", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", ".."],
            cwd=build_dir,
            capture_output=True
        )

        compile_commands = build_dir / "compile_commands.json"
        if compile_commands.exists():
            return self._run_with_compilation_db(project_path, infer_out, compile_commands)

        # Fallback to make
        cmd = [
            self.binary, "run",
            "-o", str(infer_out),
            "--", "cmake", "--build", str(build_dir)
        ]

        result = subprocess.run(cmd, cwd=project_path, capture_output=True, timeout=self.timeout)
        return self._parse_results(infer_out)

    def _run_with_compilation_db(
        self, project_path: Path, infer_out: Path,
        compile_commands: Path = None
    ) -> list[Warning]:
        """Run Infer with compilation database."""
        if compile_commands is None:
            compile_commands = project_path / "compile_commands.json"

        cmd = [
            self.binary, "run",
            "-o", str(infer_out),
            "--compilation-database", str(compile_commands)
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_path, capture_output=True, timeout=self.timeout)
        return self._parse_results(infer_out)

    def _run_direct(self, project_path: Path, infer_out: Path) -> list[Warning]:
        """Run Infer directly on C/C++ files."""
        # Find all C/C++ files
        c_files = list(project_path.glob("**/*.c")) + list(project_path.glob("**/*.cpp"))
        c_files = [f for f in c_files if "test" not in str(f).lower()]

        if not c_files:
            logger.warning("No C/C++ files found")
            return []

        # Use infer capture with clang
        for c_file in c_files[:50]:  # Limit to avoid timeout
            cmd = [
                self.binary, "capture",
                "-o", str(infer_out),
                "--", "clang", "-c", str(c_file), "-I", str(project_path)
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

        # Run analysis
        cmd = [self.binary, "analyze", "-o", str(infer_out)]
        subprocess.run(cmd, capture_output=True, timeout=self.timeout)

        return self._parse_results(infer_out)

    def _parse_results(self, infer_out: Path) -> list[Warning]:
        """Parse Infer JSON report."""
        warnings = []
        report_file = infer_out / "report.json"

        if not report_file.exists():
            logger.warning(f"Infer report not found: {report_file}")
            return []

        try:
            data = json.loads(report_file.read_text())

            for item in data:
                bug_type = item.get("bug_type", "")
                issue_type = self.issue_type_map.get(bug_type, MemoryIssueType.MEMORY_LEAK)

                # Extract trace
                trace = []
                for trace_item in item.get("bug_trace", []):
                    desc = trace_item.get("description", "")
                    filename = trace_item.get("filename", "")
                    line = trace_item.get("line_number", 0)
                    if desc:
                        trace.append(f"{filename}:{line}: {desc}")

                warnings.append(Warning(
                    file_path=item.get("file", ""),
                    line_number=item.get("line", 0),
                    function_name=item.get("procedure", ""),
                    warning_type=bug_type,
                    message=item.get("qualifier", ""),
                    issue_type=issue_type,
                    trace=trace,
                ))

            logger.info(f"Parsed {len(warnings)} Infer warnings")

        except Exception as e:
            logger.error(f"Failed to parse Infer results: {e}")

        return warnings

    def _generate_inferconfig(self, project_path: Path, annotations: AnnotationSet) -> None:
        """Generate .inferconfig with custom allocator/deallocator models."""
        config = {
            "report-suppress-errors": [],
            # Enable memory safety checkers
            "pulse": True,
            "biabduction": True,
        }

        # Collect custom allocators and deallocators
        alloc_funcs = []
        free_funcs = []

        for func_name, anns in annotations.annotations.items():
            for ann in anns:
                if ann.is_bug_annotation():
                    continue
                if ann.annotation_type in (AnnotationType.ALLOC_SOURCE, AnnotationType.ARRAY_ALLOC):
                    alloc_funcs.append(func_name)
                elif ann.annotation_type == AnnotationType.FREE_SINK:
                    free_funcs.append(func_name)

        # Write config
        config_path = project_path / ".inferconfig"
        config_path.write_text(json.dumps(config, indent=2))

        # Generate models file for custom allocators
        if alloc_funcs or free_funcs:
            self._generate_models_file(project_path, alloc_funcs, free_funcs)

        logger.info(f"Generated .inferconfig with {len(alloc_funcs)} allocators, {len(free_funcs)} deallocators")

    def _generate_models_file(
        self, project_path: Path,
        alloc_funcs: list[str],
        free_funcs: list[str]
    ) -> None:
        """Generate Infer models for custom allocators/deallocators.

        Creates a .infermodels file that tells Infer about custom
        memory management functions.
        """
        models = []

        # Model allocators
        for func in alloc_funcs:
            models.append({
                "procedure": func,
                "model": "allocation",
                "return": "fresh"
            })

        # Model deallocators
        for func in free_funcs:
            models.append({
                "procedure": func,
                "model": "deallocation",
                "parameter": 0
            })

        if models:
            models_path = project_path / ".infermodels"
            models_path.write_text(json.dumps(models, indent=2))
            logger.info(f"Generated Infer models file")


def create_analyzer(analyzer_type: str = "codeql", **kwargs):
    """Factory function to create analyzer."""
    if analyzer_type == "infer":
        return InferAnalyzer(**kwargs)
    return CodeQLAnalyzer(**kwargs)