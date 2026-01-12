"""Static analyzer adapters for CodeQL and Facebook Infer.

TODO: Facebook Infer's adapter is not fully implemented yet.
"""

import json
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

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
    """CodeQL analyzer with custom query support."""

    def __init__(self, binary: str = "codeql", timeout: int = 600):
        self.binary = binary
        self.timeout = timeout

    def analyze(
        self, project_path: Path, annotations: AnnotationSet = None,
        issue_types: list[MemoryIssueType] = None
    ) -> list[Warning]:
        """Run CodeQL analysis with custom annotations."""
        output_dir = Path(tempfile.mkdtemp())
        db_path = output_dir / "codeql-db"
        results_path = output_dir / "results.sarif"
        query_dir = output_dir / "queries"
        query_dir.mkdir(exist_ok=True)

        try:
            # Step 1: Create database
            logger.info("Step 1: Creating CodeQL database...")
            if not self._create_database(project_path, db_path):
                logger.error("Failed to create database")
                return []

            # Step 2: Generate and run custom query
            if annotations and len(annotations.annotations) > 0:
                logger.info("Step 2: Generating custom query...")
                self._setup_query_pack(query_dir, annotations)

                logger.info("Step 3: Running custom query...")
                if self._run_query(db_path, str(query_dir), results_path):
                    return self._parse_sarif(results_path)

            # Step 3: Fallback - try built-in suite
            logger.info("Trying built-in query suite...")
            if self._run_builtin(db_path, results_path):
                return self._parse_sarif(results_path)

            return []

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _create_database(self, project_path: Path, db_path: Path) -> bool:
        """Create CodeQL database."""
        logger.info(f"Project: {project_path}")

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
        """Determine appropriate build command for project."""
        # # Check for Makefile
        # if (project_path / "Makefile").exists():
        #     return "make"

        # # Check for CMakeLists.txt
        # if (project_path / "CMakeLists.txt").exists():
        #     build_dir = project_path / "build"
        #     if build_dir.exists():
        #         shutil.rmtree(build_dir)
        #     build_dir.mkdir()

        #     # Configure
        #     result = subprocess.run(
        #         ["cmake", ".."], cwd=str(build_dir), capture_output=True, timeout=120
        #     )
        #     if result.returncode == 0:
        #         return "make -C build"

        #     # Try in-source
        #     result = subprocess.run(
        #         ["cmake", "."], cwd=str(project_path), capture_output=True, timeout=120
        #     )
        #     if result.returncode == 0:
        #         return "make"

        # Direct compilation
        c_files = list(project_path.rglob("*.c"))
        cpp_files = list(project_path.rglob("*.cpp"))

        if c_files or cpp_files:
            files = [str(f.relative_to(project_path)) for f in (c_files + cpp_files)[:30]]
            compiler = "clang" if cpp_files else "clang"  # Use clang for better compatibility
            return f"{compiler} -I. -c -fsyntax-only {' '.join(files)}"

        return None

    def _setup_query_pack(self, query_dir: Path, annotations: AnnotationSet) -> None:
        """Create query pack with custom query."""
        # Create qlpack.yml
        qlpack = """name: hint-queries
version: 0.0.1
dependencies:
  codeql/cpp-all: "*"
"""
        (query_dir / "qlpack.yml").write_text(qlpack)

        # Install dependencies
        subprocess.run(
            [self.binary, "pack", "install", str(query_dir)],
            capture_output=True, timeout=300
        )

        # Generate query
        self.generate_custom_query(annotations, query_dir / "hint_memory.ql")

    def _run_query(self, db_path: Path, query_path: str, results_path: Path) -> bool:
        """Run CodeQL query."""
        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            query_path,
            "--format=sarif-latest",
            f"--output={results_path}",
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)

        if result.returncode != 0:
            logger.warning(f"Query failed: {result.stderr.decode()[:500]}")
            return False
        return True

    def _run_builtin(self, db_path: Path, results_path: Path) -> bool:
        """Run built-in query suite."""
        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "cpp-security-and-quality",
            "--format=sarif-latest",
            f"--output={results_path}",
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

    def generate_custom_query(self, annotations: AnnotationSet, output_path: Path) -> None:
        """Generate comprehensive CodeQL query for memory safety.

        Uses two sources of information:
        1. Function annotations (allocators/deallocators)
        2. Bug annotations (LLM-detected potential bugs) - used as hints
        """
        # Collect allocators and deallocators
        alloc_funcs = []
        free_funcs = []

        # Collect LLM-detected bugs as hints for CodeQL
        bug_hints = {
            "leak_vars": [],      # Variables that might leak
            "uaf_vars": [],       # Variables with potential UAF
            "double_free_vars": [], # Variables with potential double-free
            "suspect_functions": set(),  # Functions with potential bugs
        }

        for func_name, anns in annotations.annotations.items():
            for ann in anns:
                if ann.annotation_type in (AnnotationType.ALLOC_SOURCE, AnnotationType.ARRAY_ALLOC):
                    if func_name not in ("main", "_main"):
                        alloc_funcs.append(func_name)
                elif ann.annotation_type == AnnotationType.FREE_SINK:
                    free_funcs.append(func_name)
                # Collect bug hints
                elif ann.annotation_type == AnnotationType.POTENTIAL_LEAK:
                    bug_hints["leak_vars"].append((func_name, ann.target))
                    bug_hints["suspect_functions"].add(func_name)
                elif ann.annotation_type == AnnotationType.USE_AFTER_FREE:
                    bug_hints["uaf_vars"].append((func_name, ann.target))
                    bug_hints["suspect_functions"].add(func_name)
                elif ann.annotation_type == AnnotationType.DOUBLE_FREE:
                    bug_hints["double_free_vars"].append((func_name, ann.target))
                    bug_hints["suspect_functions"].add(func_name)

        alloc_list = ", ".join(f'"{f}"' for f in alloc_funcs) if alloc_funcs else '"__hint_none__"'
        free_list = ", ".join(f'"{f}"' for f in free_funcs) if free_funcs else '"__hint_none__"'

        # Generate suspect function list for targeted analysis
        suspect_list = ", ".join(f'"{f}"' for f in bug_hints["suspect_functions"]) if bug_hints["suspect_functions"] else '"__hint_none__"'

        # Generate variable hints
        leak_hints = " or ".join(
            f'(f.getName() = "{func}" and v.getName() = "{var}")'
            for func, var in bug_hints["leak_vars"]
        ) if bug_hints["leak_vars"] else "none()"

        uaf_hints = " or ".join(
            f'(f.getName() = "{func}" and v.getName() = "{var}")'
            for func, var in bug_hints["uaf_vars"]
        ) if bug_hints["uaf_vars"] else "none()"

        query = f'''/**
 * @name HINT Comprehensive Memory Safety Check
 * @description Detects memory leaks, use-after-free, and double-free
 * @kind problem
 * @problem.severity error
 * @precision medium
 * @id cpp/hint-memory-safety
 */

import cpp

//=============================================================================
// Allocation and Deallocation Functions
//=============================================================================

class CustomAlloc extends Function {{
  CustomAlloc() {{ this.getName() in [{alloc_list}] }}
}}

class CustomFree extends Function {{
  CustomFree() {{ this.getName() in [{free_list}] }}
}}

class AnyAlloc extends Function {{
  AnyAlloc() {{
    this.getName() in ["malloc", "calloc", "realloc", "strdup", "strndup", "aligned_alloc"]
    or this instanceof CustomAlloc
  }}
}}

class AnyFree extends Function {{
  AnyFree() {{
    this.getName() in ["free", "cfree", "g_free"]
    or this instanceof CustomFree
  }}
}}

//=============================================================================
// Helper predicates
//=============================================================================

// Get the variable that receives an allocation (handles both init and assignment)
Variable getAllocatedVar(FunctionCall allocCall) {{
  // Case 1: int *p = malloc(...)
  result.getInitializer().getExpr() = allocCall
  or
  // Case 2: p = malloc(...)
  exists(AssignExpr assign |
    assign.getRValue() = allocCall and
    result = assign.getLValue().(VariableAccess).getTarget()
  )
}}

// Check if variable is freed in function
predicate isFreedInFunc(Variable v, Function f) {{
  exists(FunctionCall freeCall |
    freeCall.getTarget() instanceof AnyFree and
    freeCall.getEnclosingFunction() = f and
    freeCall.getAnArgument().(VariableAccess).getTarget() = v
  )
}}

// Check if variable is returned
predicate isReturnedFromFunc(Variable v, Function f) {{
  exists(ReturnStmt ret |
    ret.getEnclosingFunction() = f and
    ret.getExpr().(VariableAccess).getTarget() = v
  )
}}

// Check if variable escapes
predicate varEscapes(Variable v) {{
  exists(AssignExpr a |
    a.getRValue().(VariableAccess).getTarget() = v and
    (
      a.getLValue().(VariableAccess).getTarget() instanceof GlobalVariable or
      a.getLValue() instanceof FieldAccess
    )
  )
  or
  // Passed to function that might store it
  exists(FunctionCall fc |
    fc.getAnArgument().(VariableAccess).getTarget() = v and
    not fc.getTarget() instanceof AnyFree and
    not fc.getTarget().getName() in ["printf", "fprintf", "sprintf", "snprintf", "puts", "fputs"]
  )
}}

//=============================================================================
// Use-After-Free Detection
//=============================================================================

// A dereference of a pointer variable
class PointerDeref extends Expr {{
  Variable ptrVar;

  PointerDeref() {{
    (
      this instanceof PointerDereferenceExpr and
      ptrVar = this.(PointerDereferenceExpr).getOperand().(VariableAccess).getTarget()
    ) or (
      this instanceof ArrayExpr and
      ptrVar = this.(ArrayExpr).getArrayBase().(VariableAccess).getTarget()
    ) or (
      this instanceof PointerFieldAccess and
      ptrVar = this.(PointerFieldAccess).getQualifier().(VariableAccess).getTarget()
    )
  }}

  Variable getPtrVar() {{ result = ptrVar }}
}}

// Free call
class FreeCall extends FunctionCall {{
  Variable freedVar;

  FreeCall() {{
    this.getTarget() instanceof AnyFree and
    freedVar = this.getAnArgument().(VariableAccess).getTarget()
  }}

  Variable getFreedVar() {{ result = freedVar }}
}}

// Simple heuristic: deref after free by line number
predicate useAfterFreeSimple(FreeCall freeCall, PointerDeref deref, Variable v) {{
  freeCall.getFreedVar() = v and
  deref.getPtrVar() = v and
  freeCall.getEnclosingFunction() = deref.getEnclosingFunction() and
  freeCall.getLocation().getStartLine() < deref.getLocation().getStartLine()
}}

//=============================================================================
// Double-Free Detection
//=============================================================================

predicate doubleFreeSimple(FreeCall f1, FreeCall f2, Variable v) {{
  f1.getFreedVar() = v and
  f2.getFreedVar() = v and
  f1 != f2 and
  f1.getEnclosingFunction() = f2.getEnclosingFunction() and
  f1.getLocation().getStartLine() < f2.getLocation().getStartLine()
}}

//=============================================================================
// Main Query
//=============================================================================

from Expr loc, string issue, string detail, Function f
where
  // Memory Leak
  (
    exists(FunctionCall allocCall, Variable v |
      allocCall.getTarget() instanceof AnyAlloc and
      v = getAllocatedVar(allocCall) and
      f = allocCall.getEnclosingFunction() and
      not isFreedInFunc(v, f) and
      not isReturnedFromFunc(v, f) and
      not varEscapes(v) and
      loc = allocCall and
      issue = "Memory Leak" and
      detail = "'" + v.getName() + "' allocated but never freed in " + f.getName() + "()"
    )
  )
  or
  // Use-After-Free
  (
    exists(FreeCall freeCall, PointerDeref deref, Variable v |
      useAfterFreeSimple(freeCall, deref, v) and
      f = freeCall.getEnclosingFunction() and
      loc = deref and
      issue = "Use-After-Free" and
      detail = "'" + v.getName() + "' dereferenced after free at line " + freeCall.getLocation().getStartLine().toString()
    )
  )
  or
  // Double-Free
  (
    exists(FreeCall f1, FreeCall f2, Variable v |
      doubleFreeSimple(f1, f2, v) and
      f = f1.getEnclosingFunction() and
      loc = f2 and
      issue = "Double-Free" and
      detail = "'" + v.getName() + "' freed again (first at line " + f1.getLocation().getStartLine().toString() + ")"
    )
  )
select loc, issue + ": " + detail
'''

        # Add LLM hints section if available
        if bug_hints["suspect_functions"]:
            hint_comment = f'''
// =============================================================================
// LLM-detected suspect locations (prioritize these in analysis):
// Functions: {list(bug_hints["suspect_functions"])}
// Potential leaks: {bug_hints["leak_vars"]}
// Potential UAF: {bug_hints["uaf_vars"]}
// Potential double-free: {bug_hints["double_free_vars"]}
// =============================================================================
'''
            query = query.replace("import cpp", f"import cpp\n{hint_comment}")

        output_path.write_text(query)
        logger.info(f"Generated query: {output_path}")
        logger.info(f"  Custom allocators: {alloc_funcs}")
        logger.info(f"  Custom deallocators: {free_funcs}")
        if bug_hints["suspect_functions"]:
            logger.info(f"  LLM suspect functions: {bug_hints['suspect_functions']}")


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