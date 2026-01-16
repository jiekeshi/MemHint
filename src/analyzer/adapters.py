import json
import logging
import re
import subprocess
import tempfile
import shutil
import yaml
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
}


class CodeQLAnalyzer:
    """CodeQL analyzer with direct model injection into codeql/cpp-all ext directory."""

    def __init__(self, binary: str = "codeql", timeout: int = 600, codeql_dir: Path | None = None, cpp_queries_dir: Path | None = None, reuse_db: bool = True):
        self.binary = binary
        self.timeout = timeout
        self.codeql_dir = codeql_dir  # Optional custom CodeQL directory
        self.cpp_queries_dir = cpp_queries_dir  # Optional direct path to cpp-queries directory
        self.reuse_db = reuse_db  # Whether to reuse existing databases
        self._injected_files: list[Path] = []  # 记录注入的文件，用于清理
    
    def _delete_database(self, db_path: Path) -> bool:
        """Delete an existing CodeQL database."""
        if not db_path.exists():
            return True
        
        try:
            logger.info(f"Deleting existing database at {db_path}")
            shutil.rmtree(db_path)
            logger.info(f"Database deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete database at {db_path}: {e}")
            return False

    def analyze(
        self, project_path: Path, hints: HintSet = None,
        issue_types: list[MemoryIssueType] = None
    ) -> list[Warning]:
        """Run CodeQL analysis with injected custom models."""
        output_dir = Path(tempfile.mkdtemp())
        results_path = output_dir / "results.sarif"

        try:
            # Step 1: Get or create database
            logger.info("Step 1: Getting CodeQL database...")
            db_path = self._get_or_create_database(project_path)
            if not db_path:
                logger.error("Failed to get or create database")
                return []

            # Step 2: Inject custom memory models directly into codeql/cpp-all
            if hints and len(hints.hints) > 0:
                logger.info("Step 2: Injecting custom memory models...")
                self._inject_models(hints)

                self._verify_injected_models()

            # Step 3: Run queries
            logger.info("Step 3: Running queries with custom models...")
            success = self._run_memory_queries(db_path, results_path)

            if success:
                return self._parse_sarif(results_path, project_path)
            return []

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            self._cleanup_models()

            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
    
    def _get_db_path(self, project_path: Path) -> Path:
        """Get the path where CodeQL database should be stored for this project."""
        # Store database in project directory under .codeql-db
        return project_path / ".codeql-db"
    
    def _is_valid_database(self, db_path: Path) -> bool:
        logger.info(f"Checking if database at {db_path} is valid...")
        """Check if a CodeQL database exists and is valid C++ database."""
        if not db_path.exists():
            logger.error(f"Database not found at {db_path}")
            return False
        
        # Check if it's a valid CodeQL database by checking for database metadata
        # CodeQL databases have a codeql-database.yml file
        db_metadata = db_path / "codeql-database.yml"
        if not db_metadata.exists():
            logger.error(f"Database metadata not found at {db_metadata}")
            return False
        
        # Verify database is not corrupted and is a C++ database
        # Read the database metadata file directly instead of using CLI command
        try:
            with open(db_metadata, 'r') as f:
                db_info = yaml.safe_load(f)
            
            # Check that it's a C++ database
            primary_language = db_info.get('primaryLanguage', '').lower()
            if primary_language not in ('cpp', 'c++'):
                logger.warning(f"Database at {db_path} is not a C++ database (primaryLanguage: {primary_language})")
                return False
            
            # Check if database is finalized (required for running queries)
            is_finalized = db_info.get('finalised', False)
            if not is_finalized:
                logger.warning(f"Database at {db_path} is not finalized, will need to finalize before use")
                # We'll finalize it if needed, but mark as valid for now
                # The finalize step will be called if needed
            
            logger.info(f"Database is valid C++ database (primaryLanguage: {primary_language}, finalized: {is_finalized})")
            return True
        except yaml.YAMLError as e:
            logger.error(f"Error parsing database metadata: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking database validity: {e}")
            return False
    
    def _get_or_create_database(self, project_path: Path) -> Path | None:
        """Get existing database or create a new one.
        
        Returns:
            Path to the database, or None if creation failed.
        """
        db_path = self._get_db_path(project_path)
        
        # If reuse_db is False, delete existing database first
        if not self.reuse_db and db_path.exists():
            if not self._delete_database(db_path):
                logger.error("Failed to delete existing database, aborting")
                return None
        
        # Check if we should reuse existing database
        if self.reuse_db and self._is_valid_database(db_path):
            logger.info(f"Reusing existing CodeQL database at {db_path}")
            # Ensure database is finalized before use
            if not self._ensure_database_finalized(db_path):
                logger.warning("Failed to finalize database, will recreate")
                shutil.rmtree(db_path, ignore_errors=True)
            else:
                return db_path
        
        # Create new database
        if self.reuse_db and db_path.exists():
            logger.info(f"Existing database at {db_path} is invalid or corrupted, recreating...")
            shutil.rmtree(db_path, ignore_errors=True)
        
        logger.info(f"Creating new CodeQL database at {db_path}")
        if not self._create_database(project_path, db_path):
            return None
        
        return db_path

    def _get_codeql_cpp_ext_dir(self) -> Path:
        # If direct cpp-queries directory is provided, use it
        if self.cpp_queries_dir:
            if not self.cpp_queries_dir.exists():
                raise FileNotFoundError(f"CodeQL cpp-queries directory not found at {self.cpp_queries_dir}")
            
            # Direct path to cpp-queries, find cpp-all library
            # Structure can be:
            # Option 1: cpp-queries/version/.codeql/libraries/codeql/cpp-all/version/ext
            # Option 2: cpp-queries/.codeql/libraries/codeql/cpp-all/version/ext
            
            # First, check if there's a version directory
            version_dirs = [d for d in self.cpp_queries_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if version_dirs:
                # Use the latest version directory
                version_dir = sorted(version_dirs, reverse=True)[0]
                codeql_dir = version_dir / ".codeql" / "libraries" / "codeql" / "cpp-all"
            else:
                # No version directory, check directly
                codeql_dir = self.cpp_queries_dir / ".codeql" / "libraries" / "codeql" / "cpp-all"
            
            if not codeql_dir.exists():
                raise FileNotFoundError(f"CodeQL cpp-all library not found at {codeql_dir}")
            
            version_cpp_all = sorted([d for d in codeql_dir.iterdir() if d.is_dir()], reverse=True)
            if not version_cpp_all:
                raise FileNotFoundError(f"No versions found in {codeql_dir}")
            
            ext_dir = codeql_dir / version_cpp_all[0] / "ext"
            logger.info(f"Using CodeQL cpp-all ext dir (direct path): {ext_dir}")
            return ext_dir
        
        # Otherwise, use the standard approach with codeql_dir or default
        if self.codeql_dir:
            base = self.codeql_dir / "packages" / "codeql" / "cpp-queries"
        else:
            base = Path.home() / ".codeql" / "packages" / "codeql" / "cpp-queries"

        if not base.exists():
            raise FileNotFoundError(f"CodeQL cpp-queries not found at {base}")

        version_codeql = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
        if not version_codeql:
            raise FileNotFoundError(f"No versions found in {base}")

        codeql_dir = version_codeql[0] / ".codeql" / "libraries" / "codeql" / "cpp-all"

        version_cpp_all = sorted([d for d in codeql_dir.iterdir() if d.is_dir()], reverse=True)
        if not version_cpp_all:
            raise FileNotFoundError(f"No versions found in {base}")

        ext_dir = codeql_dir / version_cpp_all[0] / "ext"

        logger.info(f"Using CodeQL cpp-all ext dir: {ext_dir}")

        return ext_dir

    def _inject_models(self, hints: HintSet) -> None:
        ext_dir = self._get_codeql_cpp_ext_dir()

        alloc_funcs = [f for f in hints.get_allocators() if f not in ("main", "_main", "")]
        free_funcs = [(f, idx) for f, idx in hints.get_deallocators() if f not in ("main", "_main", "")]

        logger.info(f"Allocators ({len(alloc_funcs)}): {alloc_funcs[:10]}...")
        logger.info(f"Deallocators ({len(free_funcs)}): {free_funcs[:10]}...")

        # 注入 allocation models
        if alloc_funcs:
            alloc_path = self._write_allocation_model(ext_dir, alloc_funcs)
            self._injected_files.append(alloc_path)

        # 注入 deallocation models
        if free_funcs:
            dealloc_path = self._write_deallocation_model(ext_dir, free_funcs)
            self._injected_files.append(dealloc_path)

    def _write_allocation_model(self, ext_dir: Path, alloc_funcs: list[str]) -> Path:
        """写入 allocation model 文件"""
        allocation_dir = ext_dir / "allocation"
        allocation_dir.mkdir(parents=True, exist_ok=True)

        # 生成 YAML - 参考官方格式
        # allocationFunctionModel(namespace, type, subtypes, name, sizeArg, multArg, reallocPtrArg, requiresDealloc)
        lines = ["extensions:"]
        lines.append("  - addsTo:")
        lines.append("      pack: codeql/cpp-all")
        lines.append("      extensible: allocationFunctionModel")
        lines.append("    data:")

        for func_name in alloc_funcs:
            # ["", "", false, "func_name", "", "", "", true]
            lines.append(f'      - ["", "", false, "{func_name}", "", "", "", true]')

        yaml_content = "\n".join(lines) + "\n"

        output_path = allocation_dir / "hint.allocation.model.yml"
        output_path.write_text(yaml_content)

        logger.info(f"Written allocation model: {output_path}")
        logger.debug(f"Content:\n{yaml_content}")

        return output_path

    def _write_deallocation_model(self, ext_dir: Path, free_funcs: list[tuple[str, int]]) -> Path:
        """写入 deallocation model 文件"""
        deallocation_dir = ext_dir / "deallocation"
        deallocation_dir.mkdir(parents=True, exist_ok=True)

        # 生成 YAML
        # deallocationFunctionModel(namespace, type, subtypes, name, freedArg)
        lines = ["extensions:"]
        lines.append("  - addsTo:")
        lines.append("      pack: codeql/cpp-all")
        lines.append("      extensible: deallocationFunctionModel")
        lines.append("    data:")

        for func_name, arg_idx in free_funcs:
            freed_arg = f"Argument[{arg_idx}]"
            lines.append(f'      - ["", "", false, "{func_name}", "{freed_arg}"]')

        yaml_content = "\n".join(lines) + "\n"

        output_path = deallocation_dir / "hint.deallocation.model.yml"
        output_path.write_text(yaml_content)

        logger.info(f"Written deallocation model: {output_path}")
        logger.debug(f"Content:\n{yaml_content}")

        return output_path

    def _verify_injected_models(self) -> None:
        cmd = [
            self.binary, "resolve", "extensions",
            "codeql/cpp-queries",
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60)

        if result.returncode == 0:
            try:
                data = json.loads(result.stdout.decode())
                all_files = []
                for pack_path, entries in data.get("data", {}).items():
                    for entry in entries:
                        all_files.append(entry.get("file", ""))

                for injected in self._injected_files:
                    found = any(str(injected) in f for f in all_files)
                    if found:
                        logger.info(f"✓ Verified: {injected.name} is loaded")
                    else:
                        logger.warning(f"✗ NOT FOUND: {injected.name}")

            except json.JSONDecodeError:
                logger.warning(f"Could not parse resolve extensions output")
        else:
            logger.warning(f"resolve extensions failed: {result.stderr.decode()}")

    def _cleanup_models(self) -> None:
        for f in self._injected_files:
            if f.exists():
                f.unlink()
                logger.info(f"Cleaned up: {f}")
        self._injected_files.clear()

    def _load_project_build_config(self, project_path: Path) -> dict | None:
        """Load project build configuration from centralized JSON file.
        
        Looks for ../../proj_build_command.json
        Structure: { "project_name": { "prepare_for_build": "command", "build_command": "command" } }
        Uses the last directory name from project_path to match the JSON field.
        """
        config_file = Path("../../proj_build_command.json")
        if not config_file.exists():
            logger.debug(f"Config file not found at {config_file}")
            return None
        
        try:
            with open(config_file, 'r') as f:
                all_configs = json.load(f)
            
            # Use the last directory name from the project path to match JSON field
            project_name = project_path.name
            
            if project_name in all_configs:
                config = all_configs[project_name]
                logger.info(f"Found build config for project '{project_name}' in {config_file}")
                return config
            
            logger.debug(f"No build config found for project '{project_name}' in {config_file}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {config_file}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error reading {config_file}: {e}")
            return None
    
    def _run_pre_build_commands(self, project_path: Path, commands_config: list | str) -> bool:
        """Run pre-build commands before creating CodeQL database.
        
        The commands are executed inside the project directory.
        Supports both old format (string) and new format (list of objects with can_error flag).
        """
        if not commands_config:
            return True
        
        # Verify project directory exists
        if not project_path.exists():
            logger.error(f"Project directory does not exist: {project_path}")
            return False
        
        # Handle new format: list of objects with command and can_error
        if isinstance(commands_config, list):
            commands = []
            for item in commands_config:
                if isinstance(item, dict):
                    cmd = item.get("command", "")
                    can_error = item.get("can_error", False)
                    commands.append({"command": cmd, "can_error": can_error})
                elif isinstance(item, str):
                    # Fallback: treat string as command that can error
                    commands.append({"command": item, "can_error": True})
        # Handle old format: string (split by &&)
        elif isinstance(commands_config, str):
            cmd_str = commands_config.strip()
            if not cmd_str:
                return True
            # Split by && and treat all as can_error=True for backward compatibility
            commands = [{"command": cmd.strip(), "can_error": True} 
                       for cmd in cmd_str.split('&&')]
        else:
            logger.warning(f"Unknown prepare_for_build format: {type(commands_config)}")
            return True
        
        if not commands:
            return True
        
        logger.info(f"Running {len(commands)} pre-build command(s) in {project_path}")
        
        for i, cmd_info in enumerate(commands, 1):
            cmd = cmd_info["command"]
            can_error = cmd_info["can_error"]
            
            if not cmd or not cmd.strip():
                continue
            
            logger.info(f"Running command {i}/{len(commands)}: {cmd} (can_error={can_error})")
            
            # Use shell=True to support commands with operators
            # cwd ensures the command runs inside the project directory
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(project_path),
                timeout=self.timeout,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                if can_error:
                    # Log as warning but note that failure is expected/okay for this command
                    logger.warning(f"Command {i} failed (exit code {result.returncode}): {cmd} - This is expected and okay, continuing...")
                    if result.stderr:
                        stderr_preview = result.stderr.strip()[:200]
                        logger.warning(f"  Error (expected): {stderr_preview}")
                        if len(result.stderr.strip()) > 200:
                            logger.debug(f"  Full stderr:\n{result.stderr}")
                    if result.stdout:
                        logger.debug(f"  stdout:\n{result.stdout}")
                else:
                    # Command that cannot error has failed - abort
                    logger.error(f"Command {i} failed (exit code {result.returncode}): {cmd}")
                    if result.stderr:
                        logger.error(f"  Error:\n{result.stderr}")
                    if result.stdout:
                        logger.error(f"  Output:\n{result.stdout}")
                    return False
            else:
                logger.info(f"Command {i} completed successfully")
                if result.stdout:
                    logger.debug(f"  Output:\n{result.stdout}")
        
        logger.info("All pre-build commands processed")
        return True

    def _create_database(self, project_path: Path, db_path: Path) -> bool:
        """Create CodeQL database."""
        logger.info(f"Project: {project_path}")

        # Load project-specific build configuration from centralized file
        config = self._load_project_build_config(project_path)
        if config:
            # Run pre-build commands if specified
            pre_build_cmd = config.get("prepare_for_build", "")
            if pre_build_cmd:
                if not self._run_pre_build_commands(project_path, pre_build_cmd):
                    logger.error("Pre-build command failed, aborting database creation")
                    return False

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
            # Check if config overrides the build command
            build_cmd = None
            if config and "build_command" in config:
                build_cmd_config = config["build_command"]
                # Handle new format: object with command field
                if isinstance(build_cmd_config, dict):
                    build_cmd = build_cmd_config.get("command", "")
                # Handle old format: string
                elif isinstance(build_cmd_config, str):
                    build_cmd = build_cmd_config
                
                if build_cmd:
                    logger.info(f"Using build command from config: {build_cmd}")
            
            # Otherwise, auto-detect build command
            if not build_cmd:
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
                "--command", build_cmd,
            ]

        result = subprocess.run(
            cmd, timeout=self.timeout, capture_output=True, cwd=str(project_path)
        )

        if result.returncode != 0:
            stderr_output = result.stderr.decode()
            stdout_output = result.stdout.decode()
            logger.error(f"Database creation failed:")
            if stderr_output:
                logger.error(f"stderr:\n{stderr_output}")
            if stdout_output:
                logger.error(f"stdout:\n{stdout_output}")
            return False

        logger.info("Database created successfully")
        
        # Finalize the database before running queries
        logger.info("Finalizing database...")
        if not self._finalize_database(db_path):
            logger.error("Database finalization failed")
            return False
        
        return True
    
    def _ensure_database_finalized(self, db_path: Path) -> bool:
        """Check if database is finalized, and finalize if needed."""
        try:
            db_metadata = db_path / "codeql-database.yml"
            if db_metadata.exists():
                with open(db_metadata, 'r') as f:
                    db_info = yaml.safe_load(f)
                if db_info.get('finalised', False):
                    logger.info("Database is already finalized")
                    return True
        except Exception as e:
            logger.debug(f"Could not check finalization status: {e}")
        
        # Database is not finalized, finalize it
        return self._finalize_database(db_path)
    
    def _finalize_database(self, db_path: Path) -> bool:
        """Finalize a CodeQL database so it can be used for queries."""
        # Check if already finalized first
        try:
            db_metadata = db_path / "codeql-database.yml"
            if db_metadata.exists():
                with open(db_metadata, 'r') as f:
                    db_info = yaml.safe_load(f)
                if db_info.get('finalised', False):
                    logger.info("Database is already finalized")
                    return True
        except Exception:
            pass  # Continue to try finalization
        
        cmd = [
            self.binary, "database", "finalize",
            str(db_path),
        ]
        
        result = subprocess.run(
            cmd, timeout=self.timeout, capture_output=True, text=True
        )
        
        if result.returncode != 0:
            # Check if error is because it's already finalized
            error_msg = (result.stderr + result.stdout).lower()
            if "already finalized" in error_msg or "is already finalized" in error_msg:
                logger.info("Database is already finalized (detected from error message)")
                return True
            
            logger.error(f"Database finalization failed:")
            if result.stderr:
                logger.error(f"stderr:\n{result.stderr}")
            if result.stdout:
                logger.error(f"stdout:\n{result.stdout}")
            return False
        
        logger.info("Database finalized successfully")
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

    def _run_memory_queries(self, db_path: Path, results_path: Path) -> bool:
        memory_queries = [
            # Core memory safety queries
            "codeql/cpp-queries:Critical/MemoryNeverFreed.ql",
            "codeql/cpp-queries:Critical/MemoryMayNotBeFreed.ql",
            "codeql/cpp-queries:Critical/DoubleFree.ql",
            "codeql/cpp-queries:Critical/UseAfterFree.ql",
        ]

        cmd = [
            self.binary, "database", "analyze",
            str(db_path),
            "--format=sarif-latest",
            f"--output={results_path}",
            "--download",
            "-v",
        ] + memory_queries

        logger.info(f"Running CodeQL analyze...")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, timeout=self.timeout, capture_output=True)

        stdout = result.stdout.decode()
        stderr = result.stderr.decode()

        logger.info(f"CodeQL analyze return code: {result.returncode}")
        if stdout:
            logger.debug(f"stdout:\n{stdout}")
        if stderr:
            logger.info(f"stderr:\n{stderr}")

        return results_path.exists()

    def _find_function_at_line(self, file_path: str, line_number: int, project_path: Path) -> str:
        """Find the function name that contains the given line number."""
        if not file_path or not line_number:
            return ""
        
        try:
            # Resolve file path relative to project
            if not Path(file_path).is_absolute():
                full_path = project_path / file_path
            else:
                full_path = Path(file_path)
            
            if not full_path.exists():
                return ""
            
            # Parse the file to get functions
            from src.tree_sitter_parser import CodeParser
            parser = CodeParser()
            functions = parser.parse_file(full_path)
            
            # Find function containing this line
            for func_name, func_info in functions.items():
                if func_info.start_line <= line_number <= func_info.end_line:
                    return func_name
            
        except Exception as e:
            logger.debug(f"Could not find function at line {line_number} in {file_path}: {e}")
        
        return ""

    def _parse_sarif(self, sarif_path: Path, project_path: Path = None) -> list[Warning]:
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
                    file_path = loc.get("artifactLocation", {}).get("uri", "")
                    line_number = loc.get("region", {}).get("startLine", 0)
                    
                    # Find function name by parsing the source file
                    function_name = ""
                    if project_path and file_path and line_number:
                        function_name = self._find_function_at_line(file_path, line_number, project_path)

                    warnings.append(Warning(
                        file_path=file_path,
                        line_number=line_number,
                        function_name=function_name,
                        warning_type=rule_id,
                        message=result.get("message", {}).get("text", ""),
                        issue_type=issue_type,
                        trace=[],
                    ))

            logger.info(f"Parsed {len(warnings)} warnings")

        except Exception as e:
            logger.error(f"SARIF parse error: {e}")

        return warnings