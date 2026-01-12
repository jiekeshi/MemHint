"""Code merger with cross-file dependency slicing.

Merges relevant code fragments into a single compilable file by:
1. Recursively tracking function call dependencies (callees)
2. Extracting type definitions (structs, typedefs, enums)
3. Collecting necessary includes and macros
4. Using LLM to fix remaining compilation issues

The result maps back to original file locations for accurate bug reporting.

TODO: I am not sure if this is in correct format.
"""

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser
    C_LANGUAGE = Language(tsc.language())
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OriginalLocation:
    """Original location in source file."""
    file_path: str
    start_line: int
    end_line: int
    function_name: str = ""


@dataclass
class MergedCode:
    """Result of merging code."""
    code: str
    location_map: dict[int, OriginalLocation] = field(default_factory=dict)
    included_functions: list[str] = field(default_factory=list)
    included_types: list[str] = field(default_factory=list)


@dataclass
class TypeDef:
    """A type definition (struct, typedef, enum)."""
    name: str
    code: str
    file_path: str
    line: int
    dependencies: set = field(default_factory=set)  # Other types this depends on


class DependencyTracker:
    """Track cross-file dependencies using tree-sitter."""

    def __init__(self):
        if TS_AVAILABLE:
            self.parser = Parser(C_LANGUAGE)
        else:
            self.parser = None

    def get_all_callees(self, func_name: str, functions: dict, max_depth: int = 10) -> set[str]:
        """Recursively get all callees of a function."""
        all_callees = set()
        visited = set()

        def recurse(name: str, depth: int):
            if depth > max_depth or name in visited:
                return
            visited.add(name)

            if name in functions:
                callees = functions[name].callees
                all_callees.update(callees)
                for callee in callees:
                    recurse(callee, depth + 1)

        recurse(func_name, 0)
        return all_callees

    def extract_type_dependencies(self, code: str) -> set[str]:
        """Extract type names used in code."""
        types = set()

        try:
            tree = self.parser.parse(code.encode())
            source = code.encode()

            for node in self._walk(tree.root_node):
                if node.type == "type_identifier":
                    types.add(source[node.start_byte:node.end_byte].decode())
                elif node.type == "primitive_type":
                    pass  # Skip int, char, etc.
                elif node.type in ("struct_specifier", "enum_specifier", "union_specifier"):
                    # Get the name
                    for child in node.children:
                        if child.type == "type_identifier":
                            types.add(source[child.start_byte:child.end_byte].decode())
        except Exception as e:
            logger.debug(f"Tree-sitter parsing failed: {e}")

        # Filter out standard types
        std_types = {'int', 'char', 'void', 'float', 'double', 'long', 'short',
                     'unsigned', 'signed', 'bool', 'size_t', 'ssize_t', 'NULL',
                     'FILE', 'EOF', 'true', 'false', 'uint8_t', 'uint16_t',
                     'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t'}
        return types - std_types

    def _walk(self, node):
        """Walk AST nodes."""
        yield node
        for child in node.children:
            yield from self._walk(child)


class TypeCollector:
    """Collect type definitions from source files."""

    def __init__(self):
        if TS_AVAILABLE:
            self.parser = Parser(C_LANGUAGE)
        else:
            self.parser = None
        self.tracker = DependencyTracker()

    def collect_from_file(self, file_path: Path) -> list[TypeDef]:
        """Extract all type definitions from a file."""
        try:
            code = file_path.read_text()
        except Exception as e:
            logger.debug(f"Cannot read {file_path}: {e}")
            return []

        types = []

        # Extract struct definitions
        for match in re.finditer(
            r'(typedef\s+)?(struct|union|enum)\s+(\w+)?\s*\{([^}]*)\}\s*(\w*)\s*;',
            code, re.DOTALL
        ):
            is_typedef = match.group(1) is not None
            kind = match.group(2)
            tag_name = match.group(3)
            body = match.group(4)
            alias_name = match.group(5)

            name = alias_name if alias_name else tag_name
            if not name:
                continue

            full_code = match.group(0)
            line = code[:match.start()].count('\n') + 1

            # Find type dependencies in body
            deps = self.tracker.extract_type_dependencies(body)

            types.append(TypeDef(
                name=name,
                code=full_code,
                file_path=str(file_path),
                line=line,
                dependencies=deps
            ))

        # Extract simple typedefs (typedef existing_type new_name;)
        for match in re.finditer(
            r'typedef\s+(?!struct|union|enum)([^;]+)\s+(\w+)\s*;',
            code
        ):
            existing = match.group(1).strip()
            new_name = match.group(2)

            # Skip if already captured above
            if any(t.name == new_name for t in types):
                continue

            full_code = match.group(0)
            line = code[:match.start()].count('\n') + 1
            deps = self.tracker.extract_type_dependencies(existing)

            types.append(TypeDef(
                name=new_name,
                code=full_code,
                file_path=str(file_path),
                line=line,
                dependencies=deps
            ))

        return types

    def collect_from_project(self, project_path: Path) -> dict[str, TypeDef]:
        """Collect all types from project headers and source files."""
        all_types = {}

        # Search headers first (higher priority)
        for h_file in project_path.glob("**/*.h"):
            for typedef in self.collect_from_file(h_file):
                if typedef.name not in all_types:
                    all_types[typedef.name] = typedef

        # Then source files
        for c_file in project_path.glob("**/*.c"):
            for typedef in self.collect_from_file(c_file):
                if typedef.name not in all_types:
                    all_types[typedef.name] = typedef

        return all_types

    def resolve_type_dependencies(
        self,
        needed_types: set[str],
        all_types: dict[str, TypeDef],
        max_depth: int = 10
    ) -> list[TypeDef]:
        """Resolve all type dependencies recursively, return in correct order."""
        resolved = []
        visited = set()

        def resolve(type_name: str, depth: int):
            if depth > max_depth or type_name in visited:
                return
            if type_name not in all_types:
                return

            visited.add(type_name)
            typedef = all_types[type_name]

            # First resolve dependencies
            for dep in typedef.dependencies:
                resolve(dep, depth + 1)

            resolved.append(typedef)

        for type_name in needed_types:
            resolve(type_name, 0)

        return resolved


class CodeMerger:
    """Merge relevant code into single compilable file with dependency tracking."""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.dep_tracker = DependencyTracker()
        self.type_collector = TypeCollector()

    def merge(
        self,
        annotations,  # AnnotationSet
        functions: dict,  # func_name -> FunctionInfo
        project_path: Path,
    ) -> MergedCode:
        """Merge relevant code with full dependency tracking.

        Args:
            annotations: Validated annotations
            functions: All parsed functions
            project_path: Original project path

        Returns:
            MergedCode with compilable code and location mapping
        """
        project_path = Path(project_path)

        # Step 1: Determine target functions (recursive callees)
        target_functions = self._get_target_functions(annotations, functions)
        logger.info(f"Target functions ({len(target_functions)}): {list(target_functions)[:10]}...")

        # Step 2: Collect all types from project
        all_types = self.type_collector.collect_from_project(project_path)
        logger.info(f"Found {len(all_types)} type definitions in project")

        # Step 3: Determine needed types based on function code
        needed_types = set()
        for func_name in target_functions:
            if func_name in functions:
                func_types = self.dep_tracker.extract_type_dependencies(functions[func_name].code)
                needed_types.update(func_types)

        # Step 4: Resolve type dependencies
        resolved_types = self.type_collector.resolve_type_dependencies(needed_types, all_types)
        logger.info(f"Resolved {len(resolved_types)} types needed for compilation")

        # Step 5: Collect includes
        includes = self._collect_includes(target_functions, functions, project_path)

        # Step 6: Generate merged code
        merged = self._generate_merged_code(
            target_functions, functions, resolved_types, includes, annotations
        )

        # Step 7: Use LLM to fix remaining compilation issues
        if self.llm:
            merged = self._llm_fix_code(merged)

        return merged

    def _get_target_functions(self, annotations, functions: dict) -> set[str]:
        """Get all functions to include with recursive callee tracking."""
        targets = set()

        # Start with annotated functions
        for func_name, anns in annotations.annotations.items():
            # Include if has function-level annotation or is known function
            if func_name in functions:
                targets.add(func_name)

        # Recursively add all callees
        all_with_callees = set(targets)
        for func_name in targets:
            callees = self.dep_tracker.get_all_callees(func_name, functions)
            all_with_callees.update(callees)

        # Always include main
        if "main" in functions:
            all_with_callees.add("main")

        # Filter to existing functions
        return {f for f in all_with_callees if f in functions}

    def _collect_includes(
        self,
        target_functions: set[str],
        functions: dict,
        project_path: Path
    ) -> set[str]:
        """Collect necessary includes from source files."""
        includes = set()
        seen_files = set()

        for func_name in target_functions:
            if func_name not in functions:
                continue

            file_path = functions[func_name].file_path
            if not file_path or file_path in seen_files:
                continue
            seen_files.add(file_path)

            try:
                code = Path(file_path).read_text()
                for line in code.split('\n'):
                    line = line.strip()
                    if line.startswith('#include'):
                        # Only keep standard library includes
                        if '<' in line and '>' in line:
                            includes.add(line)
            except Exception:
                pass

        return includes

    def _generate_merged_code(
        self,
        target_functions: set[str],
        functions: dict,
        resolved_types: list[TypeDef],
        includes: set[str],
        annotations
    ) -> MergedCode:
        """Generate the merged C code with location mapping."""
        lines = []
        location_map = {}
        included_functions = []
        included_types = []
        current_line = 1

        # === Header ===
        header = [
            "/* " + "=" * 60 + " */",
            "/* HINT Auto-Merged Analysis File                            */",
            "/* Generated for single-file static analysis                 */",
            "/* DO NOT EDIT - Line mappings preserved for result tracking */",
            "/* " + "=" * 60 + " */",
            "",
        ]
        lines.extend(header)
        current_line += len(header)

        # === Standard Includes ===
        std_includes = [
            "#include <stdlib.h>",
            "#include <string.h>",
            "#include <stdio.h>",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#include <stddef.h>",
        ]
        lines.extend(std_includes)
        lines.append("")
        current_line += len(std_includes) + 1

        # === Project Includes (standard library only) ===
        if includes:
            lines.append("/* Project includes */")
            current_line += 1
            for inc in sorted(includes):
                if inc not in "\n".join(std_includes):
                    lines.append(inc)
                    current_line += 1
            lines.append("")
            current_line += 1

        # === Annotations as comments ===
        lines.append("/* === HINT Annotations === */")
        current_line += 1
        for func_name, anns in annotations.annotations.items():
            for ann in anns:
                if not ann.is_bug_annotation():
                    lines.append(f"/* @{ann.annotation_type.name}({func_name}, {ann.target}) */")
                    current_line += 1
        lines.append("")
        current_line += 1

        # === Type Definitions (in dependency order) ===
        if resolved_types:
            lines.append("/* === Type Definitions === */")
            current_line += 1

            for typedef in resolved_types:
                lines.append(f"/* From: {typedef.file_path}:{typedef.line} */")
                current_line += 1

                type_lines = typedef.code.split('\n')
                for i, line in enumerate(type_lines):
                    lines.append(line)
                    location_map[current_line] = OriginalLocation(
                        file_path=typedef.file_path,
                        start_line=typedef.line + i,
                        end_line=typedef.line + i,
                    )
                    current_line += 1

                lines.append("")
                current_line += 1
                included_types.append(typedef.name)

        # === Forward Declarations ===
        lines.append("/* === Forward Declarations === */")
        current_line += 1
        for func_name in sorted(target_functions):
            if func_name in functions and func_name != "main":
                # Generate simple forward declaration
                func = functions[func_name]
                # Extract return type and params from signature
                sig = self._extract_signature(func.code, func_name)
                if sig:
                    lines.append(f"{sig};")
                    current_line += 1
        lines.append("")
        current_line += 1

        # === Function Definitions ===
        lines.append("/* === Function Definitions === */")
        current_line += 1

        # Sort: main last
        sorted_funcs = sorted(target_functions, key=lambda f: (f == "main", f))

        for func_name in sorted_funcs:
            if func_name not in functions:
                continue

            func = functions[func_name]

            lines.append(f"/* === From: {func.file_path}:{func.start_line} === */")
            current_line += 1

            func_lines = func.code.split('\n')
            for i, line in enumerate(func_lines):
                lines.append(line)
                location_map[current_line] = OriginalLocation(
                    file_path=func.file_path,
                    start_line=func.start_line + i,
                    end_line=func.start_line + i,
                    function_name=func_name,
                )
                current_line += 1

            lines.append("")
            current_line += 1
            included_functions.append(func_name)

        return MergedCode(
            code="\n".join(lines),
            location_map=location_map,
            included_functions=included_functions,
            included_types=included_types,
        )

    def _extract_signature(self, code: str, func_name: str) -> Optional[str]:
        """Extract function signature from definition."""
        # Match: return_type func_name(params)
        pattern = rf'([a-zA-Z_][a-zA-Z0-9_\s\*]*)\s+(\*?\s*{func_name})\s*\(([^)]*)\)'
        match = re.search(pattern, code)
        if match:
            ret_type = match.group(1).strip()
            name = match.group(2).strip()
            params = match.group(3).strip()
            if not params:
                params = "void"
            return f"{ret_type} {name}({params})"
        return None

    def _llm_fix_code(self, merged: MergedCode) -> MergedCode:
        """Use LLM to fix compilation issues."""
        if not self.llm:
            return merged

        prompt = f"""Fix this auto-merged C code to make it compilable.

IMPORTANT RULES:
1. Add MINIMAL stub definitions for missing types:
   `typedef struct {{ void* _opaque; }} MissingType;`
2. Add MINIMAL stub declarations for missing functions:
   `extern void* missing_func(void);`
3. DO NOT modify existing function bodies or logic
4. DO NOT remove any code or comments
5. PRESERVE all "/* From: ... */" location comments exactly
6. PRESERVE all "/* @ANNOTATION */" comments exactly
7. If code looks compilable, return UNCHANGED

```c
{merged.code[:15000]}
```

Return ONLY the fixed C code. No explanations or markdown."""

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
            )
            fixed_code = response.choices[0].message.content

            # Extract code from markdown if present
            if "```c" in fixed_code:
                fixed_code = fixed_code.split("```c", 1)[1].split("```", 1)[0]
            elif "```" in fixed_code:
                parts = fixed_code.split("```")
                if len(parts) >= 2:
                    fixed_code = parts[1]

            fixed_code = fixed_code.strip()

            # Sanity check
            if len(fixed_code) < len(merged.code) * 0.3:
                logger.warning("LLM returned suspiciously short code, using original")
                return merged

            logger.info("LLM patched merged code for compilation")
            return MergedCode(
                code=fixed_code,
                location_map=merged.location_map,
                included_functions=merged.included_functions,
                included_types=merged.included_types,
            )

        except Exception as e:
            logger.warning(f"LLM fix failed: {e}")
            return merged

    @staticmethod
    def map_result_to_original(
        merged_line: int,
        location_map: dict[int, OriginalLocation]
    ) -> Optional[OriginalLocation]:
        """Map a line in merged code back to original location."""
        # Direct match
        if merged_line in location_map:
            return location_map[merged_line]

        # Find closest preceding mapped line
        mapped_lines = sorted(location_map.keys())
        for ml in reversed(mapped_lines):
            if ml <= merged_line:
                loc = location_map[ml]
                offset = merged_line - ml
                return OriginalLocation(
                    file_path=loc.file_path,
                    start_line=loc.start_line + offset,
                    end_line=loc.end_line + offset,
                    function_name=loc.function_name,
                )

        return None