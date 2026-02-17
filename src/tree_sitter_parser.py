"""Robust C/C++ function extractor using tree-sitter.

This module extracts function information from C/C++ source files using tree-sitter.
It handles complex C++ types including templates, references, qualified names, etc.

Key design: Extract raw text from AST nodes instead of reconstructing types.
"""

import logging
import re
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy initialization of parsers
_c_parser = None
_cpp_parser = None


def _get_c_parser():
    global _c_parser
    if _c_parser is None:
        import tree_sitter_c as tsc
        from tree_sitter import Language, Parser
        _c_parser = Parser(Language(tsc.language()))
    return _c_parser


def _get_cpp_parser():
    global _cpp_parser
    if _cpp_parser is None:
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser
        _cpp_parser = Parser(Language(tscpp.language()))
    return _cpp_parser


@dataclass
class FunctionInfo:
    """Function information extracted from source code."""
    name: str
    code: str
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    return_type: str = ""
    arg_names: list[str] = field(default_factory=list)
    arg_types: list[str] = field(default_factory=list)
    callees: set[str] = field(default_factory=set)
    callers: set[str] = field(default_factory=set)
    return_expressions: set[str] = field(default_factory=set)


@dataclass
class MacroInfo:
    """Macro definition information extracted from source code."""
    name: str
    code: str
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    arg_names: list[str] = field(default_factory=list)
    expansion: str = ""  # The expansion text (right side of #define)
    is_function_like: bool = False  # True if macro has parameters like MACRO(x, y)


class CodeParser:
    """Extract functions from C/C++ source using tree-sitter."""

    def __init__(self):
        # Aggregated across the last parse_project / parse_source calls.
        # Contains typedef alias names that are pointer types.
        # Example: `typedef struct _client { ... } *client;` -> adds "client"
        self.pointer_typedef_aliases: set[str] = set()
    
    def _collect_pointer_typedef_aliases(self, root, source: bytes) -> set[str]:
        """Collect pointer typedef aliases using tree-sitter AST nodes.

        Supports:
        - typedef struct _client { ... } *client;
        - typedef some_type *client;
        - typedef char* sds;              (IMPORTANT: '*' on type side, not declarator side)
        - typedef const char *cstring;
        - typedef struct {} pointer1, *pointer2;  (multiple aliases in one typedef)
        - avoids false capture from function pointer typedef parameter names
        """

        def _unwrap_typedef_declarator(decl_node):
            if decl_node is None:
                return None
            if decl_node.type in ("parenthesized_declarator", "function_declarator"):
                inner = decl_node.child_by_field_name("declarator")
                return _unwrap_typedef_declarator(inner) if inner is not None else decl_node
            return decl_node

        def _iter_typedef_alias_declarators(type_def_node):
            """
            Yield all alias declarator nodes in this typedef:
            - the main 'declarator' field (often the first alias)
            - every init_declarator's declarator (covers comma-separated aliases)
            """
            main_decl = _unwrap_typedef_declarator(type_def_node.child_by_field_name("declarator"))
            if main_decl is not None:
                yield main_decl

            # Case: typedef struct {} a, *b;
            for child in type_def_node.children:
                if child.type == "init_declarator":
                    alias_decl = _unwrap_typedef_declarator(child.child_by_field_name("declarator"))
                    if alias_decl is not None:
                        yield alias_decl

        aliases: set[str] = set()

        for node in self._iter_nodes(root):
            if node.type != "type_definition":
                continue

            # Collect all alias declarators in this typedef
            alias_decls = list(_iter_typedef_alias_declarators(node))
            if not alias_decls:
                continue

            # Compute base-type text ONCE: from typedef start to first declarator start.
            # This avoids pollution by later comma-separated declarators (e.g., "*pointer2").
            first_decl_start = min(d.start_byte for d in alias_decls)
            base_type_text = source[node.start_byte:first_decl_start].decode(errors="ignore")

            # For each alias, decide pointer-typedef or not
            for decl in alias_decls:
                alias_name = self._extract_declarator_name_only(decl, source)
                if not alias_name:
                    continue

                # (1) declarator-side pointer: definitely a pointer typedef
                if decl.type == "pointer_declarator":
                    aliases.add(alias_name)
                    continue

                # (2) type-side pointer: typedef char* sds; typedef const char *cstring;
                #     if base type contains '*', treat alias as pointer-like typedef
                if "*" in base_type_text:
                    aliases.add(alias_name)
                    continue

        if logger.isEnabledFor(logging.DEBUG) and aliases:
            logger.debug(f"Collected {len(aliases)} pointer typedef aliases: {sorted(aliases)}")

        return aliases

    
    def _preprocess_file(self, file_path: Path) -> bytes:
        """Preprocess a file to expand macros.
        
        Uses gcc or clang to preprocess the file. Tries clang first, then gcc.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Preprocessed source code as bytes
            
        Raises:
            RuntimeError: If preprocessing fails or no compiler is found
        """
        # Determine if it's C++ based on extension
        cpp_extensions = {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".C", ".CPP"}
        is_cpp = file_path.suffix in cpp_extensions
        
        # Try to find a compiler
        compiler = None
        for cmd in ["clang", "gcc"]:
            if shutil.which(cmd):
                compiler = cmd
                break
        
        if not compiler:
            raise RuntimeError("No C/C++ compiler found (clang or gcc required for preprocessing)")
        
        # Build compiler command
        cmd = [compiler, "-E"]
        if is_cpp:
            cmd.append("-x")
            cmd.append("c++")
        else:
            cmd.append("-x")
            cmd.append("c")
        
        # Add standard includes if needed (optional, helps with system headers)
        # cmd.extend(["-I", "/usr/include", "-I", "/usr/local/include"])
        
        cmd.append(str(file_path))
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                timeout=30,  # 30 second timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Preprocessing failed: {error_msg}")
            
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Preprocessing timed out for {file_path}")
        except Exception as e:
            raise RuntimeError(f"Preprocessing error: {e}")

    def parse_file(
        self,
        file_path: Path,
        preprocess: bool = False,
        collect_typedefs: bool = True,
    ) -> dict[str, FunctionInfo]:
        """Parse a single file and extract all functions.

        Args:
            file_path: Path to the source file
            preprocess: If True, preprocess the file first to expand macros
            collect_typedefs: If True, collect pointer typedef aliases from this file
                            (set False in parse_project Pass2 to avoid rescanning)
        """
        try:
            if preprocess:
                source = self._preprocess_file(file_path)
            else:
                source = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Cannot read/preprocess {file_path}: {e}")
            return {}

        # Select parser based on extension
        cpp_extensions = {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".C", ".CPP"}
        c_only_extensions = {".c"}
        ambiguous_extensions = {".h", ".hh", ".inc"}  # Could be C or C++

        if file_path.suffix in cpp_extensions:
            parser = _get_cpp_parser()
        elif file_path.suffix in c_only_extensions:
            parser = _get_c_parser()
        elif file_path.suffix in ambiguous_extensions:
            # For .h files, try C++ parser first
            parser = _get_cpp_parser()
        else:
            parser = _get_c_parser()

        tree = parser.parse(source)

        # Collect pointer typedef aliases for this file (optional)
        if collect_typedefs:
            try:
                aliases = self._collect_pointer_typedef_aliases(tree.root_node, source)
                self.pointer_typedef_aliases.update(aliases)
            except Exception:
                pass

        functions: dict[str, FunctionInfo] = {}

        # Find all function_definition nodes
        for node in self._iter_nodes(tree.root_node):
            if node.type == "function_definition":
                func_info = self._extract_function(node, source, str(file_path))
                if func_info and func_info.name:
                    functions[func_info.name] = func_info

        return functions

    def parse_file_with_macros(
        self,
        file_path: Path,
        preprocess: bool = False,
        collect_typedefs: bool = True,
    ) -> tuple[dict[str, FunctionInfo], dict[str, MacroInfo]]:
        """Parse a single file and extract both functions and macros.

        Args:
            file_path: Path to the source file
            preprocess: If True, preprocess the file first to expand macros
                    Note: If preprocess=True, macros will already be expanded
                    and won't be extractable. Use preprocess=False to extract macros.
            collect_typedefs: If True, collect pointer typedef aliases from this file
                            and update self.pointer_typedef_aliases. For project-level
                            parsing with a global Pass1 typedef collection, set to False.

        Returns:
            Tuple of (functions dict, macros dict)
        """
        try:
            # IMPORTANT: if you want macros, you MUST read raw source (no preprocessing)
            # preprocessing expands macros and removes macro definitions.
            source = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return {}, {}

        # Select parser based on extension
        cpp_extensions = {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".C", ".CPP"}
        c_only_extensions = {".c"}
        ambiguous_extensions = {".h", ".hh", ".inc"}

        if file_path.suffix in cpp_extensions:
            parser = _get_cpp_parser()
        elif file_path.suffix in c_only_extensions:
            parser = _get_c_parser()
        elif file_path.suffix in ambiguous_extensions:
            parser = _get_cpp_parser()
        else:
            parser = _get_c_parser()

        tree = parser.parse(source)

        # Collect pointer typedef aliases (optional)
        if collect_typedefs:
            try:
                aliases = self._collect_pointer_typedef_aliases(tree.root_node, source)
                self.pointer_typedef_aliases.update(aliases)
            except Exception:
                pass

        functions: dict[str, FunctionInfo] = {}
        macros: dict[str, MacroInfo] = {}

        # Extract both functions and macros
        for node in self._iter_nodes(tree.root_node):
            if node.type == "function_definition":
                func_info = self._extract_function(node, source, str(file_path))
                if func_info and func_info.name:
                    functions[func_info.name] = func_info
            elif node.type in ("preproc_def", "preproc_function_def"):
                macro_info = self._extract_macro(node, source, str(file_path))
                if macro_info and macro_info.name:
                    macros[macro_info.name] = macro_info

        return functions, macros



    def parse_source(
        self,
        source: str | bytes,
        filename: str = "<string>",
        is_cpp: bool = True,
        collect_typedefs: bool = True,
    ) -> dict[str, FunctionInfo]:
        """Parse source code string and extract all functions."""
        if isinstance(source, str):
            source = source.encode("utf-8")

        parser = _get_cpp_parser() if is_cpp else _get_c_parser()
        tree = parser.parse(source)

        # Collect pointer typedef aliases for this source (optional)
        if collect_typedefs:
            try:
                aliases = self._collect_pointer_typedef_aliases(tree.root_node, source)
                self.pointer_typedef_aliases.update(aliases)
            except Exception:
                pass

        functions: dict[str, FunctionInfo] = {}

        for node in self._iter_nodes(tree.root_node):
            if node.type == "function_definition":
                func_info = self._extract_function(node, source, filename)
                if func_info and func_info.name:
                    functions[func_info.name] = func_info

        return functions


    def parse_project(self, project_path: Path, preprocess: bool = False) -> dict[str, FunctionInfo]:
        functions: dict[str, FunctionInfo] = {}
        self.pointer_typedef_aliases = set()
        extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}

        project_path = project_path.resolve()

        # Pass 1: collect pointer typedef aliases globally
        for file_path in project_path.rglob("*"):
            if file_path.suffix.lower() in extensions:
                try:
                    source = file_path.read_bytes()
                except Exception:
                    continue
                parser = _get_cpp_parser() if file_path.suffix.lower() in {".cpp",".cc",".cxx",".hpp",".hxx"} else _get_c_parser()
                tree = parser.parse(source)
                try:
                    self.pointer_typedef_aliases.update(
                        self._collect_pointer_typedef_aliases(tree.root_node, source)
                    )
                except Exception:
                    pass

        # Pass 2: extract functions (no pointer filtering here; filtering is done uniformly in Phase 2a)
        for file_path in project_path.rglob("*"):
            if file_path.suffix.lower() in extensions:
                try:
                    rel_path = str(file_path.relative_to(project_path))
                except ValueError:
                    rel_path = str(file_path)

                file_funcs = self.parse_file(
                    file_path,
                    preprocess=preprocess,
                    collect_typedefs=False,
                )
                for name, info in file_funcs.items():
                    info.file_path = rel_path
                    # Keep longer definition if duplicate
                    if name not in functions or len(info.code) > len(functions[name].code):
                        functions[name] = info

        # Resolve caller/callee relationships
        self._resolve_calls(functions)
        logger.info(f"Parsed {len(functions)} functions from {project_path}")
        return functions

    def _iter_nodes(self, node):
        """Iterate all nodes in the AST without recursive calls.

        Tree-sitter trees can be very deep; using recursion risks hitting
        Python's recursion limit. This implementation uses an explicit stack
        and also guards against accidental cycles.
        """
        stack = [node]
        seen_ids: set[int] = set()

        while stack:
            current = stack.pop()
            node_id = id(current)
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)

            yield current

            # Preserve original traversal order: first child first.
            if getattr(current, "children", None):
                for child in reversed(current.children):
                    stack.append(child)

    def _extract_function(self, node, source: bytes, file_path: str) -> Optional[FunctionInfo]:
        """Extract FunctionInfo from a function_definition node, with C++ trailing return type support."""
        declarator_node = None
        body_node = None

        children = list(node.children)
        declarator_idx = -1

        for i, child in enumerate(children):
            if child.type == "compound_statement":
                body_node = child
            elif child.type in ("function_declarator", "pointer_declarator", "reference_declarator", "parenthesized_declarator"):
                declarator_node = child
                declarator_idx = i
            elif child.type == "ERROR":
                continue

        if not declarator_node:
            return None

        # Base return type: everything before declarator
        return_type = ""
        if declarator_idx > 0:
            type_start = children[0].start_byte
            type_end = declarator_node.start_byte
            return_type = source[type_start:type_end].decode(errors="ignore").strip()

        # Add pointer/reference from declarator wrapper (e.g., void* foo())
        ptr_ref_suffix = self._get_ptr_ref_from_declarator(declarator_node)
        if ptr_ref_suffix:
            return_type = return_type + ptr_ref_suffix

        # C++ trailing return type: auto f(...) -> T*
        # tree-sitter-cpp typically uses node type "trailing_return_type"
        for ch in children:
            if ch.type == "trailing_return_type":
                tr = source[ch.start_byte:ch.end_byte].decode(errors="ignore").strip()
                if tr.startswith("->"):
                    tr = tr[2:].strip()
                if tr:
                    return_type = tr
                break

        name, arg_names, arg_types = self._parse_declarator(declarator_node, source, node, source)
        if not name:
            return None

        callees = set()
        return_exprs = set()

        if body_node:
            for n in self._iter_nodes(body_node):
                if n.type == "call_expression":
                    callee = self._extract_callee_name(n, source)
                    if callee:
                        callees.add(callee)
                elif n.type == "return_statement":
                    expr = self._extract_return_expr(n, source)
                    if expr:
                        return_exprs.add(expr)

        code = source[node.start_byte:node.end_byte].decode(errors="ignore")

        return FunctionInfo(
            name=name,
            code=code,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            return_type=return_type,
            arg_names=arg_names,
            arg_types=arg_types,
            callees=callees,
            return_expressions=return_exprs,
        )


    def _parse_declarator(self, node, source: bytes, function_node=None, full_source: bytes=None) -> tuple[str, list[str], list[str]]:
        """Parse declarator to extract function name and parameters.

        Handles:
        - function_declarator: foo(int x, char* y)
        - pointer_declarator: *foo(int x) for pointer return
        - reference_declarator: &foo() for reference return (rare)
        
        Args:
            node: The declarator node
            source: Source bytes for the declarator region
            function_node: The full function_definition node (for getting class context)
            full_source: Full source bytes (for getting class context)
        """
        # Unwrap pointer/reference declarators to find function_declarator
        func_decl = self._find_function_declarator(node)
        if not func_decl:
            return "", [], []

        # Extract function name (first identifier in function_declarator)
        name = ""
        is_field_identifier = False
        for child in func_decl.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte].decode()
                break
            elif child.type in ("field_identifier", "destructor_name"):
                # For C++ member functions defined inside class, field_identifier only contains method name
                # We need to check if this is inside a class and get the class name
                name = source[child.start_byte:child.end_byte].decode()
                is_field_identifier = True
                break
            elif child.type == "scoped_identifier":
                # Class::method - extract full qualified name (Class::method)
                name = source[child.start_byte:child.end_byte].decode()
                break

        # If we got a field_identifier (class member function), try to get the class name
        if is_field_identifier and function_node is not None and full_source is not None:
            class_name = self._get_containing_class_name(function_node, full_source)
            if class_name:
                # Prepend class name to create qualified name: Class::method
                name = f"{class_name}::{name}"

        # Extract parameters from parameter_list
        arg_names = []
        arg_types = []

        for child in func_decl.children:
            if child.type == "parameter_list":
                for param in child.children:
                    if param.type in ("parameter_declaration", "optional_parameter_declaration"):
                        ptype, pname = self._parse_parameter(param, source)
                        arg_types.append(ptype)
                        arg_names.append(pname)
                    elif param.type == "variadic_parameter_declaration":
                        arg_types.append("...")
                        arg_names.append("...")
                    elif param.type == "variadic_parameter":
                        arg_types.append("...")
                        arg_names.append("...")
                break

        return name, arg_names, arg_types

    def _find_function_declarator(self, node):
        """Find the function_declarator node, unwrapping pointer/reference declarators."""
        if node.type == "function_declarator":
            return node

        # For pointer_declarator, reference_declarator, look inside
        for child in node.children:
            if child.type == "function_declarator":
                return child
            elif child.type in ("pointer_declarator", "reference_declarator",
                               "parenthesized_declarator"):
                result = self._find_function_declarator(child)
                if result:
                    return result
        return None

    def _get_ptr_ref_from_declarator(self, node) -> str:
        """Extract pointer (*) and reference (&) symbols from declarator wrapper.

        For `void* foo()`, the AST is:
            primitive_type: void
            pointer_declarator: * foo()
                *
                function_declarator: foo()

        We need to count the * and & between the outer declarator and function_declarator.
        """
        symbols = ""
        current = node

        while current.type in ("pointer_declarator", "reference_declarator"):
            if current.type == "pointer_declarator":
                symbols += "*"
            elif current.type == "reference_declarator":
                symbols += "&"

            # Find the next nested declarator
            found = False
            for child in current.children:
                if child.type in ("pointer_declarator", "reference_declarator",
                                 "function_declarator", "parenthesized_declarator"):
                    current = child
                    found = True
                    break
            if not found:
                break

        return symbols

    def _parse_parameter(self, node, source: bytes) -> tuple[str, str]:
        """Parse a parameter declaration to extract type and name.

        Strategy: The parameter name is typically the last identifier or inside
        a declarator (pointer_declarator, reference_declarator, array_declarator).
        The type is everything else before the name.
        """
        full_text = source[node.start_byte:node.end_byte].decode().strip()

        # Find the parameter name
        pname = ""
        name_node = None

        for child in node.children:
            if child.type == "identifier":
                # last identifier is usually the name (OK for most cases)
                pname = source[child.start_byte:child.end_byte].decode()
                name_node = child
            elif child.type in ("pointer_declarator", "reference_declarator", "array_declarator", "parenthesized_declarator", "function_declarator"):
                cand = self._extract_declarator_name_only(child, source)
                if cand:
                    pname = cand
                    name_node = child
                    break

        # Type is everything except the name
        if pname and name_node:
            # Get type by removing name from full text
            ptype = full_text
            # Handle case like "const char* name" vs "const char *name"
            if name_node.type == "identifier":
                # Simple case: type is before the identifier
                ptype = source[node.start_byte:name_node.start_byte].decode().strip()
            else:
                # Declarator case: need to extract type parts
                ptype = self._extract_type_from_param(node, name_node, source)
        else:
            # No name found (abstract parameter like "void*")
            ptype = full_text
            pname = ""

        # Clean up type
        ptype = " ".join(ptype.split())  # Normalize whitespace

        return ptype, pname


    def _extract_declarator_name_only(self, decl_node, source: bytes) -> str:
        """
        Extract the name of the *declarator itself* by following the 'declarator' chain.
        This avoids accidentally picking identifiers from parameter lists (function pointer typedef).
        """
        if decl_node is None:
            return ""

        cur = decl_node

        # Walk down the declarator chain: pointer_declarator / reference_declarator / parenthesized_declarator
        # / function_declarator / array_declarator / init_declarator ... until we hit an identifier-like node.
        while True:
            # Direct identifier nodes
            if cur.type in ("identifier", "type_identifier", "field_identifier"):
                return source[cur.start_byte:cur.end_byte].decode()

            # Common declarator containers expose a 'declarator' field
            next_decl = cur.child_by_field_name("declarator")
            if next_decl is not None:
                cur = next_decl
                continue

            # Some forms: function_declarator keeps name in its 'declarator' field,
            # but if not available, scan ONLY its non-parameter children for first identifier-like.
            if cur.type == "function_declarator":
                for ch in cur.children:
                    if ch.type == "parameter_list":
                        continue
                    if ch.type in ("identifier", "type_identifier", "field_identifier", "scoped_identifier"):
                        return source[ch.start_byte:ch.end_byte].decode()
                return ""

            # Fallback: do NOT DFS the whole subtree (that reintroduces the bug).
            return ""

    def _extract_type_from_param(self, param_node, name_node, source: bytes) -> str:
        """Extract the type portion from a parameter, excluding the name."""
        type_parts = []

        for child in param_node.children:
            if child == name_node:
                # Add pointer/reference symbols from the declarator
                if child.type == "pointer_declarator":
                    type_parts.append("*" * self._count_pointer_depth(child))
                elif child.type == "reference_declarator":
                    type_parts.append("&")
                elif child.type == "array_declarator":
                    type_parts.append("*")  # Arrays decay to pointers
                continue

            # Skip commas and other punctuation
            if child.type in (",", "(", ")"):
                continue

            text = source[child.start_byte:child.end_byte].decode().strip()
            if text:
                type_parts.append(text)

        return " ".join(type_parts)

    def _count_pointer_depth(self, node) -> int:
        """Count the pointer depth (number of *) in a pointer_declarator."""
        depth = 0
        current = node
        while current.type == "pointer_declarator":
            depth += 1
            found = False
            for child in current.children:
                if child.type == "pointer_declarator":
                    current = child
                    found = True
                    break
            if not found:
                break
        return depth

    def _extract_callee_name(self, call_node, source: bytes) -> str:
        """Extract the function name from a call_expression."""
        func = call_node.child_by_field_name("function")
        if not func:
            return ""

        text = source[func.start_byte:func.end_byte].decode()

        # Handle member access: obj.method() or ptr->method()
        if "->" in text:
            text = text.split("->")[-1]
        if "." in text:
            text = text.split(".")[-1]
        # Remove template arguments: func<T>()
        if "<" in text:
            text = text.split("<")[0]
        # Remove parentheses: (*func_ptr)()
        text = text.strip("()")

        return text.strip()

    def _extract_return_expr(self, return_node, source: bytes) -> str:
        """Extract the return expression from a return_statement."""
        for child in return_node.children:
            if child.type not in ("return", ";"):
                return source[child.start_byte:child.end_byte].decode().strip()
        return ""

    def _get_containing_class_name(self, node, source: bytes) -> Optional[str]:
        """Get the name of the class that contains this function definition.
        
        Walks up the AST from the function_definition node to find the containing
        class_specifier or struct_specifier.
        
        Args:
            node: The function_definition node
            source: Full source bytes
            
        Returns:
            Class name if found, None otherwise
        """
        current = node.parent
        while current is not None:
            if current.type in ("class_specifier", "struct_specifier"):
                # Find the class name (usually the first type_identifier child)
                for child in current.children:
                    if child.type == "type_identifier":
                        return source[child.start_byte:child.end_byte].decode()
                # If no type_identifier, might be anonymous or have a different structure
                break
            current = current.parent
        return None

    def _extract_macro(self, node, source: bytes, file_path: str) -> Optional[MacroInfo]:
        """Extract MacroInfo from a preproc_def or preproc_function_def node.
        
        AST structure:
        - preproc_def (object-like macro):
            - #define
            - identifier (macro name)
            - preproc_arg (optional expansion text)
        
        - preproc_function_def (function-like macro):
            - #define
            - identifier (macro name)
            - preproc_params (parameter list)
            - preproc_arg (expansion text)
        """
        children = list(node.children)
        if len(children) < 2:
            return None
        
        # Skip the first child if it's '#define'
        start_idx = 0
        if children[0].type == "#define":
            start_idx = 1
        
        if start_idx >= len(children):
            return None
        
        # Next child should be the macro name (identifier)
        name_node = children[start_idx]
        if name_node.type != "identifier":
            return None
        
        name = source[name_node.start_byte:name_node.end_byte].decode()
        
        # Check if it's a function-like macro (has preproc_params)
        is_function_like = node.type == "preproc_function_def"
        arg_names = []
        expansion = ""
        
        # Look for preproc_params (function-like macro parameters)
        params_node = None
        expansion_start_idx = start_idx + 1
        
        for i, child in enumerate(children[start_idx + 1:], start=start_idx + 1):
            if child.type == "preproc_params":
                is_function_like = True
                params_node = child
                expansion_start_idx = i + 1
                break
        
        # Extract parameters if function-like
        if params_node:
            for param_child in params_node.children:
                if param_child.type == "identifier":
                    param_name = source[param_child.start_byte:param_child.end_byte].decode()
                    arg_names.append(param_name)
        
        # Extract expansion text (preproc_arg or everything after name/params)
        if expansion_start_idx < len(children):
            expansion_node = children[expansion_start_idx]
            if expansion_node.type == "preproc_arg":
                expansion = source[expansion_node.start_byte:expansion_node.end_byte].decode().strip()
            else:
                # Fallback: get everything after name/params
                expansion_start = expansion_node.start_byte
                expansion_end = node.end_byte
                expansion = source[expansion_start:expansion_end].decode().strip()
            # Remove leading/trailing whitespace and newlines
            expansion = " ".join(expansion.split())
        
        # Full macro code
        code = source[node.start_byte:node.end_byte].decode()
        
        return MacroInfo(
            name=name,
            code=code,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            arg_names=arg_names,
            expansion=expansion,
            is_function_like=is_function_like,
        )

    def _resolve_calls(self, functions: dict[str, FunctionInfo]) -> None:
        """Resolve caller/callee relationships between functions."""
        func_names = set(functions.keys())

        for func in functions.values():
            # Filter callees to only include functions in our codebase
            func.callees = {c for c in func.callees if c in func_names}

        # Build callers from callees
        for name, func in functions.items():
            for callee in func.callees:
                if callee in functions:
                    functions[callee].callers.add(name)


# Convenience function
def extract_functions(source: str | bytes | Path, is_cpp: bool = True, preprocess: bool = False) -> dict[str, FunctionInfo]:
    """Extract functions from source code or file.

    Args:
        source: Source code string, bytes, or Path to file
        is_cpp: Whether to parse as C++ (default) or C
        preprocess: If True and source is a Path, preprocess the file first to expand macros

    Returns:
        Dict mapping function names to FunctionInfo objects
    """
    extractor = CodeParser()

    if isinstance(source, Path):
        return extractor.parse_file(source, preprocess=preprocess)
    else:
        return extractor.parse_source(source, is_cpp=is_cpp)


# =============================================================================
# Main - for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Parse file from command line
        file_path = Path(sys.argv[1])
        preprocess = "--preprocess" in sys.argv or "-p" in sys.argv
        extract_macros = "--macros" in sys.argv or "-m" in sys.argv
        
        extractor = CodeParser()
        
        if extract_macros:
            functions, macros = extractor.parse_file_with_macros(file_path, preprocess=False)
            print(f"Found {len(functions)} functions and {len(macros)} macros:\n")
            
            if macros:
                print("Macros:")
                for name, info in macros.items():
                    print(f"  {name}")
                    if info.is_function_like:
                        print(f"    Type: Function-like macro")
                        print(f"    Parameters: {', '.join(info.arg_names) if info.arg_names else 'none'}")
                    else:
                        print(f"    Type: Object-like macro")
                    print(f"    Expansion: {info.expansion}")
                    print(f"    File: {info.file_path}:{info.start_line}")
                    print()
            
            if functions:
                print("\nFunctions:")
                for name, info in functions.items():
                    print(f"  {name}")
                    print(f"    File: {info.file_path}:{info.start_line}-{info.end_line}")
                    print(f"    Return type: '{info.return_type}'")
        else:
            functions = extractor.parse_file(file_path, preprocess=preprocess)
            print(f"Found {len(functions)} functions:\n")
            for name, info in functions.items():
                print(f"Function: {name}")
                print(f"  File: {info.file_path}:{info.start_line}-{info.end_line}")
                print(f"  Return type: '{info.return_type}'")
                print(f"  Parameters:")
                for i, (ptype, pname) in enumerate(zip(info.arg_types, info.arg_names)):
                    print(f"    [{i}] {ptype} {pname}")
                print(f"  Callees: {info.callees}")
                print(f"  Return expressions: {info.return_expressions}")
                print()
    else:
        # Test with sample code
        sample = '''
std::shared_ptr<Grammar> perform_core(const char *s, size_t n,
                                      const Rules &rules, std::string &start,
                                      Log log) {
    Data data;
    auto r = g["Grammar"].parse(s, n, dt);
    if (!r.ret) {
        return nullptr;
    }
    return data.grammar;
}

void* my_malloc(size_t size) {
    return malloc(size);
}

static inline int compare(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

MyClass::MyClass(int value) : m_value(value) {
}
'''
        functions = extract_functions(sample, is_cpp=True)
        print(f"Found {len(functions)} functions:\n")
        for name, info in functions.items():
            print(f"Function: {name}")
            print(f"  File: {info.file_path}:{info.start_line}-{info.end_line}")
            print(f"  Return type: '{info.return_type}'")
            print(f"  Parameters:")
            for i, (ptype, pname) in enumerate(zip(info.arg_types, info.arg_names)):
                print(f"    [{i}] {ptype} {pname}")
            print(f"  Callees: {info.callees}")
            print(f"  Return expressions: {info.return_expressions}")
            print()