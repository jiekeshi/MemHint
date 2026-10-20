import logging
import re
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.models import FunctionInfo

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
        self.pointer_typedef_aliases: set[str] = set()
    
    def _collect_pointer_typedef_aliases(self, root, source: bytes) -> set[str]:
        """Collect pointer typedef aliases using tree-sitter AST nodes."""

        def _unwrap_typedef_declarator(decl_node):
            if decl_node is None:
                return None
            if decl_node.type in ("parenthesized_declarator", "function_declarator"):
                inner = decl_node.child_by_field_name("declarator")
                return _unwrap_typedef_declarator(inner) if inner is not None else decl_node
            return decl_node

        def _iter_typedef_alias_declarators(type_def_node):
            """Yield every alias declarator in a typedef, comma-separated ones
            included."""
            main_decl = _unwrap_typedef_declarator(type_def_node.child_by_field_name("declarator"))
            if main_decl is not None:
                yield main_decl

            for child in type_def_node.children:
                if child.type == "init_declarator":
                    alias_decl = _unwrap_typedef_declarator(child.child_by_field_name("declarator"))
                    if alias_decl is not None:
                        yield alias_decl

        aliases: set[str] = set()

        for node in self._iter_nodes(root):
            if node.type != "type_definition":
                continue

            alias_decls = list(_iter_typedef_alias_declarators(node))
            if not alias_decls:
                continue

            first_decl_start = min(d.start_byte for d in alias_decls)
            base_type_text = source[node.start_byte:first_decl_start].decode(errors="ignore")

            for decl in alias_decls:
                alias_name = self._extract_declarator_name_only(decl, source)
                if not alias_name:
                    continue

                if decl.type == "pointer_declarator":
                    aliases.add(alias_name)
                    continue

                if "*" in base_type_text:
                    aliases.add(alias_name)
                    continue

        if logger.isEnabledFor(logging.DEBUG) and aliases:
            logger.debug(f"Collected {len(aliases)} pointer typedef aliases: {sorted(aliases)}")

        return aliases

    
    def _preprocess_file(self, file_path: Path) -> bytes:
        """Preprocess a file to expand macros."""
        cpp_extensions = {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".C", ".CPP"}
        is_cpp = file_path.suffix in cpp_extensions
        
        compiler = None
        for cmd in ["clang", "gcc"]:
            if shutil.which(cmd):
                compiler = cmd
                break
        
        if not compiler:
            raise RuntimeError("No C/C++ compiler found (clang or gcc required for preprocessing)")
        
        cmd = [compiler, "-E"]
        if is_cpp:
            cmd.append("-x")
            cmd.append("c++")
        else:
            cmd.append("-x")
            cmd.append("c")
        
        cmd.append(str(file_path))
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                timeout=30,
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
        """Parse a single file and extract all functions."""
        try:
            if preprocess:
                source = self._preprocess_file(file_path)
            else:
                source = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Cannot read/preprocess {file_path}: {e}")
            return {}

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

        if collect_typedefs:
            try:
                aliases = self._collect_pointer_typedef_aliases(tree.root_node, source)
                self.pointer_typedef_aliases.update(aliases)
            except Exception:
                pass

        functions: dict[str, FunctionInfo] = {}

        for node in self._iter_nodes(tree.root_node):
            if node.type == "function_definition":
                func_info = self._extract_function(node, source, str(file_path))
                if func_info and func_info.name:
                    functions[func_info.name] = func_info

        return functions

    def get_function_containing_line(
        self,
        file_path: Path,
        line_1based: int,
        preprocess: bool = False,
    ) -> Optional[FunctionInfo]:
        """Return the function that contains the given line (1-based)."""
        try:
            if preprocess:
                source = self._preprocess_file(file_path)
            else:
                source = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Cannot read/preprocess {file_path}: {e}")
            return None

        cpp_extensions = {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".C", ".CPP"}
        c_only_extensions = {".c"}
        if file_path.suffix in cpp_extensions:
            parser = _get_cpp_parser()
        elif file_path.suffix in c_only_extensions:
            parser = _get_c_parser()
        else:
            parser = _get_cpp_parser()

        tree = parser.parse(source)
        file_path_str = str(file_path)
        candidates: list[tuple[int, int, object]] = []

        for node in self._iter_nodes(tree.root_node):
            if node.type != "function_definition":
                continue
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            if start_line <= line_1based <= end_line:
                span = end_line - start_line
                candidates.append((span, start_line, node))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, node = candidates[0]
        return self._extract_function(node, source, file_path_str)

    def parse_file_with_macros(
        self,
        file_path: Path,
        preprocess: bool = False,
        collect_typedefs: bool = True,
    ) -> tuple[dict[str, FunctionInfo], dict[str, MacroInfo]]:
        """Parse a single file and extract both functions and macros."""
        try:
            source = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return {}, {}

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

        if collect_typedefs:
            try:
                aliases = self._collect_pointer_typedef_aliases(tree.root_node, source)
                self.pointer_typedef_aliases.update(aliases)
            except Exception:
                pass

        functions: dict[str, FunctionInfo] = {}
        macros: dict[str, MacroInfo] = {}

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



    def _iter_nodes(self, node):
        """Iterate all nodes in the AST without recursive calls."""
        stack = [node]
        seen_ids: set[int] = set()

        while stack:
            current = stack.pop()
            node_id = id(current)
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)

            yield current

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

        return_type = ""
        if declarator_idx > 0:
            type_start = children[0].start_byte
            type_end = declarator_node.start_byte
            return_type = source[type_start:type_end].decode(errors="ignore").strip()

        ptr_ref_suffix = self._get_ptr_ref_from_declarator(declarator_node)
        if ptr_ref_suffix:
            return_type = return_type + ptr_ref_suffix

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
        """Parse declarator to extract function name and parameters."""
        func_decl = self._find_function_declarator(node)
        if not func_decl:
            return "", [], []

        name = ""
        is_field_identifier = False
        for child in func_decl.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte].decode()
                break
            elif child.type in ("field_identifier", "destructor_name"):
                name = source[child.start_byte:child.end_byte].decode()
                is_field_identifier = True
                break
            elif child.type == "scoped_identifier":
                name = source[child.start_byte:child.end_byte].decode()
                break

        if is_field_identifier and function_node is not None and full_source is not None:
            class_name = self._get_containing_class_name(function_node, full_source)
            if class_name:
                name = f"{class_name}::{name}"

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
        """Extract pointer (*) and reference (&) symbols from declarator
        wrapper."""
        symbols = ""
        current = node

        while current.type in ("pointer_declarator", "reference_declarator"):
            if current.type == "pointer_declarator":
                symbols += "*"
            elif current.type == "reference_declarator":
                symbols += "&"

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
        """Parse a parameter declaration to extract type and name."""
        full_text = source[node.start_byte:node.end_byte].decode().strip()

        pname = ""
        name_node = None

        for child in node.children:
            if child.type == "identifier":
                pname = source[child.start_byte:child.end_byte].decode()
                name_node = child
            elif child.type in ("pointer_declarator", "reference_declarator", "array_declarator", "parenthesized_declarator", "function_declarator"):
                cand = self._extract_declarator_name_only(child, source)
                if cand:
                    pname = cand
                    name_node = child
                    break

        if pname and name_node:
            ptype = full_text
            if name_node.type == "identifier":
                ptype = source[node.start_byte:name_node.start_byte].decode().strip()
            else:
                ptype = self._extract_type_from_param(node, name_node, source)
        else:
            ptype = full_text
            pname = ""

        ptype = " ".join(ptype.split())

        return ptype, pname


    def _extract_declarator_name_only(self, decl_node, source: bytes) -> str:
        """Extract the name of the *declarator itself* by following the
        'declarator' chain."""
        if decl_node is None:
            return ""

        cur = decl_node

        while True:
            if cur.type in ("identifier", "type_identifier", "field_identifier"):
                return source[cur.start_byte:cur.end_byte].decode()

            next_decl = cur.child_by_field_name("declarator")
            if next_decl is not None:
                cur = next_decl
                continue

            if cur.type == "function_declarator":
                for ch in cur.children:
                    if ch.type == "parameter_list":
                        continue
                    if ch.type in ("identifier", "type_identifier", "field_identifier", "scoped_identifier"):
                        return source[ch.start_byte:ch.end_byte].decode()
                return ""

            return ""

    def _extract_type_from_param(self, param_node, name_node, source: bytes) -> str:
        """Extract the type portion from a parameter, excluding the name."""
        type_parts = []

        for child in param_node.children:
            if child == name_node:
                if child.type == "pointer_declarator":
                    type_parts.append("*" * self._count_pointer_depth(child))
                elif child.type == "reference_declarator":
                    type_parts.append("&")
                elif child.type == "array_declarator":
                    type_parts.append("*")
                continue

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

        if "->" in text:
            text = text.split("->")[-1]
        if "." in text:
            text = text.split(".")[-1]
        if "<" in text:
            text = text.split("<")[0]
        text = text.strip("()")

        return text.strip()

    def _extract_return_expr(self, return_node, source: bytes) -> str:
        """Extract the return expression from a return_statement."""
        for child in return_node.children:
            if child.type not in ("return", ";"):
                return source[child.start_byte:child.end_byte].decode().strip()
        return ""

    def _get_containing_class_name(self, node, source: bytes) -> Optional[str]:
        """Get the name of the class that contains this function definition."""
        current = node.parent
        while current is not None:
            if current.type in ("class_specifier", "struct_specifier"):
                for child in current.children:
                    if child.type == "type_identifier":
                        return source[child.start_byte:child.end_byte].decode()
                break
            current = current.parent
        return None

    def _extract_macro(self, node, source: bytes, file_path: str) -> Optional[MacroInfo]:
        """Extract MacroInfo from a preproc_def or preproc_function_def node."""
        children = list(node.children)
        if len(children) < 2:
            return None
        
        start_idx = 0
        if children[0].type == "#define":
            start_idx = 1
        
        if start_idx >= len(children):
            return None
        
        name_node = children[start_idx]
        if name_node.type != "identifier":
            return None
        
        name = source[name_node.start_byte:name_node.end_byte].decode()
        
        is_function_like = node.type == "preproc_function_def"
        arg_names = []
        expansion = ""
        
        params_node = None
        expansion_start_idx = start_idx + 1
        
        for i, child in enumerate(children[start_idx + 1:], start=start_idx + 1):
            if child.type == "preproc_params":
                is_function_like = True
                params_node = child
                expansion_start_idx = i + 1
                break
        
        if params_node:
            for param_child in params_node.children:
                if param_child.type == "identifier":
                    param_name = source[param_child.start_byte:param_child.end_byte].decode()
                    arg_names.append(param_name)
        
        if expansion_start_idx < len(children):
            expansion_node = children[expansion_start_idx]
            if expansion_node.type == "preproc_arg":
                expansion = source[expansion_node.start_byte:expansion_node.end_byte].decode().strip()
            else:
                expansion_start = expansion_node.start_byte
                expansion_end = node.end_byte
                expansion = source[expansion_start:expansion_end].decode().strip()
            expansion = " ".join(expansion.split())
        
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
            func.callees = {c for c in func.callees if c in func_names}

        for name, func in functions.items():
            for callee in func.callees:
                if callee in functions:
                    functions[callee].callers.add(name)


# Convenience function
