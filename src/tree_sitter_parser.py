"""Tree-sitter based C/C++ parser for function extraction."""

import logging
from pathlib import Path

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

from src.core.models import FunctionInfo

# Note: FunctionInfo doesn't have return_expressions in the new models
# We add it here for compatibility

logger = logging.getLogger(__name__)

C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())


class CodeParser:
    """Extract function information using tree-sitter."""

    def __init__(self):
        self.c_parser = Parser(C_LANGUAGE)
        self.cpp_parser = Parser(CPP_LANGUAGE)

    def parse_project(self, project_path: Path) -> dict[str, FunctionInfo]:
        """Parse all C/C++ files in project."""
        functions = {}
        extensions = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")

        # Resolve project path
        project_path = project_path.resolve()

        for file_path in project_path.rglob("*"):
            if file_path.suffix in extensions:
                # Store relative path from project root
                rel_path = file_path.relative_to(project_path)
                file_funcs = self._parse_file(file_path, str(rel_path))
                for name, info in file_funcs.items():
                    if name in functions:
                        if len(info.code) > len(functions[name].code):
                            functions[name] = info
                    else:
                        functions[name] = info

        self._resolve_calls(functions)
        logger.info(f"Parsed {len(functions)} functions from {project_path}")
        return functions

    def _parse_file(self, file_path: Path, rel_path: str = None) -> dict[str, FunctionInfo]:
        """Parse a single file."""
        try:
            code = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return {}

        # Use relative path if provided, otherwise use file name
        stored_path = rel_path or file_path.name

        parser = self.cpp_parser if file_path.suffix in (".cpp", ".cc", ".cxx", ".hpp") else self.c_parser
        tree = parser.parse(code)
        functions = {}

        for node in self._walk(tree.root_node):
            if node.type == "function_definition":
                info = self._extract_function(node, code, stored_path)
                if info:
                    functions[info.name] = info
        return functions

    def _walk(self, node):
        """Recursively walk all nodes."""
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _extract_function(self, node, source: bytes, file_path: str) -> FunctionInfo:
        """Extract FunctionInfo from function_definition node."""
        name = ""
        return_type = ""
        arg_names = []
        arg_types = []
        callees = set()
        return_exprs = set()

        declarator = None
        pointer_depth = 0  # Track number of * in return type

        for child in node.children:
            if child.type == "pointer_declarator":
                # Count pointer depth and find inner declarator
                declarator, pointer_depth = self._unwrap_pointer_declarator(child, source)
            elif child.type in ("declarator", "function_declarator"):
                declarator = child
            elif child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                return_type = source[child.start_byte:child.end_byte].decode()
            elif child.type == "type_qualifier":
                # Handle const, volatile, etc.
                qualifier = source[child.start_byte:child.end_byte].decode()
                if return_type:
                    return_type = qualifier + " " + return_type
                else:
                    return_type = qualifier

        # Append pointer stars to return type
        if pointer_depth > 0:
            return_type = return_type + "*" * pointer_depth

        if declarator:
            name, arg_names, arg_types = self._parse_declarator(declarator, source)

        if not name:
            return None

        body = None
        for child in node.children:
            if child.type == "compound_statement":
                body = child
                break

        if body:
            for n in self._walk(body):
                if n.type == "call_expression":
                    func_node = n.child_by_field_name("function")
                    if func_node:
                        callee = source[func_node.start_byte:func_node.end_byte].decode()
                        if "->" in callee or "." in callee:
                            callee = callee.split("->")[-1].split(".")[-1]
                        callees.add(callee)
                if n.type == "return_statement":
                    for c in n.children:
                        if c.type not in ("return", ";"):
                            expr = source[c.start_byte:c.end_byte].decode().strip()
                            return_exprs.add(expr)

        code = source[node.start_byte:node.end_byte].decode()
        return FunctionInfo(
            name=name,
            code=code,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            callees=callees,
            arg_names=arg_names,
            arg_types=arg_types,
            return_expressions=return_exprs,
            return_type=return_type,
        )

    def _unwrap_pointer_declarator(self, node, source: bytes) -> tuple:
        """Unwrap nested pointer_declarator to get the inner declarator and pointer depth.

        For `char* foo()`, the AST is:
            pointer_declarator
              └── function_declarator
                    └── identifier: "foo"

        For `char** foo()`, the AST is:
            pointer_declarator
              └── pointer_declarator
                    └── function_declarator
                          └── identifier: "foo"

        Returns: (inner_declarator, pointer_depth)
        """
        pointer_depth = 0
        current = node

        while current.type == "pointer_declarator":
            pointer_depth += 1
            # Find the child that's not '*'
            for child in current.children:
                if child.type in ("pointer_declarator", "function_declarator", "declarator"):
                    current = child
                    break
            else:
                # No declarator child found, break
                break

        return current, pointer_depth

    def _parse_declarator(self, node, source: bytes) -> tuple[str, list[str], list[str]]:
        """Parse function declarator for name and parameters."""
        name = ""
        arg_names = []
        arg_types = []

        for child in self._walk(node):
            if child.type == "identifier" and not name:
                name = source[child.start_byte:child.end_byte].decode()
            elif child.type == "parameter_list":
                for param in child.children:
                    if param.type == "parameter_declaration":
                        ptype, pname = self._parse_param(param, source)
                        if pname:
                            arg_names.append(pname)
                            arg_types.append(ptype)
        return name, arg_names, arg_types

    def _parse_param(self, node, source: bytes) -> tuple[str, str]:
        """Parse parameter declaration."""
        ptype = ""
        pname = ""
        for child in node.children:
            if child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                ptype = source[child.start_byte:child.end_byte].decode()
            elif child.type == "identifier":
                pname = source[child.start_byte:child.end_byte].decode()
            elif child.type == "pointer_declarator":
                for c in self._walk(child):
                    if c.type == "identifier":
                        pname = source[c.start_byte:c.end_byte].decode()
                ptype += "*"
        return ptype, pname

    def _resolve_calls(self, functions: dict[str, FunctionInfo]) -> None:
        """Resolve caller/callee relationships."""
        for func in functions.values():
            func.callees = {c for c in func.callees if c in functions}
        for name, func in functions.items():
            for callee in func.callees:
                if callee in functions:
                    functions[callee].callers.add(name)