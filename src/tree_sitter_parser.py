"""Robust C/C++ function extractor using tree-sitter.

This module extracts function information from C/C++ source files using tree-sitter.
It handles complex C++ types including templates, references, qualified names, etc.

Key design: Extract raw text from AST nodes instead of reconstructing types.
"""

import logging
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


class CodeParser:
    """Extract functions from C/C++ source using tree-sitter."""

    def __init__(self):
        pass

    def parse_file(self, file_path: Path) -> dict[str, FunctionInfo]:
        """Parse a single file and extract all functions."""
        try:
            source = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Cannot read {file_path}: {e}")
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
            # For .h files, try C++ parser first (handles more cases)
            # If it has C++ constructs like class/template, C++ parser works better
            # C++ parser is also backward compatible with most C code
            parser = _get_cpp_parser()
        else:
            parser = _get_c_parser()

        tree = parser.parse(source)
        functions = {}

        # Find all function_definition nodes
        for node in self._iter_nodes(tree.root_node):
            if node.type == "function_definition":
                func_info = self._extract_function(node, source, str(file_path))
                if func_info and func_info.name:
                    functions[func_info.name] = func_info

        return functions

    def parse_source(self, source: str | bytes, filename: str = "<string>", is_cpp: bool = True) -> dict[str, FunctionInfo]:
        """Parse source code string and extract all functions."""
        if isinstance(source, str):
            source = source.encode('utf-8')

        parser = _get_cpp_parser() if is_cpp else _get_c_parser()
        tree = parser.parse(source)
        functions = {}

        for node in self._iter_nodes(tree.root_node):
            if node.type == "function_definition":
                func_info = self._extract_function(node, source, filename)
                if func_info and func_info.name:
                    functions[func_info.name] = func_info

        return functions

    def parse_project(self, project_path: Path) -> dict[str, FunctionInfo]:
        """Parse all C/C++ files in a project directory."""
        functions = {}
        extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}

        project_path = project_path.resolve()

        for file_path in project_path.rglob("*"):
            if file_path.suffix.lower() in extensions:
                try:
                    rel_path = str(file_path.relative_to(project_path))
                except ValueError:
                    rel_path = str(file_path)

                file_funcs = self.parse_file(file_path)
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
        """Extract FunctionInfo from a function_definition node.

        AST structure for function_definition:
        - Zero or more: type specifiers, qualifiers (storage_class_specifier, type_qualifier, etc.)
        - Return type: primitive_type, type_identifier, qualified_identifier, template_type, etc.
        - Declarator: function_declarator, pointer_declarator (wrapping function_declarator), etc.
        - Body: compound_statement
        """
        # Strategy: Find the declarator and body first, everything before declarator is return type
        declarator_node = None
        body_node = None

        # Identify key children
        children = list(node.children)
        declarator_idx = -1

        for i, child in enumerate(children):
            if child.type == "compound_statement":
                body_node = child
            elif child.type in ("function_declarator", "pointer_declarator",
                               "reference_declarator", "parenthesized_declarator"):
                declarator_node = child
                declarator_idx = i
            elif child.type == "ERROR":
                # Skip error nodes
                continue

        if not declarator_node:
            return None

        # Extract return type: everything before the declarator (excluding body)
        # Plus any pointer/reference symbols from the declarator
        return_type = ""
        if declarator_idx > 0:
            # Get text from start to declarator
            type_start = children[0].start_byte
            type_end = declarator_node.start_byte
            return_type = source[type_start:type_end].decode().strip()

        # Add pointer/reference from declarator (e.g., void* -> pointer is in declarator)
        ptr_ref_suffix = self._get_ptr_ref_from_declarator(declarator_node)
        if ptr_ref_suffix:
            return_type = return_type + ptr_ref_suffix

        # Extract function name and parameters from declarator
        name, arg_names, arg_types = self._parse_declarator(declarator_node, source)

        if not name:
            return None

        # Extract callees and return expressions from body
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

        # Full function code
        code = source[node.start_byte:node.end_byte].decode()

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

    def _parse_declarator(self, node, source: bytes) -> tuple[str, list[str], list[str]]:
        """Parse declarator to extract function name and parameters.

        Handles:
        - function_declarator: foo(int x, char* y)
        - pointer_declarator: *foo(int x) for pointer return
        - reference_declarator: &foo() for reference return (rare)
        """
        # Unwrap pointer/reference declarators to find function_declarator
        func_decl = self._find_function_declarator(node)
        if not func_decl:
            return "", [], []

        # Extract function name (first identifier in function_declarator)
        name = ""
        for child in func_decl.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte].decode()
                break
            elif child.type in ("field_identifier", "destructor_name"):
                name = source[child.start_byte:child.end_byte].decode()
                break
            elif child.type == "scoped_identifier":
                # Class::method - extract method name
                name = source[child.start_byte:child.end_byte].decode()
                break

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
                # Could be type or name, last one is usually name
                pname = source[child.start_byte:child.end_byte].decode()
                name_node = child
            elif child.type in ("pointer_declarator", "reference_declarator", "array_declarator"):
                # Name is inside the declarator
                pname = self._extract_name_from_declarator(child, source)
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

    def _extract_name_from_declarator(self, node, source: bytes) -> str:
        """Extract the identifier name from a declarator."""
        for child in self._iter_nodes(node):
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte].decode()
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
        # Handle qualified names: ns::func()
        if "::" in text:
            text = text.split("::")[-1]
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
def extract_functions(source: str | bytes | Path, is_cpp: bool = True) -> dict[str, FunctionInfo]:
    """Extract functions from source code or file.

    Args:
        source: Source code string, bytes, or Path to file
        is_cpp: Whether to parse as C++ (default) or C

    Returns:
        Dict mapping function names to FunctionInfo objects
    """
    extractor = FunctionExtractor()

    if isinstance(source, Path):
        return extractor.parse_file(source)
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
        extractor = FunctionExtractor()
        functions = extractor.parse_file(file_path)
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

    # Print results
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