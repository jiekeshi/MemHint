# Hint

**Hint**: Guiding Static Analysis with LLM-Assisted Memory-Semantics Annotation

Hint uses LLMs to automatically generate memory management annotations for C/C++ code, use z3-based constraint solving to validate them, and export them to formats compatible with popular static analyzers (CodeQL, Facebook Infer is what I want to support but not finished). This process enhances the analyzers' ability to detect memory safety issues such as leaks, double frees, use-after-free, and null dereferences.

## Supported Memory Safety Issues

| Issue Type | Description | CodeQL
|------------|-------------|--------
| **Memory Leak** | Allocated memory never freed | ✓ | ✓ |
| **Double Free** | Memory freed multiple times | ✓ |
| **Use After Free** | Accessing freed memory | ✓ |
| **Null Dereference** | Dereferencing NULL pointer | ✓ |
| **Buffer Overflow** | Writing beyond buffer bounds | ? not sure, did I write the query?|
| **Uninitialized Read** | Reading uninitialized memory | ? not sure, did I write the query?|

## Annotation Types

Hint generates the following annotation types:

### Allocation Annotations
- `ALLOC_SOURCE` - Function returns newly allocated heap memory
- `ARRAY_ALLOC` - Function allocates array (like calloc)
- `REALLOC` - Function reallocates memory

### Deallocation Annotations
- `FREE_SINK` - Function frees memory at specified argument

### Ownership Annotations
- `OWNERSHIP_TRANSFER` - Ownership transferred to callee
- `OWNERSHIP_RETURN` - Ownership transferred to caller
- `OWNERSHIP_ARG_OUT` - Ownership via output parameter

### Null Safety Annotations
- `MUST_CHECK_NULL` - Return value must be null-checked
- `NULLABLE_RETURN` - May return NULL
- `NONNULL_ARG` - Argument must not be NULL

### Negative Annotations (prevent false positives)
- `STATIC_BUFFER` - Returns static/global buffer
- `BORROWED_REF` - Returns borrowed reference
- `NOT_ALLOC` - Explicitly not an allocator


## Installation

I only tried this on MacOS:

```bash
python3 -m venv py314
source py314/bin/activate
pip install -r requirements.txt

brew install llvm
brew install codeql
```

## Basic Usage

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
# Detect all memory safety issues
python main.py --project ./data/example

# Detect specific issue types
python main.py --project ./data/example --issues leak double-free uaf

```

## TODOs

- [ ] Support more static analyzers (e.g., Facebook Infer)
- [ ] Have not tested on any real-world projects or benchmarks or larger codebases yet
    - [ ] Find some C/C++ projects with known memory safety issues to test on
    - [ ] Record results and use some metrics (precision, recall, F1-score, etc.)
    - [ ] Choose some baseline static analyzers to compare with (e.g., CodeQL without annotations, Clang Static Analyzer, etc.)
- [ ] Optimize performance for large codebases, for now it use sliced analysis but still not guaranteed to be fast enough in very large codebases
- [ ] Improve annotation quality by experimenting with different LLMs like applying Gemini