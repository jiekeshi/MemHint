#!/usr/bin/env python3
"""
Script to scan for CWE files in a directory and find matching entries in manifest.xml
or analyze main functions to determine good/bad testcases.

Two methods are supported:
1. Manifest-based: Uses manifest.xml to identify testcases and flaws
2. Main function analysis: Analyzes main() functions to detect calls to _good() or _bad() 
   functions to determine testcase type

By default, both methods are run and results are saved to the 'results' directory:
- results/{dir_name}_manifest_gt.json - Ground truth from manifest method
- results/{dir_name}_main_gt.json - Ground truth from main function analysis method
- results/{dir_name}_{subdir}_combined_results.json - Combined evaluation metrics from both methods

Configuration:
    Edit the DEFAULT_CWE_DIR and DEFAULT_OUTPUT_SUBDIR variables at the top of the script
    to change the default CWE directory and output subdirectory names.

Usage:
    python cal_juliet_results.py [directory] [predictions_file] [--both|--manifest|--main-analysis]
    
    --both (default): Run both methods and combine results
    --manifest: Run only manifest method
    --main-analysis: Run only main function analysis method
"""

import os
import sys
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION SETTINGS - Modify these as needed
# ============================================================================
# Default CWE testcase directory name (e.g., "CWE401_Memory_Leak", "CWE415_Double_Free","CWE416_Use_After_Free")
DEFAULT_CWE_DIR = "CWE401_Memory_Leak"

# Default output subdirectory name (e.g., "output_codeql_only", "output")
DEFAULT_OUTPUT_SUBDIR = "output_single_file"

# ============================================================================

# Import tree-sitter parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
try:
    from tree_sitter_parser import CodeParser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree_sitter_parser not available, falling back to regex-based parsing")


def extract_cwe_number(directory):
    """
    Extract CWE number from directory path (e.g., "CWE415_Double_Free" -> "CWE415")
    
    Args:
        directory: Path to scan
        
    Returns:
        CWE number string (e.g., "CWE415") or None if not found
    """
    # Try to extract from directory name
    dir_name = os.path.basename(os.path.normpath(directory))
    match = re.match(r'(CWE\d+)', dir_name)
    if match:
        return match.group(1)
    
    # Try to find CWE pattern in any part of the path
    path_parts = Path(directory).parts
    for part in reversed(path_parts):
        match = re.match(r'(CWE\d+)', part)
        if match:
            return match.group(1)
    
    return None


def find_cwe_files(directory, cwe_number):
    """
    Recursively scan directory for files starting with the specified CWE number and having extensions .c, .cpp, or .h
    
    Args:
        directory: Path to scan
        cwe_number: CWE number to search for (e.g., "CWE401", "CWE415")
        
    Returns:
        List of file paths
    """
    cwe_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return cwe_files
    
    if not cwe_number:
        print(f"Error: CWE number not found in directory path")
        return cwe_files
    
    # Recursively find all files starting with the CWE number
    for ext in ['.c', '.cpp', '.h']:
        pattern = f'{cwe_number}*{ext}'
        for file_path in directory_path.rglob(pattern):
            if file_path.is_file() and file_path.name.startswith(cwe_number):
                cwe_files.append(file_path)
    
    return sorted(cwe_files)


def parse_manifest(manifest_path, cwe_number):
    """
    Parse manifest.xml and extract testcase entries for the specified CWE number
    
    Args:
        manifest_path: Path to manifest.xml
        cwe_number: CWE number to filter for (e.g., "CWE401", "CWE415")
        
    Returns:
        Dictionary mapping filename to list of testcase entries (as XML elements)
        Also returns dictionary mapping testcase to all files in it
        Also returns total count of testcase blocks for this CWE
    """
    if not os.path.exists(manifest_path):
        return {}, {}, 0
    
    try:
        tree = ET.parse(manifest_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return {}, {}, 0
    
    if not cwe_number:
        print(f"Error: CWE number not provided")
        return {}, {}, 0
    
    # Dictionary to store filename -> list of testcase entries
    manifest_entries = defaultdict(list)
    # Dictionary to store testcase -> all files in that testcase
    testcase_files = {}
    # Count total testcase blocks for this CWE
    total_cwe_testcases = 0
    
    # Iterate through all testcase elements
    for testcase in root.findall('testcase'):
        # Get all file elements in this testcase
        file_elems = testcase.findall('file')
        
        # Check if this testcase has any files for this CWE
        has_cwe_files = False
        testcase_cwe_files = []
        
        for file_elem in file_elems:
            file_path_attr = file_elem.get('path')
            if file_path_attr and file_path_attr.startswith(cwe_number):
                has_cwe_files = True
                filename = os.path.basename(file_path_attr)
                entry = {
                    'testcase': testcase,
                    'file_elem': file_elem,
                    'path': file_path_attr,
                    'has_flaw': file_elem.find('flaw') is not None
                }
                manifest_entries[filename].append(entry)
                testcase_cwe_files.append(entry)
        
        # If this testcase has files for this CWE, count it and store
        if has_cwe_files:
            total_cwe_testcases += 1
            testcase_files[testcase] = testcase_cwe_files
    
    return manifest_entries, testcase_files, total_cwe_testcases


def find_function_at_line(file_path, target_line):
    """
    Find the function name that contains the specified line number using tree-sitter parser
    
    Args:
        file_path: Path to the source file
        target_line: Line number to find the function for (1-indexed)
        
    Returns:
        Function name if found, None otherwise
    """
    if not os.path.exists(file_path):
        return None
    
    # Use tree-sitter parser if available
    if TREE_SITTER_AVAILABLE:
        try:
            parser = CodeParser()
            file_path_obj = Path(file_path)
            functions = parser.parse_file(file_path_obj)
            
            # Find the function that contains the target line
            for func_name, func_info in functions.items():
                if func_info.start_line <= target_line <= func_info.end_line:
                    return func_name
        except Exception as e:
            # Fall back to regex if tree-sitter fails
            pass
    
    # Fallback: simple regex-based approach (less accurate but works without tree-sitter)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return None
    
    if target_line < 1 or target_line > len(lines):
        return None
    
    # Simple pattern to match function definitions
    function_pattern = re.compile(
        r'^\s*(?:static\s+|inline\s+)?'
        r'(?:\w+\s+)*'
        r'(\w+)\s*'
        r'\([^)]*\)\s*'
        r'(?:\s*\{)?\s*$'
    )
    
    current_function = None
    brace_count = 0
    in_function = False
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip comments and preprocessor directives
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('#'):
            if in_function:
                brace_count += line.count('{') - line.count('}')
                if i == target_line:
                    return current_function
                if brace_count <= 0:
                    in_function = False
                    current_function = None
            continue
        
        # Check for function definition
        match = function_pattern.match(line)
        if match:
            func_name = match.group(1)
            has_brace = '{' in stripped
            next_line_has_brace = (i < len(lines) and '{' in lines[i].strip())
            
            if (has_brace or next_line_has_brace) and not in_function:
                current_function = func_name
                in_function = True
                brace_count = stripped.count('{') - stripped.count('}')
                if next_line_has_brace and not has_brace:
                    brace_count = 1
                continue
        
        # Track braces
        if in_function:
            brace_count += line.count('{') - line.count('}')
            if i == target_line:
                return current_function
            if brace_count <= 0:
                in_function = False
                current_function = None
    
    return current_function


def extract_testcase_data(entry_info, source_file_path=None, all_files_in_testcase=None):
    """
    Extract testcase entry data as a dictionary
    
    Args:
        entry_info: Dictionary with testcase, file_elem, and path
        source_file_path: Optional path to the source file to find function name
        all_files_in_testcase: Optional list of all files in the same testcase
        
    Returns:
        Dictionary with testcase information
    """
    file_elem = entry_info['file_elem']
    path = entry_info['path']
    has_flaw = entry_info.get('has_flaw', False)
    
    result = {
        'file_path': path,
        'type': 'positive' if has_flaw else 'negative',  # positive = has bug, negative = no bug
        'flaw': None
    }
    
    # Get flaw information if available
    flaw = file_elem.find('flaw')
    if flaw is not None:
        flaw_line = flaw.get('line', None)
        flaw_name = flaw.get('name', None)
        
        flaw_data = {
            'line': int(flaw_line) if flaw_line and flaw_line.isdigit() else None,
            'name': flaw_name
        }
        
        # Try to find function name if source file is provided
        function_name = None
        if source_file_path and flaw_line and flaw_line.isdigit():
            try:
                flaw_line_num = int(flaw_line)
                function_name = find_function_at_line(source_file_path, flaw_line_num)
            except (ValueError, TypeError):
                pass
        
        if function_name:
            flaw_data['function_name'] = function_name
        
        result['flaw'] = flaw_data
    
    # Add information about other files in the same testcase
    if all_files_in_testcase:
        other_files = [f['path'] for f in all_files_in_testcase if f['path'] != path]
        if other_files:
            result['related_files'] = other_files
    
    return result


def normalize_file_path(file_path):
    """
    Normalize file path for comparison (extract just the filename)
    
    Args:
        file_path: File path string
        
    Returns:
        Normalized filename
    """
    # Extract just the filename from path
    filename = os.path.basename(file_path)
    return filename


def extract_main_function(file_path):
    """
    Extract the main function from a C/C++ file
    
    Args:
        file_path: Path to the source file
        
    Returns:
        Tuple of (main_function_lines, start_line, end_line) or (None, None, None) if not found
    """
    if not os.path.exists(file_path):
        return None, None, None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return None, None, None
    
    # Pattern to match main function definition - more flexible
    # Matches: int main(...) with various parameter styles
    main_pattern = re.compile(
        r'^\s*int\s+main\s*\([^)]*\)'
    )
    
    main_start = None
    brace_count = 0
    in_main = False
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip comments (but track braces if in main)
        if stripped.startswith('//') or stripped.startswith('/*'):
            if in_main:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    # End of main function
                    return lines[main_start-1:i], main_start, i
            continue
        
        # Skip preprocessor directives (but track braces if already in main)
        if stripped.startswith('#'):
            if in_main:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    # End of main function
                    return lines[main_start-1:i], main_start, i
            continue
        
        # Check for main function definition (use search instead of match for more flexibility)
        if main_pattern.search(line):
            main_start = i
            in_main = True
            brace_count = stripped.count('{') - stripped.count('}')
            if '{' not in stripped:
                # Opening brace might be on next line
                if i < len(lines) and '{' in lines[i].strip():
                    brace_count = 1
            continue
        
        # Track braces if in main function
        if in_main:
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                # End of main function
                return lines[main_start-1:i], main_start, i
    
    # If we found main but didn't close it, return what we have
    if in_main and main_start:
        return lines[main_start-1:], main_start, len(lines)
    
    return None, None, None


def analyze_main_function_for_good_bad(file_path):
    """
    Analyze the main function to determine if it calls _good() or _bad() functions
    
    Args:
        file_path: Path to the source file
        
    Returns:
        Dictionary with 'type' ('positive' or 'negative'), 'good_functions' list, 'bad_functions' list,
        or None if main function not found or cannot be analyzed
    """
    main_lines, start_line, end_line = extract_main_function(file_path)
    
    if main_lines is None:
        return None
    
    # Join main function lines for analysis
    main_content = ''.join(main_lines)
    
    # Pattern to match function calls ending with _good() or _bad()
    # Matches: function_name_good() or function_name_bad()
    good_pattern = re.compile(r'(\w+_good)\s*\(')
    bad_pattern = re.compile(r'(\w+_bad)\s*\(')
    
    good_functions = good_pattern.findall(main_content)
    bad_functions = bad_pattern.findall(main_content)
    
    # Determine type: if main calls _bad(), it's positive (has bug); if only _good(), it's negative (no bug)
    # If both are present, prioritize _bad() as positive (has bug)
    if bad_functions:
        testcase_type = 'positive'  # positive = has bug
    elif good_functions:
        testcase_type = 'negative'  # negative = no bug
    else:
        # No _good() or _bad() found
        return None
    
    return {
        'type': testcase_type,
        'good_functions': good_functions,
        'bad_functions': bad_functions,
        'main_start_line': start_line,
        'main_end_line': end_line
    }


def find_bad_files_from_bin(bin_dir, cwe_number):
    """
    Find files marked as 'bad' from the bin/CWE{number}/bad directory
    
    Args:
        bin_dir: Base directory containing bin/CWE{number}/bad
        cwe_number: CWE number to filter files
        
    Returns:
        Set of base filenames (without -bad suffix) that are marked as bad
    """
    bad_files = set()
    bad_dir = os.path.join(bin_dir, "bin", cwe_number, "bad")
    
    if not os.path.exists(bad_dir):
        return bad_files
    
    try:
        for filename in os.listdir(bad_dir):
            # Files in bad directory typically end with -bad
            # Extract the base filename
            if filename.endswith("-bad"):
                base_name = filename[:-4]  # Remove "-bad" suffix
                bad_files.add(base_name)
            else:
                # Some files might not have the suffix, use as is
                bad_files.add(filename)
    except Exception as e:
        print(f"Warning: Could not scan bad directory {bad_dir}: {e}")
    
    return bad_files


def find_good_files_from_bin(bin_dir, cwe_number):
    """
    Find files marked as 'good' from the bin/CWE{number}/good directory
    
    Args:
        bin_dir: Base directory containing bin/CWE{number}/good
        cwe_number: CWE number to filter files
        
    Returns:
        Set of base filenames (without -good suffix) that are marked as good
    """
    good_files = set()
    good_dir = os.path.join(bin_dir, "bin", cwe_number, "good")
    
    if not os.path.exists(good_dir):
        return good_files
    
    try:
        for filename in os.listdir(good_dir):
            # Files in good directory typically end with -good
            # Extract the base filename
            if filename.endswith("-good"):
                base_name = filename[:-4]  # Remove "-good" suffix
                good_files.add(base_name)
            elif filename.endswith("-good1") or filename.endswith("-good2"):
                # Some files have -good1 or -good2 suffix
                base_name = filename.rsplit("-good", 1)[0]
                good_files.add(base_name)
            else:
                # Some files might not have the suffix, use as is
                good_files.add(filename)
    except Exception as e:
        print(f"Warning: Could not scan good directory {good_dir}: {e}")
    
    return good_files


def build_ground_truth_from_main_functions(directory, cwe_number, bin_dir=None):
    """
    Build ground truth by analyzing main functions in C/C++/H files
    Optionally uses bin/CWE{number}/bad directory to identify bad functions
    
    Args:
        directory: Directory to scan
        cwe_number: CWE number to filter files
        bin_dir: Optional base directory for bin/CWE{number}/bad (defaults to juliet-test-suite-c)
        
    Returns:
        Dictionary with ground truth results
    """
    # Find all CWE files
    cwe_files = find_cwe_files(directory, cwe_number)
    
    # Try to find bad and good files from bin directory
    bad_files_from_bin = set()
    good_files_from_bin = set()
    if bin_dir:
        bad_files_from_bin = find_bad_files_from_bin(bin_dir, cwe_number)
        good_files_from_bin = find_good_files_from_bin(bin_dir, cwe_number)
    else:
        # Try to find juliet-test-suite-c directory relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = script_dir
        juliet_dir = os.path.join(base_dir, "juliet-test-suite-c")
        if not os.path.exists(juliet_dir):
            base_dir = os.path.dirname(script_dir)
            juliet_dir = os.path.join(base_dir, "juliet-test-suite-c")
        if os.path.exists(juliet_dir):
            bad_files_from_bin = find_bad_files_from_bin(juliet_dir, cwe_number)
            good_files_from_bin = find_good_files_from_bin(juliet_dir, cwe_number)
    
    if bad_files_from_bin:
        print(f"Found {len(bad_files_from_bin)} files marked as 'bad' in bin directory")
    if good_files_from_bin:
        print(f"Found {len(good_files_from_bin)} files marked as 'good' in bin directory")
    
    results = {
        'summary': {
            'total_cwe_files': len(cwe_files),
            'positive_testcases': 0,
            'negative_testcases': 0,
            'files_with_main': 0,
            'files_without_main': 0
        },
        'scan_directory': directory,
        'cwe_number': cwe_number,
        'method': 'main_function_analysis',
        'total_files_found': len(cwe_files),
        'matched_files': [],
        'unmatched_files': []
    }
    
    for file_path in cwe_files:
        filename = file_path.name
        file_path_str = str(file_path)
        
        # Check if this file is marked as 'bad' or 'good' in the bin directory
        # Extract base filename (without extension) for matching
        base_filename = os.path.splitext(filename)[0]
        is_bad_from_bin = base_filename in bad_files_from_bin
        is_good_from_bin = base_filename in good_files_from_bin
        
        # Analyze main function
        main_analysis = analyze_main_function_for_good_bad(file_path_str)
        
        if main_analysis:
            results['summary']['files_with_main'] += 1
            
            # Extract bad functions (these are the ones with flaws/bugs)
            bad_functions = main_analysis.get('bad_functions', [])
            good_functions = main_analysis.get('good_functions', [])
            
            # If file is marked as bad in bin directory but no _bad() function found in main,
            # we should still mark it as positive. Try to find the bad function in the file.
            if is_bad_from_bin and not bad_functions:
                # Look for a function ending with _bad() in the file
                try:
                    with open(file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    # Pattern to find function definitions ending with _bad
                    bad_func_pattern = re.compile(r'void\s+(\w+_bad)\s*\(')
                    found_bad_funcs = bad_func_pattern.findall(content)
                    if found_bad_funcs:
                        bad_functions = found_bad_funcs
                except Exception:
                    pass  # If we can't read the file, continue with existing bad_functions
            
            # If file is marked as good in bin directory but no _good() function found in main,
            # we should still mark it as negative. Try to find the good function in the file.
            if is_good_from_bin and not good_functions:
                # Look for a function ending with _good() in the file
                try:
                    with open(file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    # Pattern to find function definitions ending with _good
                    good_func_pattern = re.compile(r'void\s+(\w+_good)\s*\(')
                    found_good_funcs = good_func_pattern.findall(content)
                    if found_good_funcs:
                        good_functions = found_good_funcs
                except Exception:
                    pass  # If we can't read the file, continue with existing good_functions
            
            # Create entries for this file
            entries = []
            
            # Create positive entries for each _bad() function (has bug)
            if bad_functions:
                for bad_func in bad_functions:
                    # Try to find the function definition to get line number
                    bad_func_line = find_function_line(file_path_str, bad_func)
                    
                    entry = {
                        'file_path': file_path_str,
                        'type': 'positive',  # positive = has bug
                        'flaw': {
                            'function_name': bad_func,
                            'line': bad_func_line
                        },
                        'identified_from_bin': is_bad_from_bin
                    }
                    entries.append(entry)
                
                results['summary']['positive_testcases'] += len(bad_functions)
            
            # Create negative entries for each _good() function (no bug)
            if good_functions:
                for good_func in good_functions:
                    entry = {
                        'file_path': file_path_str,
                        'type': 'negative',  # negative = no bug
                        'flaw': None,
                        'good_function': good_func,
                        'identified_from_bin': is_good_from_bin
                    }
                    entries.append(entry)
                
                results['summary']['negative_testcases'] += len(good_functions)
            
            results['matched_files'].append({
                'filename': filename,
                'full_path': file_path_str,
                'entries': entries,
                'main_analysis': main_analysis
            })
        elif is_bad_from_bin:
            # File is marked as bad in bin directory but no main function found
            # Still create a positive entry
            bad_func_pattern = re.compile(r'void\s+(\w+_bad)\s*\(')
            try:
                with open(file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                found_bad_funcs = bad_func_pattern.findall(content)
                
                entries = []
                for bad_func in found_bad_funcs:
                    bad_func_line = find_function_line(file_path_str, bad_func)
                    entry = {
                        'file_path': file_path_str,
                        'type': 'positive',  # positive = has bug
                        'flaw': {
                            'function_name': bad_func,
                            'line': bad_func_line
                        },
                        'identified_from_bin': True
                    }
                    entries.append(entry)
                
                if entries:
                    results['summary']['positive_testcases'] += len(entries)
                    results['matched_files'].append({
                        'filename': filename,
                        'full_path': file_path_str,
                        'entries': entries,
                        'main_analysis': None,
                        'identified_from_bin': True
                    })
                else:
                    results['summary']['files_without_main'] += 1
                    results['unmatched_files'].append({
                        'filename': filename,
                        'full_path': file_path_str,
                        'reason': 'Marked as bad in bin directory but no _bad() function found'
                    })
            except Exception as e:
                results['summary']['files_without_main'] += 1
                results['unmatched_files'].append({
                    'filename': filename,
                    'full_path': file_path_str,
                    'reason': f'Marked as bad in bin directory but error reading file: {e}'
                })
        elif is_good_from_bin:
            # File is marked as good in bin directory but no main function found
            # Still create a negative entry
            good_func_pattern = re.compile(r'void\s+(\w+_good)\s*\(')
            try:
                with open(file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                found_good_funcs = good_func_pattern.findall(content)
                
                entries = []
                for good_func in found_good_funcs:
                    entry = {
                        'file_path': file_path_str,
                        'type': 'negative',  # negative = no bug
                        'flaw': None,
                        'good_function': good_func,
                        'identified_from_bin': True
                    }
                    entries.append(entry)
                
                if entries:
                    results['summary']['negative_testcases'] += len(entries)
                    results['matched_files'].append({
                        'filename': filename,
                        'full_path': file_path_str,
                        'entries': entries,
                        'main_analysis': None,
                        'identified_from_bin': True
                    })
                else:
                    results['summary']['files_without_main'] += 1
                    results['unmatched_files'].append({
                        'filename': filename,
                        'full_path': file_path_str,
                        'reason': 'Marked as good in bin directory but no _good() function found'
                    })
            except Exception as e:
                results['summary']['files_without_main'] += 1
                results['unmatched_files'].append({
                    'filename': filename,
                    'full_path': file_path_str,
                    'reason': f'Marked as good in bin directory but error reading file: {e}'
                })
        else:
            results['summary']['files_without_main'] += 1
            results['unmatched_files'].append({
                'filename': filename,
                'full_path': file_path_str,
                'reason': 'No main function found or no _good()/_bad() calls detected'
            })
    
    return results


def find_function_line(file_path, function_name):
    """
    Find the line number where a function is defined
    
    Args:
        file_path: Path to the source file
        function_name: Name of the function to find
        
    Returns:
        Line number (1-indexed) or None if not found
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return None
    
    # Pattern to match function definition
    function_pattern = re.compile(
        r'^\s*(?:static\s+|inline\s+)?'
        r'(?:\w+\s+)*'
        rf'{re.escape(function_name)}\s*'
        r'\([^)]*\)\s*'
        r'(?:\s*\{)?\s*$'
    )
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip comments and preprocessor directives
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('#'):
            continue
        
        # Check for function definition
        if function_pattern.match(line):
            return i
    
    return None


def calculate_precision_recall_from_main_analysis(ground_truth_results, predictions_file):
    """
    Calculate precision and recall using ground truth built from main function analysis
    
    Args:
        ground_truth_results: Results dictionary from main function analysis
        predictions_file: Path to JSON file with predictions
        
    Returns:
        Dictionary with precision, recall, and detailed metrics
    """
    # Load predictions
    if not os.path.exists(predictions_file):
        print(f"Warning: Predictions file {predictions_file} not found")
        return None
    
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        return None
    
    # Extract ground truth bugs (negative testcases with _bad() functions)
    ground_truth_bugs = set()
    ground_truth_by_file = {}  # filename -> set of function names
    
    for matched_file in ground_truth_results.get('matched_files', []):
        filename = normalize_file_path(matched_file.get('full_path', ''))
        
        for entry in matched_file.get('entries', []):
            if entry.get('type') == 'positive' and entry.get('flaw'):  # positive = has bug
                flaw = entry['flaw']
                function_name = flaw.get('function_name', '')
                
                # Use filename:function_name format for matching
                if function_name:
                    bug_id = f"{filename}:{function_name}"
                    ground_truth_bugs.add(bug_id)
                    # Also store by filename for tracking
                    if filename not in ground_truth_by_file:
                        ground_truth_by_file[filename] = set()
                    ground_truth_by_file[filename].add(function_name)
    
    # Extract predicted bugs
    predicted_bugs = set()
    predicted_by_file = {}  # filename -> set of function names
    for bug_type, bugs in predictions_data.items():
        for bug in bugs:
            file_path = bug.get('file', '')
            function_name = bug.get('function', '')
            
            if file_path and function_name:
                filename = normalize_file_path(file_path)
                bug_id = f"{filename}:{function_name}"
                predicted_bugs.add(bug_id)
                # Also store by filename for tracking
                if filename not in predicted_by_file:
                    predicted_by_file[filename] = set()
                predicted_by_file[filename].add(function_name)
    
    # Debug: Print some examples to see what we're comparing
    print(f"\nDebug: Sample ground truth bugs (first 5):")
    for i, bug_id in enumerate(list(ground_truth_bugs)[:5]):
        print(f"  {bug_id}")
    print(f"\nDebug: Sample predicted bugs (first 5):")
    for i, bug_id in enumerate(list(predicted_bugs)[:5]):
        print(f"  {bug_id}")
    
    # Calculate metrics - match based on filename + function name
    true_positives = ground_truth_bugs & predicted_bugs
    
    # False positives: predicted bugs that don't match any ground truth bug in that file+function
    false_positives = set()
    for filename in predicted_by_file:
        pred_functions = predicted_by_file[filename]
        if filename in ground_truth_by_file:
            gt_functions = ground_truth_by_file[filename]
            # False positives are predicted functions that don't match any ground truth function in this file
            unmatched_pred_functions = pred_functions - gt_functions
            for func in unmatched_pred_functions:
                false_positives.add(f"{filename}:{func}")
        else:
            # If file doesn't exist in ground truth, all predictions are false positives
            for func in pred_functions:
                false_positives.add(f"{filename}:{func}")
    
    # False negatives: ground truth bugs that don't match any predicted bug in that file+function
    false_negatives = set()
    for filename in ground_truth_by_file:
        gt_functions = ground_truth_by_file[filename]
        if filename in predicted_by_file:
            pred_functions = predicted_by_file[filename]
            # False negatives are ground truth functions that don't match any predicted function in this file
            unmatched_gt_functions = gt_functions - pred_functions
            for func in unmatched_gt_functions:
                false_negatives.add(f"{filename}:{func}")
        else:
            # If file doesn't exist in predictions, all ground truth are false negatives
            for func in gt_functions:
                false_negatives.add(f"{filename}:{func}")
    
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    # Calculate precision and recall
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'true_positives': tp_count,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ground_truth_total': len(ground_truth_bugs),
        'predicted_total': len(predicted_bugs),
        'true_positive_details': list(true_positives)[:20],  # Sample of TP
        'false_positive_details': list(false_positives)[:20],  # Sample of FP
        'false_negative_details': list(false_negatives)[:20]   # Sample of FN
    }
    
    return metrics


def calculate_precision_recall(ground_truth_results, predictions_file):
    """
    Calculate precision and recall by comparing ground truth with predictions
    
    Args:
        ground_truth_results: Results dictionary from manifest parsing
        predictions_file: Path to JSON file with predictions (memory_safety_bugs.json)
        
    Returns:
        Dictionary with precision, recall, and detailed metrics
    """
    # Load predictions
    if not os.path.exists(predictions_file):
        print(f"Warning: Predictions file {predictions_file} not found")
        return None
    
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        return None
    
    # Extract ground truth bugs (positive testcases with flaws - positive = has bug)
    # Match based on filename and function name (ignore line numbers)
    ground_truth_bugs = set()
    ground_truth_by_file = {}  # filename -> set of function names
    for matched_file in ground_truth_results.get('matched_files', []):
        for entry in matched_file.get('entries', []):
            if entry.get('type') == 'positive' and entry.get('flaw'):  # positive = has bug
                flaw = entry['flaw']
                filename = normalize_file_path(entry['file_path'])
                function_name = flaw.get('function_name', '')
                
                # Use filename:function_name format for matching
                if function_name:
                    bug_id = f"{filename}:{function_name}"
                    ground_truth_bugs.add(bug_id)
                    # Also store by filename for tracking
                    if filename not in ground_truth_by_file:
                        ground_truth_by_file[filename] = set()
                    ground_truth_by_file[filename].add(function_name)
    
    # Extract predicted bugs
    predicted_bugs = set()
    predicted_by_file = {}  # filename -> set of function names
    for bug_type, bugs in predictions_data.items():
        for bug in bugs:
            file_path = bug.get('file', '')
            function_name = bug.get('function', '')
            
            if file_path and function_name:
                filename = normalize_file_path(file_path)
                bug_id = f"{filename}:{function_name}"
                predicted_bugs.add(bug_id)
                # Also store by filename for tracking
                if filename not in predicted_by_file:
                    predicted_by_file[filename] = set()
                predicted_by_file[filename].add(function_name)
    
    # Debug: Print some examples to see what we're comparing
    print(f"\nDebug: Sample ground truth bugs (first 5):")
    for i, bug_id in enumerate(list(ground_truth_bugs)[:5]):
        print(f"  {bug_id}")
    print(f"\nDebug: Sample predicted bugs (first 5):")
    for i, bug_id in enumerate(list(predicted_bugs)[:5]):
        print(f"  {bug_id}")
    
    # Calculate metrics - match based on filename + function name
    true_positives = ground_truth_bugs & predicted_bugs
    
    # False positives: predicted bugs that don't match any ground truth bug in that file+function
    false_positives = set()
    for filename in predicted_by_file:
        pred_functions = predicted_by_file[filename]
        if filename in ground_truth_by_file:
            gt_functions = ground_truth_by_file[filename]
            # False positives are predicted functions that don't match any ground truth function in this file
            unmatched_pred_functions = pred_functions - gt_functions
            for func in unmatched_pred_functions:
                false_positives.add(f"{filename}:{func}")
        else:
            # If file doesn't exist in ground truth, all predictions are false positives
            for func in pred_functions:
                false_positives.add(f"{filename}:{func}")
    
    # False negatives: ground truth bugs that don't match any predicted bug in that file+function
    false_negatives = set()
    for filename in ground_truth_by_file:
        gt_functions = ground_truth_by_file[filename]
        if filename in predicted_by_file:
            pred_functions = predicted_by_file[filename]
            # False negatives are ground truth functions that don't match any predicted function in this file
            unmatched_gt_functions = gt_functions - pred_functions
            for func in unmatched_gt_functions:
                false_negatives.add(f"{filename}:{func}")
        else:
            # If file doesn't exist in predictions, all ground truth are false negatives
            for func in gt_functions:
                false_negatives.add(f"{filename}:{func}")
    
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    # Calculate precision and recall
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'true_positives': tp_count,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ground_truth_total': len(ground_truth_bugs),
        'predicted_total': len(predicted_bugs),
        'true_positive_details': list(true_positives)[:20],  # Sample of TP
        'false_positive_details': list(false_positives)[:20],  # Sample of FP
        'false_negative_details': list(false_negatives)[:20]   # Sample of FN
    }
    
    return metrics


def build_manifest_ground_truth(scan_dir, cwe_number, default_manifest):
    """
    Build ground truth using manifest.xml method
    
    Returns:
        Dictionary with ground truth results
    """
    # Find all CWE files
    cwe_files = find_cwe_files(scan_dir, cwe_number)
    
    # Parse manifest
    manifest_entries, testcase_files, total_cwe_testcases = parse_manifest(default_manifest, cwe_number)
    
    # Count total unique testcases for this CWE (from manifest)
    total_testcases = total_cwe_testcases
    
    # Collect results
    results = {
        'summary': {
            'total_cwe_files': len(cwe_files),
            'total_testcases': total_testcases,
            'matched_with_manifest': 0,
            'not_found_in_manifest': 0,
            'positive_testcases': 0,
            'negative_testcases': 0
        },
        'scan_directory': scan_dir,
        'cwe_number': cwe_number,
        'method': 'manifest_xml',
        'total_files_found': len(cwe_files),
        'matched_files': [],
        'unmatched_files': []
    }
    
    # Track processed testcases to avoid double counting
    processed_testcases = set()
    
    # Match files with manifest entries
    for file_path in cwe_files:
        filename = file_path.name
        file_path_str = str(file_path)
        
        if filename in manifest_entries:
            entries = manifest_entries[filename]
            
            # Process each entry for this file
            file_entries = []
            for entry_info in entries:
                # Get all files in the same testcase
                testcase = entry_info['testcase']
                all_files_in_testcase = testcase_files.get(testcase, [])
                
                entry_data = extract_testcase_data(
                    entry_info, 
                    source_file_path=file_path_str,
                    all_files_in_testcase=all_files_in_testcase
                )
                file_entries.append(entry_data)
                
                # Update summary counts (only once per testcase)
                testcase_id = id(testcase)
                if testcase_id not in processed_testcases:
                    processed_testcases.add(testcase_id)
                    # Count positive/negative based on whether any file has a flaw
                    # positive = has bug (has flaw), negative = no bug (no flaw)
                    has_flaw = any(f.get('has_flaw', False) for f in all_files_in_testcase)
                    if has_flaw:
                        results['summary']['positive_testcases'] += 1  # positive = has bug
                    else:
                        results['summary']['negative_testcases'] += 1  # negative = no bug
            
            results['matched_files'].append({
                'filename': filename,
                'full_path': file_path_str,
                'entries': file_entries
            })
            results['summary']['matched_with_manifest'] += 1
        else:
            results['unmatched_files'].append({
                'filename': filename,
                'full_path': file_path_str
            })
            results['summary']['not_found_in_manifest'] += 1
    
    return results


def main():
    # Get script directory to build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default directory and manifest path (relative to script location or current working directory)
    # Try to find juliet-test-suite-c relative to script, or use current working directory
    base_dir = script_dir
    juliet_dir = os.path.join(base_dir, "juliet-test-suite-c")
    if not os.path.exists(juliet_dir):
        # Try parent directory
        base_dir = os.path.dirname(script_dir)
        juliet_dir = os.path.join(base_dir, "juliet-test-suite-c")
    
    # Use configurable settings for default paths
    default_dir = os.path.join(juliet_dir, "testcases", DEFAULT_CWE_DIR)
    default_predictions = os.path.join(base_dir, "output", "juliet-test-suite-c", DEFAULT_CWE_DIR, DEFAULT_OUTPUT_SUBDIR, "memory_safety_bugs.json")
    default_manifest = os.path.join(juliet_dir, "manifest.xml")
    
    # Check for method flag: --main-analysis, --manifest, or --both (default is --both)
    use_both = True
    use_main_analysis = False
    use_manifest = False
    
    if '--main-analysis' in sys.argv:
        use_main_analysis = True
        use_both = False
        sys.argv.remove('--main-analysis')
    elif '--manifest' in sys.argv:
        use_manifest = True
        use_both = False
        sys.argv.remove('--manifest')
    elif '--both' in sys.argv:
        use_both = True
        sys.argv.remove('--both')
    
    # If no method specified, default to both
    if not use_main_analysis and not use_manifest:
        use_both = True
    
    # Get directory from command line or use default
    if len(sys.argv) > 1:
        scan_dir = sys.argv[1]
    else:
        scan_dir = default_dir
    
    # Get predictions file from command line or use default
    if len(sys.argv) > 3:
        predictions_file = sys.argv[3]
    else:
        predictions_file = default_predictions
    
    # Extract directory name and subdirectory for file naming
    dir_name = os.path.basename(os.path.normpath(scan_dir))
    predictions_path = Path(predictions_file)
    subdir_name = predictions_path.parent.name if predictions_path.parent else ""
    
    # Extract CWE number from directory path
    cwe_number = extract_cwe_number(scan_dir)
    if not cwe_number:
        print(f"Error: Could not extract CWE number from directory path: {scan_dir}")
        print("Please ensure the directory path contains a CWE number (e.g., CWE401, CWE415)")
        sys.exit(1)
    
    print(f"Detected CWE number: {cwe_number}")
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Ground truth file names (in results directory)
    manifest_gt_file = os.path.join(results_dir, f"{dir_name}_manifest_gt.json")
    main_gt_file = os.path.join(results_dir, f"{dir_name}_main_gt.json")
    
    # Build ground truth for manifest method
    manifest_results = None
    if use_both or use_manifest:
        print("\n=== Building Ground Truth: Manifest Method ===")
        if os.path.exists(manifest_gt_file):
            print(f"Loading existing manifest ground truth from {manifest_gt_file}...")
            try:
                with open(manifest_gt_file, 'r', encoding='utf-8') as f:
                    manifest_results = json.load(f)
                print(f"Loaded existing results from {manifest_gt_file}")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                print("Recalculating...")
                manifest_results = None
        
        if manifest_results is None:
            manifest_results = build_manifest_ground_truth(scan_dir, cwe_number, default_manifest)
            
            # Output JSON to file
            json_output = json.dumps(manifest_results, indent=2, ensure_ascii=False)
            with open(manifest_gt_file, 'w', encoding='utf-8') as f:
                f.write(json_output)
            
            print(f"Manifest ground truth written to {manifest_gt_file}")
            print(f"Summary: {manifest_results['summary']['total_testcases']} testcases, "
                  f"{manifest_results['summary']['matched_with_manifest']} matched files")
    
    # Build ground truth for main function analysis method
    main_results = None
    if use_both or use_main_analysis:
        print("\n=== Building Ground Truth: Main Function Analysis Method ===")
        if os.path.exists(main_gt_file):
            print(f"Loading existing main function ground truth from {main_gt_file}...")
            try:
                with open(main_gt_file, 'r', encoding='utf-8') as f:
                    main_results = json.load(f)
                print(f"Loaded existing results from {main_gt_file}")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                print("Recalculating...")
                main_results = None
        
        if main_results is None:
            # Pass juliet_dir so it can find bin/CWE{number}/bad directory
            main_results = build_ground_truth_from_main_functions(scan_dir, cwe_number, bin_dir=juliet_dir)
            
            # Output JSON to file
            json_output = json.dumps(main_results, indent=2, ensure_ascii=False)
            with open(main_gt_file, 'w', encoding='utf-8') as f:
                f.write(json_output)
            
            print(f"Main function ground truth written to {main_gt_file}")
            print(f"Summary: {main_results['summary']['positive_testcases']} positive testcases, "
                  f"{main_results['summary']['negative_testcases']} negative testcases, "
                  f"{main_results['summary']['files_with_main']} files with main function")
    
    # Calculate precision and recall for both methods
    print("\n=== Calculating Evaluation Metrics ===")
    all_metrics = {}
    
    if manifest_results:
        print("\nCalculating metrics for manifest method...")
        manifest_metrics = calculate_precision_recall(manifest_results, predictions_file)
        if manifest_metrics:
            all_metrics['manifest_method'] = manifest_metrics
    
    if main_results:
        print("\nCalculating metrics for main function analysis method...")
        main_metrics = calculate_precision_recall_from_main_analysis(main_results, predictions_file)
        if main_metrics:
            all_metrics['main_function_method'] = main_metrics
    
    # Save combined evaluation results to one JSON file
    if all_metrics:
        # Create combined results file (in results directory)
        if subdir_name:
            combined_metrics_file = os.path.join(results_dir, f"{dir_name}_{subdir_name}_combined_results.json")
        else:
            combined_metrics_file = os.path.join(results_dir, f"{dir_name}_combined_results.json")
        
        # Prepare combined results
        combined_results = {
            'evaluation_metrics': all_metrics,
            'ground_truth_files': {
                'manifest_method': manifest_gt_file if manifest_results else None,
                'main_function_method': main_gt_file if main_results else None
            },
            'predictions_file': predictions_file,
            'scan_directory': scan_dir,
            'cwe_number': cwe_number
        }
        
        # Write combined metrics to file
        json_output = json.dumps(combined_results, indent=2, ensure_ascii=False)
        with open(combined_metrics_file, 'w', encoding='utf-8') as f:
            f.write(json_output)
        
        print(f"\nCombined evaluation metrics written to {combined_metrics_file}")
        
        # Print summary for both methods
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)
        
        if 'manifest_method' in all_metrics:
            metrics = all_metrics['manifest_method']
            print("\n--- Manifest Method ---")
            print(f"True Positives (TP): {metrics['true_positives']}")
            print(f"False Positives (FP): {metrics['false_positives']}")
            print(f"False Negatives (FN): {metrics['false_negatives']}")
            print(f"Ground Truth Total: {metrics['ground_truth_total']}")
            print(f"Predicted Total: {metrics['predicted_total']}")
            print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            print(f"Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            print(f"F1 Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        
        if 'main_function_method' in all_metrics:
            metrics = all_metrics['main_function_method']
            print("\n--- Main Function Analysis Method ---")
            print(f"True Positives (TP): {metrics['true_positives']}")
            print(f"False Positives (FP): {metrics['false_positives']}")
            print(f"False Negatives (FN): {metrics['false_negatives']}")
            print(f"Ground Truth Total: {metrics['ground_truth_total']}")
            print(f"Predicted Total: {metrics['predicted_total']}")
            print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            print(f"Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            print(f"F1 Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        
        print("="*60)
    else:
        print("Could not calculate evaluation metrics for any method")


if __name__ == "__main__":
    main()

