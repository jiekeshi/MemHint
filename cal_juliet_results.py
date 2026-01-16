#!/usr/bin/env python3
"""
Script to scan for CWE files in a directory and find matching entries in manifest.xml
"""

import os
import sys
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

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
    """
    if not os.path.exists(manifest_path):
        return {}, {}
    
    try:
        tree = ET.parse(manifest_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return {}, {}
    
    if not cwe_number:
        print(f"Error: CWE number not provided")
        return {}, {}
    
    # Dictionary to store filename -> list of testcase entries
    manifest_entries = defaultdict(list)
    # Dictionary to store testcase -> all files in that testcase
    testcase_files = {}
    
    # Iterate through all testcase elements
    for testcase in root.findall('testcase'):
        # Get all file elements in this testcase
        file_elems = testcase.findall('file')
        
        # Store all files in this testcase
        testcase_files[testcase] = []
        
        for file_elem in file_elems:
            file_path_attr = file_elem.get('path')
            if file_path_attr and file_path_attr.startswith(cwe_number):
                filename = os.path.basename(file_path_attr)
                entry = {
                    'testcase': testcase,
                    'file_elem': file_elem,
                    'path': file_path_attr,
                    'has_flaw': file_elem.find('flaw') is not None
                }
                manifest_entries[filename].append(entry)
                testcase_files[testcase].append(entry)
    
    return manifest_entries, testcase_files


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
        'type': 'negative' if has_flaw else 'positive',
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
    
    # Extract ground truth bugs (negative testcases with flaws)
    # Match based on filename and function name (ignore line numbers)
    ground_truth_bugs = set()
    ground_truth_by_file = {}  # filename -> set of function names
    for matched_file in ground_truth_results.get('matched_files', []):
        for entry in matched_file.get('entries', []):
            if entry.get('type') == 'negative' and entry.get('flaw'):
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


def main():
    # Default directory and manifest path
    default_dir = "/home/huihuihuang/Hint/juliet-test-suite-c/testcases/CWE401_Memory_Leak"
    default_predictions = "/home/huihuihuang/Hint/output/juliet-test-suite-c/CWE401_Memory_Leak/output_codeql_only/memory_safety_bugs.json"
    default_manifest = "/home/huihuihuang/Hint/juliet-test-suite-c/manifest.xml"
    
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
    
    # Get output file from command line or use default
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Default output file name for ground truth: {dir_name}_gt.json
        output_file = f"{dir_name}_gt.json"
    
    # Check if output file exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Loading existing results...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Loaded existing results from {output_file}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            print("Recalculating...")
            results = None
    else:
        results = None
    
    # Calculate results if not loaded from file
    if results is None:
        # Extract CWE number from directory path
        cwe_number = extract_cwe_number(scan_dir)
        if not cwe_number:
            print(f"Error: Could not extract CWE number from directory path: {scan_dir}")
            print("Please ensure the directory path contains a CWE number (e.g., CWE401, CWE415)")
            sys.exit(1)
        
        print(f"Detected CWE number: {cwe_number}")
        
        # Find all CWE files
        cwe_files = find_cwe_files(scan_dir, cwe_number)
        
        # Parse manifest
        manifest_entries, testcase_files = parse_manifest(default_manifest, cwe_number)
        
        # Count total unique testcases for this CWE
        total_testcases = len(testcase_files)
        
        # Collect results
        results = {
            'scan_directory': scan_dir,
            'cwe_number': cwe_number,
            'total_files_found': len(cwe_files),
            'matched_files': [],
            'unmatched_files': [],
            'summary': {
                'total_cwe_files': len(cwe_files),
                'total_testcases': total_testcases,
                'matched_with_manifest': 0,
                'not_found_in_manifest': 0,
                'positive_testcases': 0,
                'negative_testcases': 0
            }
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
                        has_negative = any(f.get('has_flaw', False) for f in all_files_in_testcase)
                        if has_negative:
                            results['summary']['negative_testcases'] += 1
                        else:
                            results['summary']['positive_testcases'] += 1
                
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
        
        # Output JSON to file
        json_output = json.dumps(results, indent=2, ensure_ascii=False)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_output)
        
        print(f"Results written to {output_file}")
        print(f"Summary: {results['summary']['total_testcases']} testcases, "
              f"{results['summary']['matched_with_manifest']} matched files")
    
    # Calculate precision and recall
    print("\nCalculating precision and recall...")
    metrics = calculate_precision_recall(results, predictions_file)
    
    if metrics:
        # Create a separate file for evaluation metrics
        # Format: {dir_name}_{subdir_name}_results.json
        if subdir_name:
            metrics_file = f"{dir_name}_{subdir_name}_results.json"
        else:
            metrics_file = f"{dir_name}_results.json"
        
        # Add metadata to metrics
        metrics_with_meta = {
            'evaluation_metrics': metrics,
            'ground_truth_file': output_file,
            'predictions_file': predictions_file,
            'scan_directory': results.get('scan_directory', '')
        }
        
        # Write metrics to separate file
        json_output = json.dumps(metrics_with_meta, indent=2, ensure_ascii=False)
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(json_output)
        
        print(f"\nEvaluation metrics written to {metrics_file}")
        print("\n=== Evaluation Metrics ===")
        print(f"True Positives (TP): {metrics['true_positives']}")
        print(f"False Positives (FP): {metrics['false_positives']}")
        print(f"False Negatives (FN): {metrics['false_negatives']}")
        print(f"Ground Truth Total: {metrics['ground_truth_total']}")
        print(f"Predicted Total: {metrics['predicted_total']}")
        print(f"\nPrecision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1 Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    else:
        print("Could not calculate evaluation metrics")


if __name__ == "__main__":
    main()

