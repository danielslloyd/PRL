#!/usr/bin/env python3
"""
Simple syntax test for the ICD-10 pipeline changes
"""

import ast
import sys

def test_file_syntax(filepath):
    """Test if a Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        print(f"OK: {filepath}")
        return True
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return False

def main():
    """Test syntax of key files"""
    files_to_test = [
        'exmed_bert/data/encoding.py',
        'exmed_bert/data/patient.py',
    ]

    print("Testing Python syntax of modified files...")
    print("=" * 50)

    all_good = True
    for filepath in files_to_test:
        if not test_file_syntax(filepath):
            all_good = False

    print("=" * 50)
    if all_good:
        print("SUCCESS: All files have valid Python syntax!")
        print("The ICD-10 pipeline modifications are syntactically correct.")
    else:
        print("FAILED: Some files have syntax errors.")

    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)