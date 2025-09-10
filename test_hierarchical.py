#!/usr/bin/env python3
"""
Test script for hierarchical code matching
Run this to verify the hierarchical initialization logic works correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from exmed_bert.utils.clinvec_integration import test_hierarchical_matching

if __name__ == "__main__":
    print("Testing Hierarchical Code Matching...")
    test_hierarchical_matching()
    print("\nTest completed!")