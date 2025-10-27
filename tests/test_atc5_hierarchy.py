"""
Test script to verify ATC5 hierarchical code matching functionality
"""
import sys
sys.path.insert(0, '.')

from exmed_bert.utils.clinvec_integration import HierarchicalCodeMatcher
import torch

def test_atc5_detection():
    """Test ATC5 code detection"""
    print("=" * 60)
    print("TEST 1: ATC5 Code Detection")
    print("=" * 60)

    matcher = HierarchicalCodeMatcher()

    test_cases = [
        ("N02BE01", True, "Valid ATC5 code"),
        ("A10BA02", True, "Valid ATC5 code"),
        ("C09AA05", True, "Valid ATC5 code"),
        ("E11.65", False, "ICD-10 code"),
        ("N02BE", False, "ATC Level 4 (too short)"),
        ("N02BE001", False, "Invalid (too long)"),
    ]

    passed = 0
    failed = 0

    for code, expected, description in test_cases:
        result = matcher.is_atc5_code(code)
        status = "[PASS]" if result == expected else "[FAIL]"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status}: {code:15} -> {result:5} (expected: {expected:5}) - {description}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_atc5_hierarchy():
    """Test ATC5 hierarchical code generation"""
    print("\n" + "=" * 60)
    print("TEST 2: ATC5 Hierarchy Generation")
    print("=" * 60)

    matcher = HierarchicalCodeMatcher()

    test_cases = [
        ("N02BE01", ["N02BE01", "N02BE", "N02B", "N02", "N"]),
        ("A10BA02", ["A10BA02", "A10BA", "A10B", "A10", "A"]),
        ("C09AA05", ["C09AA05", "C09AA", "C09A", "C09", "C"]),
    ]

    passed = 0
    failed = 0

    for code, expected in test_cases:
        result = matcher.get_code_hierarchy(code)

        if result == expected:
            status = "[PASS]"
            passed += 1
        else:
            status = "[FAIL]"
            failed += 1

        print(f"\n{status}: {code}")
        print(f"  Expected: {' -> '.join(expected)}")
        print(f"  Got:      {' -> '.join(result)}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_parent_embedding_matching():
    """Test finding parent embeddings for novel ATC5 codes"""
    print("\n" + "=" * 60)
    print("TEST 3: Parent Embedding Matching")
    print("=" * 60)

    matcher = HierarchicalCodeMatcher()

    # Create mock embeddings for parent codes
    mock_embeddings = {
        "N02BE01": torch.randn(64),  # Level 5: Paracetamol
        "N02BE": torch.randn(64),     # Level 4: Anilides
        "N02B": torch.randn(64),      # Level 3: Other analgesics
        "A10BA": torch.randn(64),     # Level 4: Biguanides
    }

    test_cases = [
        ("N02BE75", "N02BE", "Novel anilide should find N02BE parent"),
        ("N02BE99", "N02BE", "Another novel anilide should find N02BE parent"),
        ("N02BA01", "N02B", "Novel salicylic acid should find N02B parent (no N02BA available)"),
        ("A10BA99", "A10BA", "Novel biguanide should find A10BA parent"),
        ("Z99ZZ99", None, "Completely novel code should find no parent"),
    ]

    passed = 0
    failed = 0

    for novel_code, expected_parent, description in test_cases:
        result = matcher.find_best_parent_embedding(novel_code, mock_embeddings)

        if expected_parent is None:
            success = result is None
        else:
            success = result is not None and torch.equal(result, mock_embeddings[expected_parent])

        status = "[PASS]" if success else "[FAIL]"

        if success:
            passed += 1
        else:
            failed += 1

        parent_found = "None" if result is None else "Found"
        print(f"{status}: {novel_code} -> {parent_found} (expected: {expected_parent or 'None'})")
        print(f"  {description}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_mixed_code_types():
    """Test that different code types are handled correctly"""
    print("\n" + "=" * 60)
    print("TEST 4: Mixed Code Types")
    print("=" * 60)

    matcher = HierarchicalCodeMatcher()

    test_cases = [
        ("N02BE01", "ATC5"),
        ("E11.65", "ICD-10"),
        ("UNKNOWN", "Unknown"),
    ]

    passed = 0
    failed = 0

    for code, expected_type in test_cases:
        if expected_type == "ATC5":
            result = matcher.is_atc5_code(code)
        elif expected_type == "ICD-10":
            result = matcher.is_icd10_code(code)
        else:
            result = not (matcher.is_atc5_code(code) or
                         matcher.is_icd10_code(code))

        status = "[PASS]" if result else "[FAIL]"

        if result:
            passed += 1
        else:
            failed += 1

        hierarchy = matcher.get_code_hierarchy(code)
        print(f"{status}: {code:15} detected as {expected_type:10} - Hierarchy: {' -> '.join(hierarchy)}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all tests"""
    print("\n")
    print("=" * 60)
    print(" " * 10 + "ATC5 Hierarchical Matching Tests")
    print("=" * 60)

    all_passed = True

    all_passed &= test_atc5_detection()
    all_passed &= test_atc5_hierarchy()
    all_passed &= test_parent_embedding_matching()
    all_passed &= test_mixed_code_types()

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[FAILURE] SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
