#!/usr/bin/env python3
"""
Test script for ATC5 medication embeddings implementation

This script tests:
1. ATC5 code conversion from RxNorm
2. ATC hierarchy generation
3. Cousin-based initialization for novel ATC codes
"""

import torch
from exmed_bert.data.encoding import rxcui_to_atc5
from exmed_bert.utils.clinvec_integration import HierarchicalCodeMatcher


def test_rxcui_to_atc5():
    """Test RxNorm to ATC5 conversion"""
    print("=" * 60)
    print("TEST 1: RxNorm to ATC5 Conversion")
    print("=" * 60)

    # Example mapping
    rx_to_atc_map = {
        '860975': 'A01AA01',
        '197361': 'B01AC06',
        '123456': 'C09AA05',
    }

    # Test single conversion
    result = rxcui_to_atc5('860975', rx_to_atc_map)
    print(f"Single conversion: '860975' -> {result}")
    assert result == ['A01AA01'], f"Expected ['A01AA01'], got {result}"
    print("✓ Single conversion passed")

    # Test list conversion
    result = rxcui_to_atc5(['860975', '197361'], rx_to_atc_map)
    print(f"List conversion: ['860975', '197361'] -> {result}")
    assert result == ['A01AA01', 'B01AC06'], f"Expected ['A01AA01', 'B01AC06'], got {result}"
    print("✓ List conversion passed")

    # Test unknown code
    result = rxcui_to_atc5('999999', rx_to_atc_map)
    print(f"Unknown code: '999999' -> {result}")
    assert result == ['UNK'], f"Expected ['UNK'], got {result}"
    print("✓ Unknown code handling passed")

    print()


def test_atc_hierarchy():
    """Test ATC hierarchy generation"""
    print("=" * 60)
    print("TEST 2: ATC Hierarchy Generation")
    print("=" * 60)

    matcher = HierarchicalCodeMatcher()

    # Test ATC5 code (7 chars)
    code = 'A01AA01'
    hierarchy = matcher.get_code_hierarchy(code)
    expected = ['A01AA01', 'A01AA', 'A01A', 'A01']
    print(f"ATC5 hierarchy for '{code}':")
    print(f"  Result:   {hierarchy}")
    print(f"  Expected: {expected}")
    assert hierarchy == expected, f"Expected {expected}, got {hierarchy}"
    print("✓ ATC5 hierarchy passed")

    # Test ATC4 code (5 chars)
    code = 'A01AA'
    hierarchy = matcher.get_code_hierarchy(code)
    expected = ['A01AA', 'A01A', 'A01']
    print(f"\nATC4 hierarchy for '{code}':")
    print(f"  Result:   {hierarchy}")
    print(f"  Expected: {expected}")
    assert hierarchy == expected, f"Expected {expected}, got {hierarchy}"
    print("✓ ATC4 hierarchy passed")

    # Test ATC3 code (4 chars)
    code = 'A01A'
    hierarchy = matcher.get_code_hierarchy(code)
    expected = ['A01A', 'A01']
    print(f"\nATC3 hierarchy for '{code}':")
    print(f"  Result:   {hierarchy}")
    print(f"  Expected: {expected}")
    assert hierarchy == expected, f"Expected {expected}, got {hierarchy}"
    print("✓ ATC3 hierarchy passed")

    # Test pattern recognition
    print(f"\nPattern recognition tests:")
    print(f"  'A01AA01' is ATC: {matcher.is_atc_code('A01AA01')}")
    assert matcher.is_atc_code('A01AA01') == True
    print(f"  'E11.65' is ATC: {matcher.is_atc_code('E11.65')}")
    assert matcher.is_atc_code('E11.65') == False
    print(f"  '250.01' is ATC: {matcher.is_atc_code('250.01')}")
    assert matcher.is_atc_code('250.01') == False
    print("✓ Pattern recognition passed")

    print()


def test_cousin_finding():
    """Test cousin-based initialization"""
    print("=" * 60)
    print("TEST 3: Cousin-Based Initialization")
    print("=" * 60)

    matcher = HierarchicalCodeMatcher()

    # Create mock embeddings for cousin codes
    available_embeddings = {
        'A01AA02': torch.tensor([1.0, 2.0, 3.0]),
        'A01AA03': torch.tensor([2.0, 3.0, 4.0]),
        'A01AA04': torch.tensor([3.0, 4.0, 5.0]),
        'B01AC01': torch.tensor([10.0, 20.0, 30.0]),
        'A01AB01': torch.tensor([5.0, 6.0, 7.0]),  # Different parent (A01AB)
    }

    # Test finding cousins for A01AA01
    novel_code = 'A01AA01'
    cousin_embedding = matcher.find_cousin_average_embedding(novel_code, available_embeddings)

    print(f"Finding cousins for '{novel_code}':")
    print(f"  Available embeddings:")
    for code in available_embeddings.keys():
        print(f"    - {code}")

    if cousin_embedding is not None:
        expected_avg = torch.tensor([2.0, 3.0, 4.0])  # Average of A01AA02, A01AA03, A01AA04
        print(f"  Cousin average: {cousin_embedding}")
        print(f"  Expected:       {expected_avg}")
        assert torch.allclose(cousin_embedding, expected_avg), \
            f"Expected {expected_avg}, got {cousin_embedding}"
        print("✓ Cousin averaging passed")
    else:
        raise AssertionError("Expected to find cousin embeddings, but got None")

    # Test with no cousins available
    novel_code = 'C09AA01'
    cousin_embedding = matcher.find_cousin_average_embedding(novel_code, available_embeddings)
    print(f"\nFinding cousins for '{novel_code}' (should find none):")
    print(f"  Result: {cousin_embedding}")
    assert cousin_embedding is None, "Expected None when no cousins available"
    print("✓ No cousins handling passed")

    # Test parent fallback still works
    available_with_parent = {
        **available_embeddings,
        'C09AA': torch.tensor([100.0, 200.0, 300.0]),  # Parent code
    }
    parent_embedding = matcher.find_best_parent_embedding(novel_code, available_with_parent)
    print(f"\nFinding parent for '{novel_code}':")
    print(f"  Parent embedding: {parent_embedding}")
    assert parent_embedding is not None, "Expected to find parent embedding"
    print("✓ Parent fallback passed")

    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ATC5 MEDICATION EMBEDDINGS - IMPLEMENTATION TEST")
    print("=" * 60 + "\n")

    try:
        test_rxcui_to_atc5()
        test_atc_hierarchy()
        test_cousin_finding()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nImplementation Summary:")
        print("1. ✓ RxNorm codes are converted to ATC5 (7-char codes)")
        print("2. ✓ ATC hierarchy correctly generated (A01AA01 -> A01AA -> A01A -> A01)")
        print("3. ✓ Cousin-based initialization works (averages siblings with same parent)")
        print("4. ✓ Parent-based fallback available when no cousins exist")
        print("\nNext Steps:")
        print("- Ensure ClinVec data includes ATC vocabulary")
        print("- Update config.yaml to include 'atc' in vocab_types (already done)")
        print("- Run full training with ClinVec integration")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
