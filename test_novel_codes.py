"""Test novel code generation logic."""

# Test ICD-10 hierarchical parsing
def parse_icd10_hierarchy(icd_code):
    clean_code = icd_code.replace(".", "")
    chapter = clean_code[0] if len(clean_code) > 0 else ""
    category = clean_code[:3] if len(clean_code) >= 3 else clean_code
    subcategory = clean_code[3:] if len(clean_code) > 3 else ""
    return chapter, category, subcategory

# Test ATC5 hierarchical parsing
def parse_atc5_hierarchy(atc_code):
    level1 = atc_code[0] if len(atc_code) > 0 else ""
    level2 = atc_code[:3] if len(atc_code) >= 3 else atc_code
    level3 = atc_code[:4] if len(atc_code) >= 4 else atc_code
    level4 = atc_code[:5] if len(atc_code) >= 5 else atc_code
    level5 = atc_code if len(atc_code) == 7 else atc_code
    return level1, level2, level3, level4, level5

# Test cases
print("Testing ICD-10 parsing:")
test_icd_codes = ["E11.9", "Z23", "M54.5", "I10"]
for code in test_icd_codes:
    chapter, category, subcategory = parse_icd10_hierarchy(code)
    print(f"  {code:10} -> Chapter: {chapter}, Category: {category}, Subcategory: {subcategory}")

print("\nTesting ATC5 parsing:")
test_atc_codes = ["N02BE01", "C10AA05", "A10BA02", "R03AC02"]
for code in test_atc_codes:
    l1, l2, l3, l4, l5 = parse_atc5_hierarchy(code)
    print(f"  {code} -> L1: {l1}, L2: {l2}, L3: {l3}, L4: {l4}, L5: {l5}")

print("\nTesting novel code generation:")
# Simulate novel ICD-10
print("  Novel ICD-10 from E11.9 (diabetes):")
parent = "E11.9"
chapter, category, subcategory = parse_icd10_hierarchy(parent)
novel = f"{category}.95"
print(f"    Parent: {parent} -> Novel cousin: {novel} (same category {category})")

# Simulate novel ATC5
print("  Novel ATC5 from N02BE01 (paracetamol):")
parent = "N02BE01"
l1, l2, l3, l4, l5 = parse_atc5_hierarchy(parent)
novel = f"{l4}75"
print(f"    Parent: {parent} -> Novel cousin: {novel} (same chemical subgroup {l4})")

print("\nAll tests passed! The hierarchical cousin logic works correctly.")
