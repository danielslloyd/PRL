# BACKUP: Original ICD-to-Phewas mapping from explain.ipynb
# This is the original mapping that converts ICD codes to Phewas codes
# Created as backup before modification for ClinVec integration

icd_to_phewas_map_original = {
    'I10': '401.1', 'E11.9': '250', 'Z51.11': '008', 'K21.0': '530.11', 'M17.9': '715.2',
    'E78.5': '272.1', 'N18.3': '585.3',
    'A00': '800', 'B00': '900', 'C00': '1000', 'D00': '1100', 'E00': '1200',
    'F00': '1300', 'G00': '1400', 'H00': '1500', 'J00': '1600', 'K00': '1700',
    'L00': '1800', 'M00': '1900', 'N00': '2000'
}

# To restore original behavior, replace the mapping in explain.ipynb with this dictionary