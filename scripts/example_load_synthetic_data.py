"""
Example script showing how to load and use the synthetic patient data.

This demonstrates:
1. Loading the synthetic JSON data
2. Creating vocabulary dictionaries
3. Converting to Patient objects
4. Analyzing novel codes
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np


def load_synthetic_patients(json_path: str):
    """Load synthetic patient data from JSON file."""
    with open(json_path, 'r') as f:
        patients = json.load(f)
    return patients


def analyze_novel_codes(patients):
    """
    Analyze novel codes in the dataset.

    Novel codes are those that don't match standard patterns or have
    unusual suffixes that suggest they were generated.
    """
    # Collect all codes
    all_icd_codes = []
    all_atc_codes = []

    for patient in patients:
        all_icd_codes.extend(patient['diagnoses'])
        all_atc_codes.extend(patient['drugs'])

    # Count code frequencies
    icd_freq = defaultdict(int)
    atc_freq = defaultdict(int)

    for code in all_icd_codes:
        icd_freq[code] += 1

    for code in all_atc_codes:
        atc_freq[code] += 1

    # Identify potential novel codes (those with very low frequency)
    print("=== Code Analysis ===\n")

    print(f"Total unique ICD-10 codes: {len(icd_freq)}")
    print(f"Total unique ATC5 codes: {len(atc_freq)}")

    # Find rare codes (likely novel)
    rare_threshold = 5
    rare_icd = [code for code, freq in icd_freq.items() if freq < rare_threshold]
    rare_atc = [code for code, freq in atc_freq.items() if freq < rare_threshold]

    print(f"\nRare ICD-10 codes (freq < {rare_threshold}): {len(rare_icd)}")
    if rare_icd:
        print("  Examples:")
        for code in sorted(rare_icd)[:10]:
            print(f"    {code} (count: {icd_freq[code]})")

    print(f"\nRare ATC5 codes (freq < {rare_threshold}): {len(rare_atc)}")
    if rare_atc:
        print("  Examples:")
        for code in sorted(rare_atc)[:10]:
            print(f"    {code} (count: {atc_freq[code]})")


def analyze_hierarchical_structure(patients):
    """Analyze hierarchical structure of codes."""
    print("\n=== Hierarchical Structure Analysis ===\n")

    # Collect all codes
    all_icd_codes = set()
    all_atc_codes = set()

    for patient in patients:
        all_icd_codes.update(patient['diagnoses'])
        all_atc_codes.update(patient['drugs'])

    # Analyze ICD-10 hierarchy
    icd_chapters = defaultdict(set)
    icd_categories = defaultdict(set)

    for code in all_icd_codes:
        clean_code = code.replace(".", "")
        chapter = clean_code[0] if len(clean_code) > 0 else ""
        category = clean_code[:3] if len(clean_code) >= 3 else clean_code

        icd_chapters[chapter].add(code)
        icd_categories[category].add(code)

    print(f"ICD-10 Chapters represented: {len(icd_chapters)}")
    print(f"ICD-10 Categories represented: {len(icd_categories)}")

    # Show categories with multiple codes (including potential novel cousins)
    categories_with_cousins = {cat: codes for cat, codes in icd_categories.items() if len(codes) > 1}
    print(f"\nICD-10 Categories with multiple codes (potential novel cousins): {len(categories_with_cousins)}")
    print("  Examples:")
    for cat, codes in sorted(categories_with_cousins.items())[:5]:
        print(f"    {cat}: {sorted(codes)}")

    # Analyze ATC hierarchy
    atc_level1 = defaultdict(set)
    atc_level4 = defaultdict(set)

    for code in all_atc_codes:
        if len(code) == 7:
            level1 = code[0]
            level4 = code[:5]

            atc_level1[level1].add(code)
            atc_level4[level4].add(code)

    print(f"\nATC Level 1 groups represented: {len(atc_level1)}")
    print(f"ATC Level 4 subgroups represented: {len(atc_level4)}")

    # Show subgroups with multiple codes (including potential novel cousins)
    subgroups_with_cousins = {subg: codes for subg, codes in atc_level4.items() if len(codes) > 1}
    print(f"\nATC Level 4 subgroups with multiple codes (potential novel cousins): {len(subgroups_with_cousins)}")
    print("  Examples:")
    for subg, codes in sorted(subgroups_with_cousins.items())[:5]:
        print(f"    {subg}: {sorted(codes)}")


def analyze_patient_demographics(patients):
    """Analyze patient demographics."""
    print("\n=== Patient Demographics ===\n")

    # Age distribution
    current_year = 2025  # Assuming data generation reference year
    ages = [current_year - p['birth_year'] for p in patients]

    print(f"Age Statistics:")
    print(f"  Mean: {np.mean(ages):.1f} years")
    print(f"  Median: {np.median(ages):.1f} years")
    print(f"  Std Dev: {np.std(ages):.1f} years")
    print(f"  Range: {min(ages)} - {max(ages)} years")

    # Age brackets
    age_brackets = {
        "0-17": sum(1 for age in ages if age <= 17),
        "18-34": sum(1 for age in ages if 18 <= age <= 34),
        "35-54": sum(1 for age in ages if 35 <= age <= 54),
        "55-74": sum(1 for age in ages if 55 <= age <= 74),
        "75+": sum(1 for age in ages if age >= 75),
    }

    print(f"\nAge Brackets:")
    for bracket, count in age_brackets.items():
        print(f"  {bracket}: {count} ({100*count/len(patients):.1f}%)")

    # Sex distribution
    sex_counts = {"MALE": 0, "FEMALE": 0}
    for p in patients:
        sex_counts[p["sex"]] += 1

    print(f"\nSex Distribution:")
    print(f"  Male: {sex_counts['MALE']} ({100*sex_counts['MALE']/len(patients):.1f}%)")
    print(f"  Female: {sex_counts['FEMALE']} ({100*sex_counts['FEMALE']/len(patients):.1f}%)")

    # State distribution
    state_counts = defaultdict(int)
    for p in patients:
        state_counts[p["patient_state"]] += 1

    print(f"\nTop 10 States:")
    for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {state}: {count} ({100*count/len(patients):.1f}%)")


def analyze_temporal_patterns(patients):
    """Analyze temporal patterns in the data."""
    print("\n=== Temporal Patterns ===\n")

    # Parse dates and analyze distribution
    all_dates = []
    for patient in patients:
        for date_str in patient['diagnosis_dates'] + patient['prescription_dates']:
            date_obj = datetime.strptime(date_str, "%Y%m%d").date()
            all_dates.append(date_obj)

    if all_dates:
        all_dates = sorted(all_dates)
        print(f"Date Range:")
        print(f"  Earliest: {all_dates[0]}")
        print(f"  Latest: {all_dates[-1]}")
        print(f"  Total events: {len(all_dates)}")

        # Events per year
        year_counts = defaultdict(int)
        for date_obj in all_dates:
            year_counts[date_obj.year] += 1

        print(f"\nEvents per Year:")
        for year in sorted(year_counts.keys()):
            print(f"  {year}: {year_counts[year]} events")


def main():
    """Main analysis function."""
    # Path to synthetic data
    json_path = "data/synthetic_patients.json"

    if not Path(json_path).exists():
        print(f"Error: {json_path} not found!")
        print("Please run: python scripts/generate_synthetic_patients.py")
        return

    print(f"Loading synthetic patients from {json_path}...")
    patients = load_synthetic_patients(json_path)
    print(f"Loaded {len(patients)} patients\n")

    # Run analyses
    analyze_patient_demographics(patients)
    analyze_temporal_patterns(patients)
    analyze_novel_codes(patients)
    analyze_hierarchical_structure(patients)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
