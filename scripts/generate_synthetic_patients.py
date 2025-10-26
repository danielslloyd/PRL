"""
Generate synthetic patient data with realistic ICD-10 and ATC5 codes.

This script generates N patients with:
- Most common ICD-10 diagnosis codes
- Most common ATC5 drug codes
- 1% novel codes that have hierarchical cousins
- US population-based demographics (age, sex, state)
- Dates of care within the last 5 years
"""

import argparse
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np


# Most common ICD-10 codes (based on US healthcare utilization)
COMMON_ICD10_CODES = [
    "Z23",      # Encounter for immunization
    "E78.5",    # Hyperlipidemia
    "I10",      # Essential hypertension
    "E11.9",    # Type 2 diabetes mellitus
    "Z00.00",   # General adult medical examination
    "M79.3",    # Unspecified myalgia
    "E66.9",    # Obesity, unspecified
    "M54.5",    # Low back pain
    "K21.9",    # Gastro-esophageal reflux disease
    "J06.9",    # Acute upper respiratory infection
    "F41.9",    # Anxiety disorder, unspecified
    "R05",      # Cough
    "F33.1",    # Major depressive disorder, recurrent
    "J44.9",    # Chronic obstructive pulmonary disease
    "M25.551",  # Pain in right hip
    "N39.0",    # Urinary tract infection
    "E03.9",    # Hypothyroidism, unspecified
    "M19.90",   # Unspecified osteoarthritis
    "F32.9",    # Major depressive disorder, single episode
    "R51",      # Headache
    "E78.0",    # Pure hypercholesterolemia
    "J02.9",    # Acute pharyngitis
    "I25.10",   # Atherosclerotic heart disease
    "K76.0",    # Fatty liver disease
    "M17.9",    # Osteoarthritis of knee
    "G47.00",   # Insomnia, unspecified
    "E55.9",    # Vitamin D deficiency
    "K59.00",   # Constipation, unspecified
    "N18.3",    # Chronic kidney disease, stage 3
    "I73.9",    # Peripheral vascular disease
    "J45.909",  # Asthma, unspecified
    "R42",      # Dizziness and giddiness
    "E04.9",    # Nontoxic goiter
    "M81.0",    # Age-related osteoporosis
    "F17.210",  # Nicotine dependence, cigarettes
    "R10.9",    # Unspecified abdominal pain
    "H52.4",    # Presbyopia
    "N40.0",    # Benign prostatic hyperplasia
    "E28.2",    # Polycystic ovarian syndrome
    "J30.9",    # Allergic rhinitis
    "M16.9",    # Osteoarthritis of hip
    "I48.91",   # Atrial fibrillation
    "H35.31",   # Age-related macular degeneration
    "L70.0",    # Acne vulgaris
    "R53.83",   # Fatigue
    "G43.909",  # Migraine
    "K58.9",    # Irritable bowel syndrome
    "I50.9",    # Heart failure, unspecified
    "F10.20",   # Alcohol dependence
    "E66.01",   # Morbid obesity
]

# Most common ATC5 codes (5th level - chemical substance)
# Format: ATC5 code
COMMON_ATC5_CODES = [
    "N02BE01",  # Paracetamol (Acetaminophen)
    "M01AE01",  # Ibuprofen
    "C09AA05",  # Ramipril
    "C10AA05",  # Atorvastatin
    "A02BC05",  # Esomeprazole
    "N06AB06",  # Sertraline
    "C07AB07",  # Bisoprolol
    "N06AX21",  # Duloxetine
    "A10BA02",  # Metformin
    "C09CA01",  # Losartan
    "N05BA01",  # Diazepam
    "R03AC02",  # Salbutamol
    "C08CA01",  # Amlodipine
    "C03CA01",  # Furosemide
    "B01AC06",  # Acetylsalicylic acid
    "C01DA14",  # Isosorbide mononitrate
    "H03AA01",  # Levothyroxine
    "A11CC05",  # Cholecalciferol (Vitamin D3)
    "N02AX02",  # Tramadol
    "N05CF02",  # Zolpidem
    "R06AE07",  # Cetirizine
    "A07EA06",  # Loperamide
    "J01CA04",  # Amoxicillin
    "C01BD01",  # Amiodarone
    "N06AB04",  # Citalopram
    "M05BA04",  # Alendronic acid
    "N03AX16",  # Pregabalin
    "N06AX11",  # Mirtazapine
    "C09AA02",  # Enalapril
    "A02BC01",  # Omeprazole
    "N05AH03",  # Olanzapine
    "C10AA01",  # Simvastatin
    "N05BA12",  # Alprazolam
    "C09CA03",  # Valsartan
    "A10AB01",  # Insulin (human)
    "N03AX12",  # Gabapentin
    "C01AA05",  # Digoxin
    "N02AA01",  # Morphine
    "N06AB03",  # Fluoxetine
    "R03BA05",  # Fluticasone
    "C07AB02",  # Metoprolol
    "N05CD07",  # Temazepam
    "A06AB02",  # Bisacodyl
    "B03AA07",  # Ferrous sulfate
    "J01FA10",  # Azithromycin
    "C03AA03",  # Hydrochlorothiazide
    "N02AJ13",  # Oxycodone
    "G04CA02",  # Tamsulosin
    "N06AB05",  # Paroxetine
    "A06AD11",  # Lactulose
]

# US states with approximate population weights (2020 Census)
US_STATES_WITH_WEIGHTS = [
    ("CA", 0.119),  # California
    ("TX", 0.087),  # Texas
    ("FL", 0.065),  # Florida
    ("NY", 0.058),  # New York
    ("PA", 0.039),  # Pennsylvania
    ("IL", 0.038),  # Illinois
    ("OH", 0.035),  # Ohio
    ("GA", 0.032),  # Georgia
    ("NC", 0.031),  # North Carolina
    ("MI", 0.030),  # Michigan
    ("NJ", 0.027),  # New Jersey
    ("VA", 0.026),  # Virginia
    ("WA", 0.023),  # Washington
    ("AZ", 0.022),  # Arizona
    ("MA", 0.021),  # Massachusetts
    ("TN", 0.020),  # Tennessee
    ("IN", 0.020),  # Indiana
    ("MO", 0.018),  # Missouri
    ("MD", 0.018),  # Maryland
    ("WI", 0.018),  # Wisconsin
    ("CO", 0.017),  # Colorado
    ("MN", 0.017),  # Minnesota
    ("SC", 0.015),  # South Carolina
    ("AL", 0.015),  # Alabama
    ("LA", 0.014),  # Louisiana
    ("KY", 0.013),  # Kentucky
    ("OR", 0.013),  # Oregon
    ("OK", 0.012),  # Oklahoma
    ("CT", 0.011),  # Connecticut
    ("UT", 0.010),  # Utah
    ("IA", 0.009),  # Iowa
    ("NV", 0.009),  # Nevada
    ("AR", 0.009),  # Arkansas
    ("MS", 0.009),  # Mississippi
    ("KS", 0.009),  # Kansas
    ("NM", 0.006),  # New Mexico
    ("NE", 0.006),  # Nebraska
    ("WV", 0.005),  # West Virginia
    ("ID", 0.005),  # Idaho
    ("HI", 0.004),  # Hawaii
    ("NH", 0.004),  # New Hampshire
    ("ME", 0.004),  # Maine
    ("MT", 0.003),  # Montana
    ("RI", 0.003),  # Rhode Island
    ("DE", 0.003),  # Delaware
    ("SD", 0.003),  # South Dakota
    ("ND", 0.002),  # North Dakota
    ("AK", 0.002),  # Alaska
    ("VT", 0.002),  # Vermont
    ("WY", 0.002),  # Wyoming
]


# Cache for generated novel codes to ensure consistency
_NOVEL_ICD10_CACHE: Set[str] = set()
_NOVEL_ATC5_CACHE: Set[str] = set()


def parse_icd10_hierarchy(icd_code: str) -> Tuple[str, str, str]:
    """
    Parse ICD-10 code into hierarchical components.

    ICD-10 structure:
    - Letter (1st char): Chapter (e.g., E = Endocrine)
    - Numbers (2-3 chars): Category (e.g., E11 = Type 2 diabetes)
    - Decimal + numbers: Subcategory (e.g., E11.9 = unspecified)

    Args:
        icd_code: ICD-10 code (e.g., "E11.9")

    Returns:
        Tuple of (chapter, category, subcategory)
    """
    # Remove dots for parsing
    clean_code = icd_code.replace(".", "")

    # Chapter is first letter
    chapter = clean_code[0] if len(clean_code) > 0 else ""

    # Category is first 3 characters (letter + 2 digits)
    category = clean_code[:3] if len(clean_code) >= 3 else clean_code

    # Subcategory is everything after first 3 chars
    subcategory = clean_code[3:] if len(clean_code) > 3 else ""

    return chapter, category, subcategory


def generate_novel_icd10(parent_code: str) -> str:
    """
    Generate a novel ICD-10 code that is a hierarchical cousin of parent_code.

    Strategy:
    - Keep same chapter (letter) and category (first 3 chars)
    - Generate new subcategory suffix

    Args:
        parent_code: Existing ICD-10 code to use as template

    Returns:
        Novel ICD-10 code
    """
    chapter, category, subcategory = parse_icd10_hierarchy(parent_code)

    # Generate new subcategory that doesn't exist in our common codes
    attempts = 0
    while attempts < 100:
        # For codes with subcategories, modify the subcategory
        if subcategory:
            # Generate random subcategory (1-3 digits)
            new_sub_len = len(subcategory)
            if new_sub_len == 1:
                new_subcategory = str(random.randint(0, 9))
            elif new_sub_len == 2:
                new_subcategory = f"{random.randint(0, 99):02d}"
            else:
                new_subcategory = f"{random.randint(0, 999):03d}"

            # Format as ICD-10 (with decimal after category)
            if len(category) == 3:
                novel_code = f"{category}.{new_subcategory}"
            else:
                novel_code = f"{category}{new_subcategory}"
        else:
            # For codes without subcategories, add one
            new_subcategory = str(random.randint(1, 99))
            novel_code = f"{category}.{new_subcategory}"

        # Check if this novel code already exists or was already generated
        if novel_code not in COMMON_ICD10_CODES and novel_code not in _NOVEL_ICD10_CACHE:
            _NOVEL_ICD10_CACHE.add(novel_code)
            return novel_code

        attempts += 1

    # Fallback: add NOVEL prefix
    return f"NOVEL_{parent_code}"


def parse_atc5_hierarchy(atc_code: str) -> Tuple[str, str, str, str, str]:
    """
    Parse ATC5 code into hierarchical components.

    ATC structure (7 characters):
    - Level 1 (1 char): Anatomical main group (e.g., N = Nervous system)
    - Level 2 (2 chars): Therapeutic subgroup (e.g., N02 = Analgesics)
    - Level 3 (3 chars): Pharmacological subgroup (e.g., N02B = Other analgesics)
    - Level 4 (4 chars): Chemical subgroup (e.g., N02BE = Anilides)
    - Level 5 (7 chars): Chemical substance (e.g., N02BE01 = Paracetamol)

    Args:
        atc_code: ATC5 code (e.g., "N02BE01")

    Returns:
        Tuple of (level1, level2, level3, level4, level5)
    """
    level1 = atc_code[0] if len(atc_code) > 0 else ""
    level2 = atc_code[:3] if len(atc_code) >= 3 else atc_code
    level3 = atc_code[:4] if len(atc_code) >= 4 else atc_code
    level4 = atc_code[:5] if len(atc_code) >= 5 else atc_code
    level5 = atc_code if len(atc_code) == 7 else atc_code

    return level1, level2, level3, level4, level5


def generate_novel_atc5(parent_code: str) -> str:
    """
    Generate a novel ATC5 code that is a hierarchical cousin of parent_code.

    Strategy:
    - Keep same level 1-4 (first 5 characters)
    - Generate new level 5 suffix (last 2 digits)

    Args:
        parent_code: Existing ATC5 code to use as template

    Returns:
        Novel ATC5 code
    """
    if len(parent_code) != 7:
        # Invalid format, return with NOVEL prefix
        return f"NOVEL_{parent_code}"

    level1, level2, level3, level4, level5 = parse_atc5_hierarchy(parent_code)

    # Generate new chemical substance code (last 2 digits)
    attempts = 0
    while attempts < 100:
        new_suffix = f"{random.randint(50, 99):02d}"  # Use 50-99 to avoid common codes
        novel_code = f"{level4}{new_suffix}"

        # Check if this novel code already exists or was already generated
        if novel_code not in COMMON_ATC5_CODES and novel_code not in _NOVEL_ATC5_CACHE:
            _NOVEL_ATC5_CACHE.add(novel_code)
            return novel_code

        attempts += 1

    # Fallback: add NOVEL prefix
    return f"NOVEL_{parent_code}"


def generate_age_distribution() -> int:
    """
    Generate age following US population distribution.

    US age distribution (approximate):
    - 0-17: 22%
    - 18-34: 21%
    - 35-54: 25%
    - 55-74: 22%
    - 75+: 10%

    Returns birth year (assuming current year is 2025 for the 5-year window ending now)
    """
    age_bracket = random.choices(
        [(0, 17), (18, 34), (35, 54), (55, 74), (75, 95)],
        weights=[0.22, 0.21, 0.25, 0.22, 0.10],
        k=1
    )[0]

    age = random.randint(age_bracket[0], age_bracket[1])
    # Assuming we're generating data for last 5 years ending in 2025
    birth_year = 2025 - age
    return birth_year


def generate_sex_distribution() -> str:
    """
    Generate sex following US population distribution.

    US distribution: ~50.5% Female, ~49.5% Male
    """
    return random.choices(["FEMALE", "MALE"], weights=[0.505, 0.495], k=1)[0]


def generate_state() -> str:
    """Generate state following US population distribution."""
    states, weights = zip(*US_STATES_WITH_WEIGHTS)
    return random.choices(states, weights=weights, k=1)[0]


def generate_dates_last_5_years(n_events: int) -> List[date]:
    """
    Generate n_events dates within the last 5 years.

    Args:
        n_events: Number of dates to generate

    Returns:
        Sorted list of dates
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=5*365)

    date_range = (end_date - start_date).days
    dates = []

    for _ in range(n_events):
        random_days = random.randint(0, date_range)
        random_date = start_date + timedelta(days=random_days)
        dates.append(random_date)

    return sorted(dates)


def generate_patient_events(patient_id: int, novel_code_prob: float = 0.01) -> dict:
    """
    Generate medical events for a single patient.

    Args:
        patient_id: Unique patient identifier
        novel_code_prob: Probability of generating a novel code (default: 0.01 = 1%)

    Returns:
        Dictionary with patient data compatible with Patient class
    """
    # Generate patient demographics
    birth_year = generate_age_distribution()
    sex = generate_sex_distribution()
    patient_state = generate_state()

    # Generate variable number of events (realistic range)
    # Most patients have 5-50 events over 5 years
    n_diagnoses = random.randint(3, 30)
    n_prescriptions = random.randint(2, 40)

    # Generate medical events
    # Use power law distribution - some codes more common than others
    diagnoses = random.choices(
        COMMON_ICD10_CODES,
        weights=[1/(i+1)**0.5 for i in range(len(COMMON_ICD10_CODES))],
        k=n_diagnoses
    )

    drugs = random.choices(
        COMMON_ATC5_CODES,
        weights=[1/(i+1)**0.5 for i in range(len(COMMON_ATC5_CODES))],
        k=n_prescriptions
    )

    # Inject novel codes (1% of the time)
    # For diagnoses
    for i in range(len(diagnoses)):
        if random.random() < novel_code_prob:
            parent_code = diagnoses[i]
            diagnoses[i] = generate_novel_icd10(parent_code)

    # For drugs
    for i in range(len(drugs)):
        if random.random() < novel_code_prob:
            parent_code = drugs[i]
            drugs[i] = generate_novel_atc5(parent_code)

    # Generate dates
    diagnosis_dates = generate_dates_last_5_years(n_diagnoses)
    prescription_dates = generate_dates_last_5_years(n_prescriptions)

    return {
        "patient_id": patient_id,
        "diagnoses": diagnoses,
        "drugs": drugs,
        "diagnosis_dates": [d.strftime("%Y%m%d") for d in diagnosis_dates],
        "prescription_dates": [d.strftime("%Y%m%d") for d in prescription_dates],
        "birth_year": birth_year,
        "sex": sex,
        "patient_state": patient_state,
    }


def generate_synthetic_patients(
    n_patients: int,
    output_file: Path,
    seed: int = 42,
    novel_code_prob: float = 0.01
):
    """
    Generate N synthetic patients and save to JSON file.

    Args:
        n_patients: Number of patients to generate
        output_file: Path to output JSON file
        seed: Random seed for reproducibility
        novel_code_prob: Probability of generating novel codes (default: 0.01 = 1%)
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating {n_patients} synthetic patients...")
    print(f"Novel code probability: {novel_code_prob*100:.1f}%")

    patients = []
    for i in range(n_patients):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_patients} patients")

        patient_data = generate_patient_events(patient_id=i, novel_code_prob=novel_code_prob)
        patients.append(patient_data)

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(patients, f, indent=2)

    print(f"\nSuccessfully generated {n_patients} patients")
    print(f"Saved to: {output_file}")

    # Print statistics
    total_diagnoses = sum(len(p['diagnoses']) for p in patients)
    total_drugs = sum(len(p['drugs']) for p in patients)

    # Count novel codes
    all_diagnoses = [dx for p in patients for dx in p['diagnoses']]
    all_drugs = [drug for p in patients for drug in p['drugs']]

    novel_diagnoses = len(_NOVEL_ICD10_CACHE)
    novel_drugs = len(_NOVEL_ATC5_CACHE)
    novel_diagnoses_count = sum(1 for dx in all_diagnoses if dx in _NOVEL_ICD10_CACHE)
    novel_drugs_count = sum(1 for drug in all_drugs if drug in _NOVEL_ATC5_CACHE)

    print("\nDataset Statistics:")
    print(f"  Total diagnoses: {total_diagnoses}")
    print(f"  Total prescriptions: {total_drugs}")
    print(f"  Avg diagnoses per patient: {np.mean([len(p['diagnoses']) for p in patients]):.1f}")
    print(f"  Avg prescriptions per patient: {np.mean([len(p['drugs']) for p in patients]):.1f}")

    print(f"\nNovel Codes:")
    print(f"  Unique novel ICD-10 codes generated: {novel_diagnoses}")
    print(f"  Total novel ICD-10 code instances: {novel_diagnoses_count} ({100*novel_diagnoses_count/total_diagnoses:.2f}%)")
    print(f"  Unique novel ATC5 codes generated: {novel_drugs}")
    print(f"  Total novel ATC5 code instances: {novel_drugs_count} ({100*novel_drugs_count/total_drugs:.2f}%)")

    # Show examples of novel codes
    if novel_diagnoses > 0:
        print(f"\n  Example novel ICD-10 codes (hierarchical cousins):")
        for code in sorted(list(_NOVEL_ICD10_CACHE))[:5]:
            print(f"    {code}")
    if novel_drugs > 0:
        print(f"\n  Example novel ATC5 codes (hierarchical cousins):")
        for code in sorted(list(_NOVEL_ATC5_CACHE))[:5]:
            print(f"    {code}")

    # Sex distribution
    sex_counts = {"MALE": 0, "FEMALE": 0}
    for p in patients:
        sex_counts[p["sex"]] += 1
    print(f"\nSex Distribution:")
    print(f"  Male: {sex_counts['MALE']} ({100*sex_counts['MALE']/n_patients:.1f}%)")
    print(f"  Female: {sex_counts['FEMALE']} ({100*sex_counts['FEMALE']/n_patients:.1f}%)")

    # Age distribution
    ages = [2025 - p["birth_year"] for p in patients]
    print(f"\nAge Distribution:")
    print(f"  Mean age: {np.mean(ages):.1f}")
    print(f"  Median age: {np.median(ages):.1f}")
    print(f"  Age range: {min(ages)} - {max(ages)}")

    # State distribution (top 10)
    state_counts = {}
    for p in patients:
        state = p["patient_state"]
        state_counts[state] = state_counts.get(state, 0) + 1

    print(f"\nTop 10 States:")
    for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {state}: {count} ({100*count/n_patients:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient data with realistic demographics and novel codes"
    )
    parser.add_argument(
        "-n", "--n_patients",
        type=int,
        default=1000,
        help="Number of patients to generate (default: 1000)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/synthetic_patients.json",
        help="Output JSON file path (default: data/synthetic_patients.json)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "-p", "--novel_prob",
        type=float,
        default=0.01,
        help="Probability of novel codes (default: 0.01 = 1%%)"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    generate_synthetic_patients(
        args.n_patients,
        output_path,
        args.seed,
        args.novel_prob
    )


if __name__ == "__main__":
    main()
