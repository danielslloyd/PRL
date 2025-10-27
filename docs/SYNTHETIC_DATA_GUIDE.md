# Synthetic Patient Data Generator

## Overview

This script generates synthetic patient data with realistic demographics and medical codes, including novel codes that are hierarchical cousins of existing codes.

## Features

### Medical Codes
- **50 most common ICD-10 diagnosis codes** from US healthcare
- **50 most common ATC5 drug codes** (5th level chemical substances)
- **Novel code generation**: Creates hierarchically valid "cousin" codes that share the same parent categories

### Demographics (US Population-based)
- **Age**: Matches US Census distribution across age brackets
- **Sex**: 50.5% Female, 49.5% Male
- **State**: All 50 states weighted by actual population

### Temporal Data
- All medical events occur within the **last 5 years**
- Realistic event frequency: 3-30 diagnoses, 2-40 prescriptions per patient

## Novel Code Generation

The script generates **1% novel codes by default** using hierarchical cousin relationships:

### ICD-10 Novel Codes
ICD-10 codes have a hierarchical structure:
- **Chapter** (1st letter): Anatomical system (e.g., E = Endocrine)
- **Category** (first 3 chars): Disease category (e.g., E11 = Type 2 diabetes)
- **Subcategory** (after decimal): Specific manifestation (e.g., E11.9 = unspecified)

**Novel generation strategy**: Keep the same chapter and category, generate a new subcategory.

**Example:**
- Parent: `E11.9` (Type 2 diabetes, unspecified)
- Novel cousin: `E11.95` (Type 2 diabetes, novel manifestation)
- Both belong to the same disease category (E11 = Type 2 diabetes)

### ATC5 Novel Codes
ATC codes have 5 hierarchical levels (7 characters total):
- **Level 1** (1 char): Anatomical main group (e.g., N = Nervous system)
- **Level 2** (3 chars): Therapeutic subgroup (e.g., N02 = Analgesics)
- **Level 3** (4 chars): Pharmacological subgroup (e.g., N02B = Other analgesics)
- **Level 4** (5 chars): Chemical subgroup (e.g., N02BE = Anilides)
- **Level 5** (7 chars): Chemical substance (e.g., N02BE01 = Paracetamol)

**Novel generation strategy**: Keep levels 1-4 (first 5 characters), generate new level 5 suffix.

**Example:**
- Parent: `N02BE01` (Paracetamol)
- Novel cousin: `N02BE75` (Novel anilide analgesic)
- Both are anilide-type analgesics from the same chemical subgroup (N02BE)

## Usage

### Basic Usage
```bash
# Generate 1000 patients with default settings (1% novel codes)
python scripts/generate_synthetic_patients.py

# Generate custom number of patients
python scripts/generate_synthetic_patients.py -n 5000

# Specify output file
python scripts/generate_synthetic_patients.py -o data/my_patients.json
```

### Advanced Options
```bash
# Control novel code probability (2% instead of 1%)
python scripts/generate_synthetic_patients.py -n 1000 -p 0.02

# Set random seed for reproducibility
python scripts/generate_synthetic_patients.py -n 1000 -s 123

# Disable novel codes entirely
python scripts/generate_synthetic_patients.py -n 1000 -p 0.0

# All options combined
python scripts/generate_synthetic_patients.py -n 5000 -o data/train.json -s 42 -p 0.01
```

### Command-Line Arguments
- `-n, --n_patients`: Number of patients to generate (default: 1000)
- `-o, --output`: Output JSON file path (default: data/synthetic_patients.json)
- `-s, --seed`: Random seed for reproducibility (default: 42)
- `-p, --novel_prob`: Probability of novel codes (default: 0.01 = 1%)

## Output Format

The generated JSON file contains a list of patient objects:

```json
[
  {
    "patient_id": 0,
    "diagnoses": ["I10", "E78.5", "Z23", "E11.95"],
    "drugs": ["N02BE01", "C10AA05", "N02BE75"],
    "diagnosis_dates": ["20200315", "20210622", "20220104", "20231215"],
    "prescription_dates": ["20200320", "20210625", "20230810"],
    "birth_year": 1965,
    "sex": "FEMALE",
    "patient_state": "CA"
  }
]
```

Note: Novel codes like `E11.95` and `N02BE75` are hierarchical cousins that don't exist in the common code lists.

## Statistics Output

The script prints comprehensive statistics:

```
Dataset Statistics:
  Total diagnoses: 15432
  Total prescriptions: 18765
  Avg diagnoses per patient: 15.4
  Avg prescriptions per patient: 18.8

Novel Codes:
  Unique novel ICD-10 codes generated: 127
  Total novel ICD-10 code instances: 154 (1.00%)
  Unique novel ATC5 codes generated: 143
  Total novel ATC5 code instances: 188 (1.00%)

  Example novel ICD-10 codes (hierarchical cousins):
    E03.95
    E11.73
    E28.86
    E66.52
    F10.77

  Example novel ATC5 codes (hierarchical cousins):
    A02BC73
    A10BA67
    A11CC82
    B01AC59
    C01BD91
```

## Integration with ExMed-BERT

The output format is compatible with the `Patient` class in the codebase:

```python
import json
from exmed_bert.data.patient import Patient
from exmed_bert.data.encoding import CodeDict, AgeDict, SexDict, StateDict

# Load synthetic data
with open("data/synthetic_patients.json", "r") as f:
    patient_data = json.load(f)

# Convert to Patient objects
for p in patient_data:
    patient = Patient.from_dict(
        p,
        code_embed=code_dict,
        sex_embed=sex_dict,
        age_embed=age_dict,
        state_embed=state_dict,
        max_length=512
    )
```

## Use Cases

1. **Testing model robustness**: Novel codes test how well the model handles unseen codes with known hierarchical relationships
2. **Hierarchical embedding evaluation**: Verify that embeddings capture hierarchical structure
3. **Transfer learning**: Test generalization to new codes within known categories
4. **ClinVec integration**: Test hierarchical initialization for novel codes

## Technical Details

- Novel codes are cached to ensure consistency across patients
- Novel codes never conflict with common codes in the predefined lists
- Hierarchical relationships are preserved (same parent categories)
- Power-law distribution for common codes (more frequent codes appear more often)
- Dates are uniformly distributed over 5 years with proper sorting
