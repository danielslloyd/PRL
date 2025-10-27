# CPT Code Support Added

## Overview

Added CPT (Current Procedural Terminology) procedure code support to the ClinVec vocabulary auto-detection feature in the interactive training script.

## Changes Made

### [train_interactive.bat](train_interactive.bat:310-324)

Added CPT detection block after PheCode detection:

```batch
REM Check for CPT
if exist "%CLINVEC_DIR%\ClinVec_cpt.csv" (
    echo   [FOUND] CPT Procedure Codes ^(ClinVec_cpt.csv^)
    set /p USE_CPT="  Include CPT codes? (y/n) [y]: "
    if "!USE_CPT!"=="" set USE_CPT=y
    if /i "!USE_CPT!"=="y" (
        if defined VOCAB_TYPES (
            set VOCAB_TYPES=!VOCAB_TYPES!,cpt
        ) else (
            set VOCAB_TYPES=cpt
        )
    )
) else (
    echo   [NOT FOUND] CPT Procedure Codes ^(ClinVec_cpt.csv^)
)
```

## Detected Vocabularies

Your ClinVec directory now contains:
- ✅ `ClinVec_icd10cm.csv` - ICD-10 diagnosis codes (46.3 MB)
- ✅ `ClinVec_atc.csv` - ATC drug codes (6.8 MB)
- ✅ `ClinVec_cpt.csv` - CPT procedure codes (5.7 MB)
- ❌ `ClinVec_rxnorm.csv` - Not present
- ❌ `ClinVec_phecode.csv` - Not present

## Updated Documentation

### [INTERACTIVE_TRAINING_GUIDE.md](INTERACTIVE_TRAINING_GUIDE.md)
- Added CPT to the vocabulary selection list
- Updated example to show CPT detection and selection
- Example now shows: `icd10cm,atc,cpt`

### [ICD9_REMOVAL_SUMMARY.md](ICD9_REMOVAL_SUMMARY.md)
- Added `ClinVec_cpt.csv` to the list of optional ClinVec files

### [VOCAB_AUTO_DETECTION.md](VOCAB_AUTO_DETECTION.md)
- Added CPT to automatic file detection list
- Updated example interaction
- Added CPT to file naming convention section

## Example Interaction

When you run `train_interactive.bat`, you'll now see:

```
Detecting available ClinVec vocabulary files...
  [FOUND] ICD-10 CM (ClinVec_icd10cm.csv)
  Include ICD-10 codes? (y/n) [y]: y
  [FOUND] ATC Drug Codes (ClinVec_atc.csv)
  Include ATC drug codes? (y/n) [y]: y
  [NOT FOUND] RxNorm (ClinVec_rxnorm.csv)
  [NOT FOUND] PheCode (ClinVec_phecode.csv)
  [FOUND] CPT Procedure Codes (ClinVec_cpt.csv)
  Include CPT codes? (y/n) [y]: y

Selected vocabularies: icd10cm,atc,cpt
```

## Technical Details

### CPT Code Format
CPT codes are 5-digit numeric codes used for medical procedures and services:
- Example: `99213` (Office visit)
- Example: `80053` (Comprehensive metabolic panel)
- Example: `36415` (Venipuncture)

### Integration with ClinVec
CPT embeddings are loaded the same way as other vocabularies:
- File: `ClinVec_cpt.csv`
- Vocabulary type: `cpt`
- No hierarchical initialization (CPT doesn't have a natural hierarchy like ICD-10 or ATC)

### Default Behavior
- If CPT file exists: Prompts user y/n (defaults to 'y')
- If CPT file doesn't exist: Shows [NOT FOUND] and skips prompt
- CPT is added to the comma-separated vocabulary list if selected

## Date
2025-10-27

## Related
- [VOCAB_AUTO_DETECTION.md](VOCAB_AUTO_DETECTION.md) - Original auto-detection feature
- [ICD9_REMOVAL_SUMMARY.md](ICD9_REMOVAL_SUMMARY.md) - ICD-9 removal and ClinVec standardization
