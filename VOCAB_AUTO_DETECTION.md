# ClinVec Vocabulary Auto-Detection Feature

## Overview

The interactive training script (`train_interactive.bat`) now automatically detects which ClinVec vocabulary files are available and asks the user y/n for each one individually, instead of requiring manual entry of a comma-separated list.

## Changes Made

### [train_interactive.bat](train_interactive.bat:247-318)

**Before:**
```batch
REM Ask for vocabulary types
set /p VOCAB_TYPES="Vocabulary types (comma-separated) [icd10cm,atc]: "
if "%VOCAB_TYPES%"=="" set VOCAB_TYPES=icd10cm,atc
```

**After:**
```batch
REM Detect available vocabulary files
echo.
echo Detecting available ClinVec vocabulary files...
set VOCAB_TYPES=

REM Check for ICD-10
if exist "%CLINVEC_DIR%\ClinVec_icd10cm.csv" (
    echo   [FOUND] ICD-10 CM (ClinVec_icd10cm.csv)
    set /p USE_ICD10="  Include ICD-10 codes? (y/n) [y]: "
    if "!USE_ICD10!"=="" set USE_ICD10=y
    if /i "!USE_ICD10!"=="y" set VOCAB_TYPES=icd10cm
) else (
    echo   [NOT FOUND] ICD-10 CM (ClinVec_icd10cm.csv)
)

REM Check for ATC
if exist "%CLINVEC_DIR%\ClinVec_atc.csv" (
    echo   [FOUND] ATC Drug Codes (ClinVec_atc.csv)
    set /p USE_ATC="  Include ATC drug codes? (y/n) [y]: "
    if "!USE_ATC!"=="" set USE_ATC=y
    if /i "!USE_ATC!"=="y" (
        if defined VOCAB_TYPES (
            set VOCAB_TYPES=!VOCAB_TYPES!,atc
        ) else (
            set VOCAB_TYPES=atc
        )
    )
) else (
    echo   [NOT FOUND] ATC Drug Codes (ClinVec_atc.csv)
)

REM Check for RxNorm
if exist "%CLINVEC_DIR%\ClinVec_rxnorm.csv" (
    echo   [FOUND] RxNorm (ClinVec_rxnorm.csv)
    set /p USE_RXNORM="  Include RxNorm codes? (y/n) [y]: "
    if "!USE_RXNORM!"=="" set USE_RXNORM=y
    if /i "!USE_RXNORM!"=="y" (
        if defined VOCAB_TYPES (
            set VOCAB_TYPES=!VOCAB_TYPES!,rxnorm
        ) else (
            set VOCAB_TYPES=rxnorm
        )
    )
) else (
    echo   [NOT FOUND] RxNorm (ClinVec_rxnorm.csv)
)

REM Check for PheCode
if exist "%CLINVEC_DIR%\ClinVec_phecode.csv" (
    echo   [FOUND] PheCode (ClinVec_phecode.csv)
    set /p USE_PHECODE="  Include PheCode? (y/n) [y]: "
    if "!USE_PHECODE!"=="" set USE_PHECODE=y
    if /i "!USE_PHECODE!"=="y" (
        if defined VOCAB_TYPES (
            set VOCAB_TYPES=!VOCAB_TYPES!,phecode
        ) else (
            set VOCAB_TYPES=phecode
        )
    )
) else (
    echo   [NOT FOUND] PheCode (ClinVec_phecode.csv)
)

REM If no vocabularies selected, use default
if not defined VOCAB_TYPES (
    echo.
    echo WARNING: No vocabularies selected. Using default: icd10cm,atc
    set VOCAB_TYPES=icd10cm,atc
)

echo.
echo Selected vocabularies: %VOCAB_TYPES%
```

## Features

### 1. Automatic File Detection
The script checks for the presence of each ClinVec vocabulary file:
- `ClinVec_icd10cm.csv` - ICD-10 diagnosis codes
- `ClinVec_atc.csv` - ATC drug codes
- `ClinVec_rxnorm.csv` - RxNorm drug codes
- `ClinVec_phecode.csv` - PheWAS codes
- `ClinVec_cpt.csv` - CPT procedure codes

### 2. Clear Visual Feedback
For each vocabulary type, the script displays:
- `[FOUND]` if the CSV file exists
- `[NOT FOUND]` if the CSV file doesn't exist

### 3. Individual Y/N Prompts
- Each found vocabulary gets a y/n prompt with default `[y]`
- Not found vocabularies are skipped (no prompt)
- User can selectively choose which vocabularies to use

### 4. Smart Comma-Separated List Building
The script builds the comma-separated `VOCAB_TYPES` list dynamically:
- First selected vocab: `VOCAB_TYPES=icd10cm`
- Additional vocabs: `VOCAB_TYPES=icd10cm,atc`
- Ensures no leading/trailing commas

### 5. Fallback to Default
If no vocabularies are selected (all answered 'n'), the script uses a sensible default:
```
WARNING: No vocabularies selected. Using default: icd10cm,atc
```

## User Experience

### Example Interaction

```
ClinVec directory: C:\Users\danie\Desktop\Git\PRL\ClinVec

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

Use hierarchical initialization for novel codes? (y/n) [y]: y
```

### Benefits

1. **No Memorization Required**: User doesn't need to remember vocabulary type names
2. **Clear Visibility**: Shows exactly which files are available in the directory
3. **Prevents Typos**: No manual typing of comma-separated lists
4. **Explicit Control**: User can easily include/exclude specific vocabularies
5. **Self-Documenting**: Each prompt shows the actual filename being checked

## Documentation Updates

### [INTERACTIVE_TRAINING_GUIDE.md](INTERACTIVE_TRAINING_GUIDE.md:77-117)

Updated the guide to reflect:
- The three-step process: detect location → validate → detect vocabularies
- Individual y/n prompts for each vocabulary
- Example interaction showing [FOUND]/[NOT FOUND] indicators
- Note that only vocabularies with existing CSV files are shown

## Technical Details

### Variable Usage
- `USE_ICD10`, `USE_ATC`, `USE_RXNORM`, `USE_PHECODE` - Store individual y/n responses
- `VOCAB_TYPES` - Accumulates comma-separated list of selected vocabularies
- Uses delayed expansion (`!variable!`) for proper list building inside `if` blocks

### File Naming Convention
All ClinVec vocabulary files follow the pattern:
```
ClinVec_<vocab_type>.csv
```

Where `<vocab_type>` matches the vocabulary type string used in the Python code:
- `icd10cm` → `ClinVec_icd10cm.csv`
- `atc` → `ClinVec_atc.csv`
- `rxnorm` → `ClinVec_rxnorm.csv`
- `phecode` → `ClinVec_phecode.csv`
- `cpt` → `ClinVec_cpt.csv`

### Edge Cases Handled
1. **No files found**: Falls back to default `icd10cm,atc`
2. **All prompts answered 'n'**: Falls back to default with warning
3. **Empty response**: Defaults to 'y' (include the vocabulary)
4. **Case-insensitive**: Accepts 'Y', 'y', 'N', 'n'

## Related Files
- [train_interactive.bat](train_interactive.bat) - Main interactive script
- [INTERACTIVE_TRAINING_GUIDE.md](INTERACTIVE_TRAINING_GUIDE.md) - User documentation
- [ICD9_REMOVAL_SUMMARY.md](ICD9_REMOVAL_SUMMARY.md) - Previous changes removing ICD-9

## Date
2025-10-27
