# ICD-9 Removal and ClinVec Directory Standardization Summary

## Overview

This document summarizes all changes made to remove ICD-9 code support and standardize the ClinVec directory structure to use `/ClinVec` as the standard location.

## Changes Made

### 1. Core Code Changes

#### [exmed_bert/utils/clinvec_integration.py](exmed_bert/utils/clinvec_integration.py)

**Removed:**
- `self.icd9_pattern` regex pattern (line 281)
- `is_icd9_code()` method (lines 288-290)
- `_get_icd9_hierarchy()` method (lines 335-353)
- ICD-9 case from `get_code_hierarchy()` (lines 308-309)
- ICD-9 format variations from `find_best_parent_embedding()` (lines 419-423)

**Updated:**
- Default `vocab_types` changed from `["icd9cm", "icd10cm", "phecode"]` to `["icd10cm", "atc"]`
- Updated docstring to remove ICD-9 from vocabulary type list
- Updated example test function to remove ICD-9 detection logic
- Changed test type detection from "ICD-9 if matcher.is_icd9_code(code)" to "ATC5 if matcher.is_atc5_code(code)"

**Result:** The `HierarchicalCodeMatcher` class now only supports ICD-10 and ATC5 hierarchical initialization.

---

### 2. Training Scripts

#### [scripts/pretrain-exmed-bert-clinvec.py](scripts/pretrain-exmed-bert-clinvec.py)

**Updated:**
- Line 54: Changed default `vocab_types` from `"icd9cm,icd10cm,phecode"` to `"icd10cm,atc"`

---

### 3. Batch Files

#### [train_interactive.bat](train_interactive.bat)

**ClinVec Directory Standardization (lines 207-211):**
```batch
# Before:
if exist "clinvec_data" (...)
if exist "data\clinvec" (...)
if exist "..\clinvec_data" (...)

# After:
if exist "ClinVec" (
    echo Found: ClinVec\
    set DEFAULT_CLINVEC=ClinVec
    set CLINVEC_FOUND=y
)
```

**Vocabulary Types (lines 248-249):**
```batch
# Before:
set /p VOCAB_TYPES="Vocabulary types (comma-separated) [icd9cm,icd10cm,atc]: "
if "%VOCAB_TYPES%"=="" set VOCAB_TYPES=icd9cm,icd10cm,atc

# After:
set /p VOCAB_TYPES="Vocabulary types (comma-separated) [icd10cm,atc]: "
if "%VOCAB_TYPES%"=="" set VOCAB_TYPES=icd10cm,atc
```

---

### 4. Configuration Files

#### [config.yaml](config.yaml)

**Updated (lines 32-33):**
```yaml
# Before:
clinvec_dir: "../ClinVec"
vocab_types: ["icd10cm"]

# After:
clinvec_dir: "ClinVec"
vocab_types: ["icd10cm", "atc"]
```

**Changes:**
- Standardized directory to `ClinVec` (relative to project root)
- Added `atc` to default vocabulary types
- Updated comment to reflect ICD-10 and ATC drug codes

---

### 5. Documentation Updates

#### [BATCH_FILES_SUMMARY.md](BATCH_FILES_SUMMARY.md)

**Updated:**
- Line 66: Changed examples from "icd9cm, icd10cm, atc, etc." to "icd10cm, atc, phecode, rxnorm, etc."
- Line 153: Changed default vocabularies from "icd9cm,icd10cm,atc" to "icd10cm,atc"

#### [INTERACTIVE_TRAINING_GUIDE.md](INTERACTIVE_TRAINING_GUIDE.md)

**Updated Auto-detection Section (lines 78-80):**
```markdown
# Before:
1. **Auto-detect** common ClinVec locations:
   - `clinvec_data\`
   - `data\clinvec\`
   - `..\clinvec_data\`

# After:
1. **Auto-detect** ClinVec location:
   - `ClinVec\`
```

**Updated Vocabulary Types (lines 84-88):**
- Removed ICD-9 from the list
- Changed default from `[icd9cm,icd10cm,atc]` to `[icd10cm,atc]`
- Removed bullet point for ICD-9 diagnosis codes

**Updated Example (lines 100-101):**
```markdown
# Before:
ClinVec data directory [clinvec_data]: D:\datasets\clinvec
Vocabulary types (comma-separated) [icd9cm,icd10cm,atc]: icd10cm,atc

# After:
ClinVec data directory [ClinVec]: D:\datasets\ClinVec
Vocabulary types (comma-separated) [icd10cm,atc]: icd10cm,atc
```

---

### 6. Test Files

#### [test_atc5_hierarchy.py](test_atc5_hierarchy.py)

**Removed ICD-9 Test Cases:**
- Removed `("250.01", False, "ICD-9 code")` from ATC5 detection tests (line 23)
- Removed `("250.01", "ICD-9")` from mixed type tests (line 141)
- Removed ICD-9 detection logic from test loops (lines 153-158)

**Test Count Impact:**
- Test 1 (ATC5 Detection): 7 tests → 6 tests
- Test 4 (Mixed Types): 4 tests → 3 tests
- Total tests: 19 tests → 18 tests

---

## Impact Summary

### What's Removed
- ✅ All ICD-9 code pattern recognition
- ✅ All ICD-9 hierarchy generation
- ✅ All ICD-9 format variations in parent matching
- ✅ All ICD-9 references in documentation
- ✅ Multiple ClinVec directory auto-detection paths

### What's Standardized
- ✅ **ClinVec Directory:** Now expects `ClinVec/` in project root
- ✅ **Default Vocabularies:** Now `icd10cm,atc` everywhere (consistent)
- ✅ **Code Support:** ICD-10 and ATC5 only (with full hierarchical initialization)

### What Still Works
- ✅ ICD-10 hierarchical initialization (E11.65 → E11.6 → E11)
- ✅ ATC5 hierarchical initialization (N02BE01 → N02BE → N02B → N02 → N)
- ✅ Novel code handling with parent embeddings
- ✅ All ClinVec integration features (excluding ICD-9)
- ✅ Interactive training workflow
- ✅ All test suites (with updated test counts)

### Files Modified
1. `exmed_bert/utils/clinvec_integration.py` - Core integration code
2. `scripts/pretrain-exmed-bert-clinvec.py` - Training script
3. `train_interactive.bat` - Interactive batch file
4. `config.yaml` - Configuration defaults
5. `BATCH_FILES_SUMMARY.md` - Documentation
6. `INTERACTIVE_TRAINING_GUIDE.md` - Documentation
7. `test_atc5_hierarchy.py` - Test suite

### Migration Guide

**For Users:**

If you previously used ICD-9 codes or non-standard ClinVec directories:

1. **ClinVec Directory:**
   - Move your ClinVec data to `/ClinVec` in the project root
   - Or manually specify the path when prompted by the interactive script

2. **ICD-9 Codes:**
   - ICD-9 codes are no longer supported
   - Convert ICD-9 codes to ICD-10 using standard mapping tools
   - Or exclude ICD-9 codes from your dataset

3. **Vocabulary Types:**
   - Update any custom scripts using `vocab_types` parameter
   - Change from `["icd9cm", ...]` to `["icd10cm", "atc"]`

**Required ClinVec Files (in `/ClinVec` directory):**
- `ClinGraph_nodes.csv` - Node mapping (required)
- `ClinVec_icd10cm.csv` - ICD-10 embeddings
- `ClinVec_atc.csv` - ATC drug code embeddings (optional)
- `ClinVec_cpt.csv` - CPT procedure code embeddings (optional)
- `ClinVec_rxnorm.csv` - RxNorm embeddings (optional)
- `ClinVec_phecode.csv` - PheCode embeddings (optional)

**NOT Required:**
- ~~`ClinVec_icd9cm.csv`~~ (no longer supported)
- ~~`ClinGraph_adjlist.tab`~~ (not used by ExMed-BERT)
- ~~`ClinGraph_edges.csv`~~ (not used by ExMed-BERT)
- ~~`ClinGraph_features.csv`~~ (not used by ExMed-BERT)
- ~~`ClinGraph_dgl.bin`~~ (GNN-specific, not used)
- ~~`ClinGraph_pyg.pt`~~ (GNN-specific, not used)

---

## Verification

To verify the changes work correctly:

1. **Run the test suite:**
   ```bash
   python test_atc5_hierarchy.py
   ```
   Expected: 18 tests pass (6 ATC5 detection + 5 hierarchy + 4 parent matching + 3 mixed types)

2. **Run interactive training:**
   ```bash
   train_interactive.bat
   ```
   - Verify ClinVec auto-detection finds `ClinVec/` directory
   - Verify default vocab types are `icd10cm,atc`
   - Verify no ICD-9 options appear

3. **Check ClinVec integration:**
   ```python
   from exmed_bert.utils.clinvec_integration import HierarchicalCodeMatcher

   matcher = HierarchicalCodeMatcher()

   # Should work:
   print(matcher.get_code_hierarchy("E11.65"))  # ICD-10
   print(matcher.get_code_hierarchy("N02BE01"))  # ATC5

   # No longer available:
   # matcher.is_icd9_code("250.01")  # AttributeError
   ```

---

## Date
2025-10-27

## Related Files
- Previous work: ATC5 hierarchical initialization (see test_atc5_hierarchy.py)
- Interactive training: INTERACTIVE_TRAINING_GUIDE.md
- Batch file fixes: INTERACTIVE_BAT_FIXES.md
