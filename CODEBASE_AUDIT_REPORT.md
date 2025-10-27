# PRL Codebase Deep Cleanup Audit

**Date:** 2025-10-27
**Status:** Ready for Cleanup
**Impact:** 22 files to remove/reorganize, ~2000 lines of duplicate documentation eliminated

---

## Executive Summary

The PRL (ExMed-BERT) codebase is **functionally sound** but suffers from **documentation sprawl**. Analysis shows:

- **69 total files** (Python, Markdown, Batch, Config)
- **21 markdown documentation files** with significant redundancy
- **Core functionality is solid** (16-file exmed_bert package, 10 active scripts)
- **Opportunity to reduce to ~47 files** with better organization

### Key Findings

✅ **CORE IS EXCELLENT:** exmed_bert package is well-structured
⚠️ **DOCUMENTATION SPRAWL:** 5 overlapping "getting started" guides
⚠️ **CHANGE LOG CLUTTER:** 7 git-history files in root directory
⚠️ **TEST DISORGANIZATION:** Test files scattered in root
⚠️ **EXAMPLE SCRIPTS:** Should be in separate examples/ folder

---

## Quick Action Summary

### Phase 1: DELETE (13 files - 5 minutes)
```
✗ temp_envs.txt
✗ patient data format.txt
✗ icd_to_phewas_map_backup.py
✗ debug_position_ids.py
✗ test_hierarchical.py
✗ test_syntax.bat
✗ run_test.bat
✗ ICD10_CONVERSION_SUMMARY.md
✗ ICD9_REMOVAL_SUMMARY.md
✗ CLINVEC_DEFAULT_CHANGED.md
✗ CPT_SUPPORT_ADDED.md
✗ VOCAB_AUTO_DETECTION.md
✗ INTERACTIVE_BAT_FIXES.md
```

### Phase 2: CONSOLIDATE (12 MD files → 3 guides)
- 5 getting started guides → `docs/GETTING_STARTED.md`
- 3 training guides → `docs/TRAINING_GUIDE.md`
- 2 troubleshooting guides → `docs/TROUBLESHOOTING.md`

### Phase 3: REORGANIZE (7 files to new folders)
- 4 files → `examples/`
- 3 files → `tests/`

---

## Detailed Analysis

### 1. CORE FILES (Keep As-Is) ✅

#### exmed_bert Package (16 files)
All files are essential and actively used:

**Data Module:**
- `exmed_bert/data/__init__.py`
- `exmed_bert/data/data_exceptions.py`
- `exmed_bert/data/dataset.py`
- `exmed_bert/data/encoding.py`
- `exmed_bert/data/patient.py`

**Models Module:**
- `exmed_bert/models/__init__.py`
- `exmed_bert/models/config.py`
- `exmed_bert/models/model.py`
- `exmed_bert/models/trainer.py`

**Utils Module:**
- `exmed_bert/utils/__init__.py`
- `exmed_bert/utils/clinvec_integration.py` (recently updated for ATC5, CPT)
- `exmed_bert/utils/helpers.py`
- `exmed_bert/utils/metrics.py`
- `exmed_bert/utils/sequence_prep.py`
- `exmed_bert/utils/trainer.py`

#### Main Training Scripts (10 files in scripts/)
- `pretrain-exmed-bert.py` ✅
- `pretrain-exmed-bert-clinvec.py` ✅ (used by batch files)
- `finetune-exmed-bert.py` ✅
- `train-rf.py` ✅
- `train-xgboost.py` ✅
- `generate_synthetic_patients.py` ✅ (recently updated)
- `convert_synthetic_to_dataset.py` ✅
- `example_load_synthetic_data.py` ✅
- `calculate_iptw_scores.py` ✅
- `integrate_clinvec.py` ✅

#### Entry Point Batch Files (2 files)
- `train.bat` ✅ (main entry point)
- `train_interactive.bat` ✅ (recently updated with vocab auto-detection)

---

### 2. FILES TO DELETE (13 files)

#### Temporary Files (4 files)
```bash
rm temp_envs.txt
rm "patient data format.txt"
rm icd_to_phewas_map_backup.py  # Obsolete backup
rm debug_position_ids.py        # Debug script, not used
```

#### Test Utility Files (3 files)
```bash
rm test_hierarchical.py   # 462 bytes stub
rm test_syntax.bat        # Just calls python test_syntax.py
rm run_test.bat          # Not documented
```

#### Change Log Files (6 files)
These document code changes that are already in git history:
```bash
rm ICD10_CONVERSION_SUMMARY.md
rm ICD9_REMOVAL_SUMMARY.md
rm CLINVEC_DEFAULT_CHANGED.md
rm CPT_SUPPORT_ADDED.md
rm VOCAB_AUTO_DETECTION.md
rm INTERACTIVE_BAT_FIXES.md
```

**Rationale:** All changes are documented in git commits. These files clutter the root directory and confuse users looking for getting started documentation.

---

### 3. DOCUMENTATION CONSOLIDATION

#### Problem: 21 Markdown Files (5 overlapping guides)

**Current State:**
```
QUICKSTART.md                (162 lines) - Quick start for synthetic data
HOW_TO_RUN.md                (229 lines) - Step-by-step guide
SIMPLE_START.md              (124 lines) - 2-step approach
INSTALLATION_GUIDE.md        (232 lines) - PyTorch installation
README_SYNTHETIC_TRAINING.md (175 lines) - Automated script
TRAINING_WITH_SYNTHETIC_DATA.md (351 lines) - Complete guide
QUICK_REFERENCE.md           (122 lines) - Command reference
INTERACTIVE_TRAINING_GUIDE.md (272 lines) - Interactive mode
FIX_CONDA_PATH.md            (161 lines) - Conda troubleshooting
VSCODE_SETUP.md              (226 lines) - VS Code setup
BATCH_FILES_SUMMARY.md       (218 lines) - Batch file comparison
```

**Total:** ~2,300 lines of overlapping documentation

#### Solution: Consolidate to 3 Guides

**NEW: `docs/GETTING_STARTED.md`** (Merge 5 files)
```markdown
# Getting Started with ExMed-BERT

## Prerequisites
- Python 3.8-3.12 (64-bit)
- Conda/Miniconda
- PyTorch with CUDA (optional)

## Quick Start (One Command)
.\train_interactive.bat

## Manual Step-by-Step
1. Generate synthetic data
2. Convert to dataset
3. Train model

## Installation Issues
→ See TROUBLESHOOTING.md
```

**Files to merge:**
- QUICKSTART.md ✗
- HOW_TO_RUN.md ✗
- SIMPLE_START.md ✗
- INSTALLATION_GUIDE.md ✗
- README_SYNTHETIC_TRAINING.md ✗

---

**NEW: `docs/TRAINING_GUIDE.md`** (Merge 4 files)
```markdown
# Training Guide

## Quick Reference
| Task | Command |
|------|---------|
| Interactive | .\train_interactive.bat |
| Automated | .\train.bat |

## Training Pipeline
1. Data generation
2. ClinVec configuration
3. Model training

## Interactive Mode
(From INTERACTIVE_TRAINING_GUIDE.md)

## Advanced Configuration
- ClinVec vocabularies (ICD-10, ATC, CPT)
- Custom model architectures
- Hierarchical initialization
```

**Files to merge:**
- TRAINING_WITH_SYNTHETIC_DATA.md ✗
- QUICK_REFERENCE.md ✗
- INTERACTIVE_TRAINING_GUIDE.md ✗
- BATCH_FILES_SUMMARY.md ✗

---

**NEW: `docs/TROUBLESHOOTING.md`** (Merge 2 files)
```markdown
# Troubleshooting Guide

## Conda Setup Issues
(From FIX_CONDA_PATH.md)
- PATH configuration
- Environment activation

## VS Code Configuration
(From VSCODE_SETUP.md)
- Terminal settings
- Python interpreter

## Common Errors
- PyTorch installation
- Module not found
- Out of memory
```

**Files to merge:**
- FIX_CONDA_PATH.md ✗
- VSCODE_SETUP.md ✗

---

**RENAME:**
- `SYNTHETIC_DATA_README.md` → `docs/SYNTHETIC_DATA_GUIDE.md` (technical reference)

---

### 4. FILES TO REORGANIZE

#### Create `examples/` Folder (4 files)
```bash
mkdir examples
mv pretrain_example.py examples/
mv pretrain_example_with_clinvec.py examples/
mv test_novel_codes.py examples/
mv regenerate_training_data.py examples/
```

#### Create `tests/` Folder (3 files)
```bash
mkdir tests
mv test_atc5_hierarchy.py tests/
mv test_icd_pipeline.py tests/
mv test_syntax.py tests/
```

Create `tests/pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

---

### 5. BATCH FILE CLEANUP

#### Delete Duplicate Batch Files (2 files)
```bash
rm scripts/train_on_synthetic_data.bat      # Duplicate of train.bat
rm scripts/train_on_synthetic_data_simple.bat  # Merge into train.bat
```

**Keep:**
- `train.bat` (main entry point)
- `train_interactive.bat` (interactive mode with vocab auto-detection)
- `scripts/setup_environment.bat` (standalone setup)

#### Update `scripts/train_on_synthetic_data.sh` (if exists)
Unix users may still use the .sh version - keep it.

---

## Proposed Final Structure

```
PRL/
├── exmed_bert/              # Core package (16 files) ✅
│   ├── data/
│   ├── models/
│   └── utils/
│
├── scripts/                 # Training scripts (10 files) ✅
│   ├── pretrain-exmed-bert.py
│   ├── pretrain-exmed-bert-clinvec.py
│   ├── finetune-exmed-bert.py
│   ├── train-rf.py
│   ├── train-xgboost.py
│   ├── generate_synthetic_patients.py
│   ├── convert_synthetic_to_dataset.py
│   ├── example_load_synthetic_data.py
│   ├── calculate_iptw_scores.py
│   ├── integrate_clinvec.py
│   └── setup_environment.bat
│
├── tests/                   # Test suite (3 files) ✅
│   ├── test_atc5_hierarchy.py
│   ├── test_icd_pipeline.py
│   ├── test_syntax.py
│   └── pytest.ini
│
├── examples/                # Example scripts (4 files) ✅
│   ├── pretrain_example.py
│   ├── pretrain_example_with_clinvec.py
│   ├── test_novel_codes.py
│   └── regenerate_training_data.py
│
├── docs/                    # Documentation (5 files) ✅
│   ├── GETTING_STARTED.md       [NEW - Consolidated]
│   ├── TRAINING_GUIDE.md        [NEW - Consolidated]
│   ├── TROUBLESHOOTING.md       [NEW - Consolidated]
│   ├── SYNTHETIC_DATA_GUIDE.md  [Renamed]
│   └── ClinVec_Integration.md
│
├── ClinVec/                 # ClinVec embeddings ✅
│   ├── ClinGraph_nodes.csv
│   ├── ClinVec_icd10cm.csv
│   ├── ClinVec_atc.csv
│   └── ClinVec_cpt.csv
│
├── train.bat                # Main entry point ✅
├── train_interactive.bat    # Interactive mode ✅
├── README.md                # Enhanced main README ✅
├── CLAUDE.md                # Claude Code instructions ✅
├── config.yaml              # Configuration ✅
├── environment.yaml         # Conda environment ✅
├── pyproject.toml           # Poetry config ✅
└── requirements.txt         # Pip fallback ✅
```

---

## Implementation Plan

### Phase 1: Quick Deletions (5 minutes)

```bash
# Temporary files
rm temp_envs.txt
rm "patient data format.txt"
rm icd_to_phewas_map_backup.py
rm debug_position_ids.py

# Test utilities
rm test_hierarchical.py
rm test_syntax.bat
rm run_test.bat

# Change logs
rm ICD10_CONVERSION_SUMMARY.md
rm ICD9_REMOVAL_SUMMARY.md
rm CLINVEC_DEFAULT_CHANGED.md
rm CPT_SUPPORT_ADDED.md
rm VOCAB_AUTO_DETECTION.md
rm INTERACTIVE_BAT_FIXES.md
```

**Verification:** `git status` should show 13 deleted files

---

### Phase 2: Create New Folders (1 minute)

```bash
mkdir tests
mkdir examples
```

---

### Phase 3: Move Files to New Folders (2 minutes)

```bash
# Examples
mv pretrain_example.py examples/
mv pretrain_example_with_clinvec.py examples/
mv test_novel_codes.py examples/
mv regenerate_training_data.py examples/

# Tests
mv test_atc5_hierarchy.py tests/
mv test_icd_pipeline.py tests/
mv test_syntax.py tests/
```

---

### Phase 4: Documentation Consolidation (2 hours)

**Step 1: Create `docs/GETTING_STARTED.md`**
- Merge content from QUICKSTART, HOW_TO_RUN, SIMPLE_START, INSTALLATION_GUIDE, README_SYNTHETIC_TRAINING
- Focus on clear prerequisites and quick start
- Delete source files after verification

**Step 2: Create `docs/TRAINING_GUIDE.md`**
- Start with TRAINING_WITH_SYNTHETIC_DATA.md as base
- Add Quick Reference section from QUICK_REFERENCE.md
- Add Interactive Mode section from INTERACTIVE_TRAINING_GUIDE.md
- Add Batch Files section from BATCH_FILES_SUMMARY.md
- Delete source files after verification

**Step 3: Create `docs/TROUBLESHOOTING.md`**
- Merge FIX_CONDA_PATH.md and VSCODE_SETUP.md
- Add common errors section
- Delete source files after verification

**Step 4: Rename**
```bash
mv SYNTHETIC_DATA_README.md docs/SYNTHETIC_DATA_GUIDE.md
```

**Step 5: Delete consolidated files**
```bash
rm QUICKSTART.md
rm HOW_TO_RUN.md
rm SIMPLE_START.md
rm INSTALLATION_GUIDE.md
rm README_SYNTHETIC_TRAINING.md
rm TRAINING_WITH_SYNTHETIC_DATA.md
rm QUICK_REFERENCE.md
rm INTERACTIVE_TRAINING_GUIDE.md
rm BATCH_FILES_SUMMARY.md
rm FIX_CONDA_PATH.md
rm VSCODE_SETUP.md
```

---

### Phase 5: Update Cross-References (30 minutes)

**Files to update:**
- `README.md` - Point to new consolidated docs
- `CLAUDE.md` - Update file references
- `train.bat` - Update any doc references
- `train_interactive.bat` - Update any doc references

**Example README.md structure:**
```markdown
# ExMed-BERT

Transformer-based model for medical code sequences with ClinVec pre-trained embeddings.

## Quick Start

For first-time users:
→ **[Getting Started Guide](docs/GETTING_STARTED.md)**

For training:
→ **[Training Guide](docs/TRAINING_GUIDE.md)**

For issues:
→ **[Troubleshooting](docs/TROUBLESHOOTING.md)**

## One-Command Training

```bash
.\train_interactive.bat
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Training Guide](docs/TRAINING_GUIDE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Synthetic Data Guide](docs/SYNTHETIC_DATA_GUIDE.md)
- [ClinVec Integration](docs/ClinVec_Integration.md)

## Project Structure

- `exmed_bert/` - Core package
- `scripts/` - Training scripts
- `tests/` - Test suite
- `examples/` - Example usage
```

---

## Summary

### Before Cleanup
- 69 total files
- 21 markdown files (many redundant)
- Documentation scattered at root
- No clear structure for tests/examples

### After Cleanup
- 47 total files (-22 files)
- 7 markdown files at root + 5 in docs/ (-9 at root)
- Clear folder structure (tests/, examples/, docs/)
- Single source of truth for documentation

### Benefits
1. **Easier onboarding** - One clear getting started guide
2. **Better organization** - Tests and examples in dedicated folders
3. **Reduced confusion** - No duplicate or outdated guides
4. **Cleaner git history** - Remove change log files
5. **Professional structure** - Follows Python project best practices

---

## Risk Assessment

**LOW RISK:**
- Deleting temporary files ✅
- Deleting change logs (in git history) ✅
- Moving examples to examples/ ✅
- Moving tests to tests/ ✅

**MEDIUM RISK:**
- Consolidating documentation (verify merged content)
- Deleting batch file duplicates (ensure train.bat covers all cases)

**MITIGATION:**
- Create git branch before cleanup
- Test training pipeline after consolidation
- Verify all batch files work correctly

---

## Next Steps

**Option 1: Manual Cleanup**
User executes commands from this document

**Option 2: Automated Script**
Create `cleanup.bat` script to automate phases 1-3

**Option 3: Interactive Cleanup**
Claude assists with each phase, verifying as we go

---

## Appendix: File Size Savings

| Category | Files | Lines | Size |
|----------|-------|-------|------|
| Deleted change logs | 6 | ~840 | ~28 KB |
| Consolidated getting started | 5 → 1 | ~900 → ~300 | ~31 KB → ~10 KB |
| Consolidated training | 4 → 1 | ~750 → ~400 | ~26 KB → ~14 KB |
| Consolidated troubleshooting | 2 → 1 | ~390 → ~200 | ~13 KB → ~7 KB |
| Deleted test utilities | 3 | ~100 | ~3 KB |
| Deleted temporary files | 4 | ~50 | ~2 KB |
| **TOTAL SAVINGS** | **25 files** | **~2,000 lines** | **~70 KB** |

---

**Generated:** 2025-10-27
**Status:** Ready for implementation
**Approval needed:** Yes (verify documentation consolidation strategy)
