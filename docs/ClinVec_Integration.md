# ClinVec Integration with ExMed-BERT

This guide explains how to integrate pre-trained ClinVec embeddings with your ExMed-BERT model.

## Overview

**ClinVec** provides high-quality, pre-trained embeddings for medical codes including:
- **ICD-9-CM & ICD-10-CM** (diagnosis codes)
- **PheCode** (phenotype codes)
- **RxNorm** (medication codes) 
- **ATC** (anatomical therapeutic chemical codes)
- **SNOMED-CT** (clinical terminology)

The integration allows you to:
1. **Initialize known medical codes** with meaningful pre-trained representations
2. **Train novel codes** from scratch during model training
3. **Fine-tune all embeddings** during your training process

## Quick Start

### 1. Basic Integration

```python
from exmed_bert.utils.clinvec_integration import load_for_exmedbert_pretraining

# Load model and data (your existing code)
model = ExMedBertModel(config)
code_dict = load_your_code_dictionary()

# Integrate ClinVec embeddings
stats = load_for_exmedbert_pretraining(
    model=model,
    code_dict=code_dict,
    clinvec_dir="path/to/ClinVec"
)

print(f"Loaded {sum(stats.values())} pre-trained embeddings")

# Continue with training...
trainer.train()
```

### 2. Custom Vocabulary Selection

```python
from exmed_bert.utils.clinvec_integration import integrate_clinvec_with_exmedbert

# Load only specific vocabularies with hierarchical initialization
stats = integrate_clinvec_with_exmedbert(
    model=model,
    code_dict=code_dict,
    clinvec_dir="path/to/ClinVec",
    vocab_types=["icd10cm", "phecode"],  # Focus on specific vocabularies
    resize_if_needed=True,
    use_hierarchical_init=True,          # Enable hierarchical initialization
    verbose=True
)
```

### 3. Disable Hierarchical Initialization (Pure Random for Novel Codes)

```python
# If you prefer random initialization for novel codes
stats = integrate_clinvec_with_exmedbert(
    model=model,
    code_dict=code_dict,
    clinvec_dir="path/to/ClinVec",
    use_hierarchical_init=False,  # Disable hierarchical initialization
    verbose=True
)
```

## Integration Process

### Step 1: Direct Code Matching
The integration tries multiple code format variants to match ClinVec codes with your vocabulary:

```
ClinVec Code: "250.0" (diabetes)
Tries matching:
- "250.0"           # Exact match
- "2500"            # Without decimal
- "ICD_250.0"       # With ICD prefix  
- "ICD9CM_250.0"    # With full vocabulary prefix
```

### Step 2: Hierarchical Initialization (NEW!)
For novel codes not found in ClinVec, the system uses **hierarchical initialization**:

```
Novel Code: "E11.321" (Type 2 diabetes with mild nonproliferative diabetic retinopathy)
Hierarchy:  E11.321 → E11.32 → E11.3 → E11
            
If E11.3 exists in ClinVec:
- Initialize E11.321 = E11.3_embedding + small_noise
- This gives meaningful starting point instead of random initialization
```

**Why This Works:**
- **ICD-10**: `E11.321` (specific complication) is semantically similar to `E11.3` (general retinopathy)  
- **ICD-9**: `250.01` (Type I diabetes) shares meaning with `250.0` (general diabetes)
- **Better convergence**: Novel codes start with relevant medical knowledge

### Step 3: Dimension Handling
- **ClinVec embeddings**: 128 dimensions
- **Your model**: Configurable (e.g., 288 in original ExMed-BERT)

Options:
- **Resize automatically**: Truncate or pad embeddings to match model
- **Fail on mismatch**: Ensure exact dimension matching

### Step 4: Preservation During Training
The integration modifies the model's `_init_weights` method to preserve pre-trained embeddings while allowing continued training for all codes.

## File Structure

```
exmed_bert/utils/clinvec_integration.py  # Main integration code
scripts/integrate_clinvec.py             # Example/test script
docs/ClinVec_Integration.md             # This documentation
```

## Expected Performance Gains

With ClinVec integration, you should expect:

1. **Faster Convergence**: Pre-trained codes start with meaningful representations
2. **Better Rare Code Handling**: Known codes benefit from large-scale pre-training  
3. **Improved Transfer Learning**: Rich medical knowledge from ClinVec dataset
4. **Hierarchical Novel Code Learning**: Unknown codes start with related parent knowledge instead of random noise

## Example Integration Results

```
=== Loading ICD10CM embeddings ===
Resizing icd10cm embeddings: 128 → 288
  ✓ E11.65 → E11.65 (idx: 1247)
  ✓ I10 → I10 (idx: 2156)
  ✓ Z51.11 → Z51.11 (idx: 3891)
  Direct matches: 8,943/15,234 icd10cm embeddings
  Direct coverage: 58.7%

=== Hierarchical Initialization ===
  ◐ E11.321 initialized from hierarchy (parent: E11.3)
  ◐ I10.1 initialized from hierarchy (parent: I10)
  ◐ Z51.12 initialized from hierarchy (parent: Z51.1)
  Hierarchical initializations: 2,847

=== Integration Complete ===
Total embeddings loaded: 11,790
  icd10cm: 8,943
  hierarchical_init: 2,847
```

**What This Means:**
- **58.7%** of your ICD-10 codes got exact ClinVec embeddings
- **Additional 2,847** novel codes got hierarchical initialization 
- **Total coverage**: ~77% of codes start with meaningful representations
- **Remaining codes**: Use standard random initialization

## Troubleshooting

### Low Coverage (Few Embeddings Loaded)

**Problem**: Only a small percentage of ClinVec embeddings match your vocabulary.

**Solutions**:
1. **Check code formats** in your `CodeDict`:
   ```python
   print("First 10 codes in your vocabulary:")
   print(list(code_dict.stoi.keys())[:10])
   ```

2. **Inspect ClinVec codes**:
   ```python
   from exmed_bert.utils.clinvec_integration import ClinVecLoader
   loader = ClinVecLoader("path/to/ClinVec")
   icd_codes = loader.load_embeddings_by_vocab("icd10cm")
   print("First 10 ClinVec ICD-10 codes:")
   print(list(icd_codes.keys())[:10])
   ```

3. **Add custom code mapping** in the integration function

### Dimension Mismatch

**Problem**: ClinVec (128D) vs your model (different dimension)

**Solutions**:
1. **Enable automatic resizing**:
   ```python
   stats = integrate_clinvec_with_exmedbert(..., resize_if_needed=True)
   ```

2. **Change model dimension** to 128 to match ClinVec exactly
3. **Use PCA/projection** for more sophisticated dimension matching

### Memory Issues

**Problem**: Loading multiple large vocabulary embeddings

**Solutions**:
1. **Load vocabularies selectively**:
   ```python
   vocab_types=["icd10cm"]  # Instead of all vocabularies
   ```

2. **Process in batches** for very large models

## Advanced Usage

### Custom Code Preprocessing

```python
def preprocess_codes_for_matching(clinvec_codes, your_vocab):
    """Custom function to improve code matching"""
    mapping = {}
    for clinvec_code in clinvec_codes:
        # Your custom logic to map ClinVec codes to your vocabulary
        matched_code = your_mapping_logic(clinvec_code)
        if matched_code in your_vocab:
            mapping[clinvec_code] = matched_code
    return mapping
```

### Selective Vocabulary Loading

```python
# For COVID-19 research, focus on relevant codes
covid_relevant_vocabs = ["icd10cm", "phecode", "snomedct"]

stats = integrate_clinvec_with_exmedbert(
    model=model,
    code_dict=code_dict,
    clinvec_dir=clinvec_dir,
    vocab_types=covid_relevant_vocabs
)
```

## Testing Your Integration

Run the test script to verify integration:

```bash
# Test with ICD-9 and ICD-10 codes
python scripts/integrate_clinvec.py \
    --clinvec_dir path/to/ClinVec \
    --test_only \
    --vocab_types icd9cm icd10cm

# Test with all available vocabularies  
python scripts/integrate_clinvec.py \
    --clinvec_dir path/to/ClinVec \
    --test_only \
    --vocab_types icd9cm icd10cm phecode rxnorm atc snomedct
```

## Integration in Training Scripts

### Pre-training Script

```python
# In scripts/pretrain-exmed-bert.py

from exmed_bert.utils.clinvec_integration import load_for_exmedbert_pretraining

def main():
    # ... existing setup code ...
    
    model = ExMedBertModel(config)
    
    # NEW: Integrate ClinVec embeddings
    if args.use_clinvec:
        stats = load_for_exmedbert_pretraining(
            model=model,
            code_dict=code_dict,
            clinvec_dir=args.clinvec_dir
        )
        logger.info(f"Loaded {sum(stats.values())} ClinVec embeddings")
    
    # ... rest of training code ...
```

### Fine-tuning Script

```python
# In scripts/finetune-exmed-bert.py

from exmed_bert.utils.clinvec_integration import load_for_exmedbert_finetuning

def main():
    # ... existing setup code ...
    
    if args.use_clinvec:
        stats = load_for_exmedbert_finetuning(
            model=model,
            code_dict=code_dict,
            clinvec_dir=args.clinvec_dir,
            focus_vocab="icd10cm"  # Focus on specific vocabulary for task
        )
    
    # ... rest of fine-tuning code ...
```

## References

- **ClinVec Paper**: [Insert paper reference when available]
- **Harvard Dataverse**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Z6H1A8
- **ExMed-BERT Paper**: [Your ExMed-BERT paper reference]