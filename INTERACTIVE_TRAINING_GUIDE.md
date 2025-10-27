# Interactive Training Guide

This guide explains how to use the interactive training batch file (`train_interactive.bat`).

## Quick Start

```bash
# From Anaconda Prompt or VS Code terminal
cd C:\Users\danie\Desktop\Git\PRL
train_interactive.bat
```

The script will guide you through all configuration options with interactive prompts.

## Step-by-Step Walkthrough

### Step 1: Data Configuration

**Question: Generate new synthetic data?**
- `y` - Generate new synthetic patient data
- `n` - Use existing patient data file

#### If generating new data (Option: y)

You'll be asked for:
- **Number of patients** [1000]: How many synthetic patients to create
- **Novel code probability** [0.01]: Percentage of codes that are novel (0.01 = 1%)
- **Output JSON file** [data\synthetic_patients.json]: Where to save generated data
- **Random seed** [42]: For reproducible results

Example:
```
Number of patients to generate [1000]: 5000
Novel code probability (0.0-1.0) [0.01]: 0.02
Output JSON file [data\synthetic_patients.json]:
Random seed for reproducibility [42]: 123
```

#### If using existing data (Option: n)

The script will:
1. **Auto-detect** common data file locations:
   - `data\synthetic_patients.json`
   - `data\patients.json`
2. **Prompt you** to confirm or enter a different path

Example:
```
Found: data\synthetic_patients.json
Patient data JSON file [data\synthetic_patients.json]:
```

### Step 2: Dataset Conversion Configuration

You'll be asked for:
- **Maximum sequence length** [50]: How many medical codes per patient sequence
  - Use 50 for demo/testing
  - Use 512 for production
- **Training set ratio** [0.8]: Train/validation split (0.8 = 80% train, 20% val)
- **Dataset output directory** [pretrain_stuff]: Where to save .pt files

Example:
```
Maximum sequence length [50]: 100
Training set ratio (0.0-1.0) [0.8]: 0.9
Dataset output directory [pretrain_stuff]: my_datasets
```

### Step 3: ClinVec Embeddings Configuration

**Question: Use ClinVec pre-trained embeddings? (Default: y)**
- `y` - Use ClinVec embeddings (better performance, requires ClinVec data) **[DEFAULT]**
- `n` - Use random initialization (simpler, no external data needed)

#### If using ClinVec (Option: y)

The script will:
1. **Auto-detect** ClinVec location:
   - `ClinVec\`
2. **Validate** the directory contains required files (`ClinGraph_nodes.csv`)
3. **Auto-detect available vocabularies** by checking for ClinVec CSV files

You'll then be asked for:
- **ClinVec data directory**: Path to ClinVec dataset
- **Vocabulary selection**: For each available vocabulary file, you'll be asked y/n:
  - `icd10cm` - ICD-10 diagnosis codes (supports hierarchical init!)
  - `atc` - ATC drug codes (supports hierarchical init!)
  - `rxnorm` - RxNorm drug codes
  - `phecode` - PheWAS codes
  - `cpt` - CPT procedure codes
  - The script only shows vocabularies with existing CSV files in your ClinVec directory
- **Hierarchical initialization** [y]: Use parent embeddings for novel codes
- **Resize method** [auto]: How to handle dimension mismatches
  - `auto` - Automatically choose best method
  - `truncate` - Cut off extra dimensions
  - `pca` - Use PCA for dimensionality reduction
  - `learned_projection` - Learn a projection layer
  - `pad_smart` - Pad with learned values
  - `pad_random` - Pad with random values

Example:
```
ClinVec data directory [ClinVec]: D:\datasets\ClinVec

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
Embedding resize method [auto]: pca
```

### Step 4: Training Configuration

**Select model size:**
1. Tiny (1 layer, 32 hidden) - Very fast, minimal accuracy
2. Small (2 layers, 64 hidden) - **DEFAULT** - Good for testing
3. Medium (4 layers, 128 hidden) - Balanced
4. Large (6 layers, 288 hidden) - Production quality
5. Custom - Specify your own architecture

You'll then be asked for:
- **Number of epochs** [2]: Training iterations over the full dataset
- **Batch size** [1]: Samples per training step (increase if you have GPU)
- **Learning rate** [1e-4]: Step size for optimizer
- **Output directory** [output\model]: Where to save trained model

Example:
```
Select model size [2]: 3
Number of epochs [2]: 10
Batch size [1]: 4
Learning rate [1e-4]: 5e-5
Output directory [output\model]: output\my_experiment
```

### Step 5: Training

The script will:
1. Show a complete summary of all settings
2. Ask for final confirmation
3. Run the training pipeline
4. Display progress and results

## Example Complete Session

```
==========================================
ExMed-BERT Interactive Training Pipeline
==========================================

Generate new synthetic data? (y/n) [y]: y

--- Synthetic Data Generation Settings ---
Number of patients to generate [1000]: 2000
Novel code probability (0.0-1.0) [0.01]:
Output JSON file [data\synthetic_patients.json]:
Random seed for reproducibility [42]:

Summary:
  Patients: 2000
  Novel code probability: 0.01
  Output: data\synthetic_patients.json
  Random seed: 42

Proceed with generation? (y/n) [y]: y

==========================================
STEP 2: Dataset Conversion Configuration
==========================================

Maximum sequence length [50]:
Training set ratio (0.0-1.0) [0.8]:
Dataset output directory [pretrain_stuff]:

==========================================
STEP 3: ClinVec Embeddings Configuration
==========================================

Use ClinVec pre-trained embeddings? (y/n) [y]:

==========================================
STEP 4: Training Configuration
==========================================

Model size presets:
  1. Tiny   (1 layer, 32 hidden)
  2. Small  (2 layers, 64 hidden) - DEFAULT
  3. Medium (4 layers, 128 hidden)
  4. Large  (6 layers, 288 hidden)
  5. Custom

Select model size [2]:

Number of epochs [2]: 5
Batch size [1]:
Learning rate [1e-4]:
Output directory [output\model]: output\test_run

Training Summary:
  Model: 2 layers, 64 hidden, 2 heads
  Epochs: 5
  Batch size: 1
  Learning rate: 1e-4
  Max sequence length: 50
  Output: output\test_run\
  ClinVec: DISABLED

Start training? (y/n) [y]: y

[Training begins...]
```

## Tips

1. **Press Enter** to accept default values (shown in brackets)
2. **Use arrows** to navigate back if you make a mistake (then press Ctrl+C to restart)
3. **Check auto-detected paths** before confirming them
4. **Start small** - Use default settings for your first run
5. **Save your settings** - Make notes of configurations that work well

## Comparison: Original vs Interactive

| Feature | train.bat | train_interactive.bat |
|---------|-----------|----------------------|
| Data source | Always generates new | Choose generate or existing |
| Configuration | Command line args only | Interactive prompts |
| ClinVec setup | Manual command editing | Guided configuration |
| Validation | None | Checks file existence |
| User-friendly | Moderate | High |
| Automation | Easy | Requires input |

## When to Use Which

**Use `train.bat`:**
- Automated scripts/pipelines
- You know exact parameters
- Running experiments in batch

**Use `train_interactive.bat`:**
- First time setup
- Exploring different configurations
- When you're unsure about paths
- Learning the system

## Troubleshooting

**"Could not activate environment"**
- Run from Anaconda Prompt instead of regular command prompt
- Or run: `conda init cmd.exe` first

**"File not found" errors**
- Double-check paths when prompted
- Use absolute paths if relative paths don't work
- Make sure ClinVec data is downloaded if using that option

**Training fails**
- Check the log file path shown in error message
- Reduce batch size if out of memory
- Reduce model size if training is too slow

**ClinVec directory not recognized**
- Ensure `ClinGraph_nodes.csv` exists in the directory
- Check that vocabulary CSV files exist (e.g., `ClinVec_icd10cm.csv`)
