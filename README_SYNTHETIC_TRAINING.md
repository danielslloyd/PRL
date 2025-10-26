# Quick Start: Train ExMed-BERT on Synthetic Data

## One-Time Setup (5-10 minutes)

### Install Miniconda

1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Get: **Miniconda3 Windows 64-bit**
3. Run installer (use default settings)
4. **Restart your terminal/Command Prompt**

That's it! The training script will handle the rest automatically.

---

## Run the Training Pipeline

```bash
# Navigate to the project
cd c:\Users\danie\Desktop\Git\PRL

# Run the complete pipeline
scripts\train_on_synthetic_data.bat
```

### What Happens Automatically

The script will:

**✅ Step 0:** Check for conda environment
- If missing, creates it automatically using `environment.yaml`
- Installs PyTorch, transformers, and all dependencies
- This only happens once (5-10 minutes)

**✅ Step 1:** Generate synthetic patients
- Creates 1000 patients with realistic medical data
- Includes 1% novel codes with hierarchical cousins

**✅ Step 2:** Convert to dataset format
- Builds vocabulary from synthetic data
- Splits into train (80%) and validation (20%)
- Saves as PyTorch `.pt` files

**✅ Step 3:** Train ExMed-BERT
- Runs masked language model pretraining
- Saves checkpoints every 50 steps
- Outputs trained model to `output/synthetic_pretrain/`

---

## Custom Parameters

```bash
# Syntax: scripts\train_on_synthetic_data.bat [patients] [novel%] [seq_len] [seed]

# Generate 2000 patients with 2% novel codes
scripts\train_on_synthetic_data.bat 2000 0.02

# Use longer sequences (100 instead of 50)
scripts\train_on_synthetic_data.bat 1000 0.01 100

# Different random seed
scripts\train_on_synthetic_data.bat 1000 0.01 50 123
```

---

## Output Files

After running, you'll find:

```
data/
  synthetic_patients.json          ← Raw synthetic data (JSON)

pretrain_stuff/
  synthetic_train.pt               ← Training dataset
  synthetic_val.pt                 ← Validation dataset

output/
  synthetic_pretrain/
    pytorch_model.bin              ← Trained model
    config.json                    ← Model config
    output.log                     ← Training log

  synthetic_pretrain_data/
    eval_results.txt               ← Final metrics
    logs/                          ← TensorBoard logs
```

---

## Troubleshooting

### "conda: command not found"
- Install Miniconda first (link above)
- **Restart your terminal** after installation
- Or use Anaconda Prompt from Start menu

### "Failed to activate conda environment"
Try manually:
```bash
conda activate exmed-bert
scripts\train_on_synthetic_data.bat
```

### Environment creation fails
Create manually:
```bash
cd c:\Users\danie\Desktop\Git\PRL
conda env create -f environment.yaml
```

### Still having issues?
See detailed guides:
- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - Full setup instructions
- [TRAINING_WITH_SYNTHETIC_DATA.md](TRAINING_WITH_SYNTHETIC_DATA.md) - Detailed training guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference

---

## What's Different Now?

**Before:** You had to manually:
1. Install conda
2. Create environment
3. Activate environment
4. Run each step separately

**Now:** Just run one command:
```bash
scripts\train_on_synthetic_data.bat
```

The script automatically:
- ✅ Checks if conda is installed
- ✅ Creates environment if missing
- ✅ Activates environment
- ✅ Runs all 3 steps
- ✅ Provides helpful error messages

---

## Next Steps

After training completes:

**View Results:**
```bash
# Check training log
type output\synthetic_pretrain\output.log

# Check metrics
type output\synthetic_pretrain_data\eval_results.txt
```

**Use the Model:**
```python
from exmed_bert.models import ExMedBertForMaskedLM

model = ExMedBertForMaskedLM.from_pretrained("output/synthetic_pretrain")
```

**Train with ClinVec:**
See [TRAINING_WITH_SYNTHETIC_DATA.md](TRAINING_WITH_SYNTHETIC_DATA.md) for ClinVec integration

---

## Summary

**Required:** Install Miniconda (one time, 5 minutes)

**To Train:** Run `scripts\train_on_synthetic_data.bat`

**Everything else is automatic!** 🚀
