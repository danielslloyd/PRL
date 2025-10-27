# Training Batch Files Summary

## Available Scripts

### 1. `train.bat` - Automated Training
**Best for:** Scripting, automation, known configurations

**Usage:**
```bash
train.bat [n_patients] [novel_prob] [max_length] [seed]

# Examples:
train.bat                           # Use all defaults
train.bat 5000                     # 5000 patients, other defaults
train.bat 5000 0.02 100 42        # All custom parameters
```

**Features:**
- ✅ Fully automated (no user input required)
- ✅ Command-line arguments
- ✅ Auto-installs dependencies
- ✅ Always generates new synthetic data
- ✅ Fixed ClinVec settings (disabled by default)
- ⚠️ No validation of file paths
- ⚠️ Requires editing file to change ClinVec settings

---

### 2. `train_interactive.bat` - Interactive Training (NEW!)
**Best for:** First-time use, exploration, configuration testing

**Status:** ✅ Fixed and tested - all syntax errors resolved

**Usage:**
```bash
train_interactive.bat
# Then follow the prompts!
```

**Features:**
- ✅ Interactive prompts for all settings
- ✅ Choose to generate OR use existing data
- ✅ Auto-detects common file locations
- ✅ Validates file/directory existence
- ✅ Guided ClinVec configuration
- ✅ Model size presets (Tiny/Small/Medium/Large/Custom)
- ✅ Shows summary before starting
- ✅ Helpful defaults for all options
- ⚠️ Requires user input (not suitable for automation)

**Interactive Options:**

**Step 1: Data Configuration**
- Generate new synthetic data or use existing
- If generating: patients, novel probability, output path
- If existing: auto-detect or specify path

**Step 2: Dataset Conversion**
- Maximum sequence length
- Train/validation split ratio
- Output directory

**Step 3: ClinVec Embeddings**
- Enable/disable ClinVec
- Auto-detect ClinVec directory
- Choose vocabulary types (icd10cm, atc, phecode, rxnorm, etc.)
- Hierarchical initialization option
- Embedding resize method

**Step 4: Training Configuration**
- Model size preset or custom architecture
- Number of epochs
- Batch size
- Learning rate
- Output directory

---

## Quick Comparison

| Feature | train.bat | train_interactive.bat |
|---------|-----------|----------------------|
| **User Input** | None (uses defaults) | Interactive prompts |
| **Data Source** | Always generates new | Generate OR existing |
| **ClinVec** | Requires editing file | Guided configuration |
| **Validation** | None | Validates paths |
| **Best For** | Automation | First-time/exploration |
| **Speed** | Fast (no waiting) | Slower (prompts) |
| **Flexibility** | Medium | High |

---

## Common Workflows

### First Time Training
```bash
# Use interactive version
train_interactive.bat

# Follow prompts:
# 1. Generate data: y
# 2. Use defaults (just press Enter)
# 3. ClinVec: n (for first run)
# 4. Model: 2 (Small)
# 5. Accept defaults
```

### Quick Experiments
```bash
# Use automated version
train.bat 1000 0.01 50 42
```

### Training with ClinVec
```bash
# Use interactive version for easy setup
train_interactive.bat

# Configure:
# 1. Use existing data: n → y
# 2. ClinVec: y
# 3. Point to your ClinVec directory
# 4. Choose vocabularies: icd10cm,atc
```

### Production Training
```bash
# Edit train.bat for your exact settings, then:
train.bat 50000 0.01 512 42
```

---

## Default Settings Reference

### Data Generation
- **Patients:** 1000
- **Novel code probability:** 0.01 (1%)
- **Random seed:** 42

### Dataset Conversion
- **Max sequence length:** 50 (use 512 for production)
- **Train/val split:** 0.8 (80% train, 20% validation)

### Training
- **Model:** Small (2 layers, 64 hidden, 2 heads)
- **Epochs:** 2
- **Batch size:** 1
- **Learning rate:** 1e-4

### ClinVec
- **Enabled:** No (by default in train.bat)
- **Vocabularies:** icd10cm,atc
- **Hierarchical init:** Yes (when ClinVec enabled)
- **Resize method:** auto

---

## Tips

1. **Start with interactive** for your first run to understand all options
2. **Use automated** once you know your preferred settings
3. **ClinVec is optional** - start without it to ensure basic pipeline works
4. **Batch size = 1** is safe but slow; increase if you have GPU
5. **Max length = 50** is good for testing; use 512 for real models
6. **Model size** affects training time dramatically:
   - Tiny: ~1 min for 1000 patients
   - Small: ~3 min for 1000 patients
   - Medium: ~10 min for 1000 patients
   - Large: ~30 min for 1000 patients

---

## Files Created by Training

Both scripts create the same output files:

```
data/
  └── synthetic_patients.json          # Generated patient data (if generated)

pretrain_stuff/
  ├── synthetic_train.pt               # Training dataset
  └── synthetic_val.pt                 # Validation dataset

output/
  └── [model_name]/                    # Your specified output dir
      ├── output.log                   # Training log
      ├── config.json                  # Model configuration
      ├── pytorch_model.bin            # Trained weights
      └── ...                          # Other checkpoint files

  └── [model_name]_data/               # Training data dir
      ├── mlruns.db                    # MLflow tracking database
      └── ...                          # Evaluation results
```

---

## Troubleshooting

**Script won't run:**
- Use Anaconda Prompt instead of regular CMD
- Or initialize conda: `conda init cmd.exe`

**"Module not found" errors:**
- Scripts auto-install dependencies
- If issues persist: `conda activate exmed-bert` first

**Out of memory:**
- Reduce batch size to 1
- Use smaller model size
- Reduce max_seq_length

**ClinVec not working:**
- Ensure directory contains `ClinGraph_nodes.csv`
- Check vocabulary CSV files exist
- Verify vocabulary types match available files
