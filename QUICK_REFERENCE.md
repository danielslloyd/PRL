# Quick Reference - Synthetic Data Training

## One-Line Command (Windows)

```bash
scripts\train_on_synthetic_data.bat
```

That's it! This runs the complete pipeline.

---

## Manual Steps (If You Want Control)

### 1. Generate Data
```bash
py scripts\generate_synthetic_patients.py -n 1000
```

### 2. Convert to Dataset
```bash
py scripts\convert_synthetic_to_dataset.py ^
    --input data\synthetic_patients.json ^
    --output pretrain_stuff\synthetic.pt ^
    --split all ^
    --max-length 50
```

### 3. Train
```bash
py scripts\pretrain-exmed-bert-clinvec.py ^
    --training-data pretrain_stuff\synthetic_train.pt ^
    --validation-data pretrain_stuff\synthetic_val.pt ^
    --output-dir output\synthetic_pretrain ^
    --output-data-dir output\synthetic_pretrain_data ^
    --max-seq-length 50 ^
    --train-batch-size 2 ^
    --epochs 10 ^
    --dynamic-masking
```

---

## Common Variations

### More patients
```bash
scripts\train_on_synthetic_data.bat 5000
```

### More novel codes (2%)
```bash
scripts\train_on_synthetic_data.bat 1000 0.02
```

### Longer sequences
```bash
scripts\train_on_synthetic_data.bat 1000 0.01 100
```

### Different random seed
```bash
scripts\train_on_synthetic_data.bat 1000 0.01 50 123
```

---

## File Locations

**Input:** `data/synthetic_patients.json`
**Datasets:** `pretrain_stuff/synthetic_train.pt` and `synthetic_val.pt`
**Model:** `output/synthetic_pretrain/`
**Logs:** `output/synthetic_pretrain_data/`

---

## Key Parameters

| Parameter | Generate | Convert | Train |
|-----------|----------|---------|-------|
| **Number of patients** | `-n 1000` | - | - |
| **Novel code %** | `-p 0.01` | - | - |
| **Max sequence length** | - | `--max-length 50` | `--max-seq-length 50` |
| **Random seed** | `-s 42` | `--seed 42` | `--seed 42` |
| **Batch size** | - | - | `--train-batch-size 2` |
| **Epochs** | - | - | `--epochs 10` |

⚠️ **Max length must match** between Convert and Train steps!

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Use `py` instead of `python` |
| "No module named 'torch'" | Install dependencies: `conda env create -f environment.yaml` |
| "Out of memory" | Reduce batch size: `--train-batch-size 1` |
| "max_length mismatch" | Use same value in Convert and Train |

---

## Check Your Results

```bash
# View training log
type output\synthetic_pretrain\output.log

# View evaluation results
type output\synthetic_pretrain_data\eval_results.txt

# List saved models
dir output\synthetic_pretrain
```

---

## Full Documentation

- **Complete Guide:** [TRAINING_WITH_SYNTHETIC_DATA.md](TRAINING_WITH_SYNTHETIC_DATA.md)
- **Setup Help:** [HOW_TO_RUN.md](HOW_TO_RUN.md)
- **Data Details:** [SYNTHETIC_DATA_README.md](SYNTHETIC_DATA_README.md)
