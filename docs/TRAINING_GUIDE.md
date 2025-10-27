# ExMed-BERT Training Guide

Complete guide for training ExMed-BERT with synthetic data and ClinVec embeddings.

---

## Quick Reference

| Task | Command |
|------|---------|
| **Interactive training** | `.\train_interactive.bat` |
| **Automated training** | `.\train.bat` |
| **Generate data** | `python scripts/generate_synthetic_patients.py -n 1000` |
| **Convert dataset** | `python scripts/convert_synthetic_to_dataset.py --input data/synthetic_patients.json` |
| **Train with ClinVec** | `python scripts/pretrain-exmed-bert-clinvec.py --use-clinvec` |
| **Fine-tune model** | `python scripts/finetune-exmed-bert.py` |

---

## Training Modes

### Interactive Mode (Recommended for First-Time Users)

```bash
.\train_interactive.bat
```

**Features:**
- Step-by-step prompts for data generation and ClinVec setup
- Auto-detects available ClinVec files
- Reads model/training configuration from config.yaml
- Validates paths and files
- Shows complete configuration summary for final approval

**Example Interaction:**
```
==========================================
STEP 1: Data Generation
==========================================

Generate new synthetic data? (y/n) [y]: y
Number of patients to generate [1000]: 500
Novel code probability (0.0-1.0) [0.01]: 0.02

==========================================
STEP 2: Dataset Conversion
==========================================

Maximum sequence length [512]: 512
Training set ratio (0.0-1.0) [0.8]: 0.8

==========================================
STEP 3: ClinVec Embeddings
==========================================

Use ClinVec pre-trained embeddings? (y/n) [y]: y

Detecting available ClinVec vocabulary files...
  [FOUND] ICD-10 CM (ClinVec_icd10cm.csv)
  Include ICD-10 codes? (y/n) [y]: y
  [FOUND] ATC Drug Codes (ClinVec_atc.csv)
  Include ATC drug codes? (y/n) [y]: y
  [FOUND] CPT Procedure Codes (ClinVec_cpt.csv)
  Include CPT codes? (y/n) [y]: y

Selected vocabularies: icd10cm,atc,cpt

Use hierarchical initialization for novel codes? (y/n) [y]: y
Embedding resize method [auto]: auto

==========================================
STEP 4: Review Training Configuration
==========================================

Reading configuration from config.yaml...

Configuration Summary:

  Model Architecture:
    - Layers: 2
    - Hidden size: 64
    - Intermediate size: 128
    - Attention heads: 2

  Training Parameters:
    - Epochs: 2
    - Batch size: 1
    - Learning rate: 1e-4
    - Max sequence length: 512

  Data:
    - Training data: pretrain_stuff/synthetic_train.pt
    - Output directory: output\model

  ClinVec Settings:
    - Status: ENABLED
    - Directory: ClinVec
    - Vocabularies: icd10cm,atc,cpt
    - Hierarchical init: y

NOTE: To change model/training settings, edit config.yaml

Proceed with training? (y/n) [y]:
```

---

### Automated Mode (Recommended for Repeated Training)

```bash
.\train.bat
```

**Features:**
- Uses settings from `config.yaml`
- No prompts or interaction
- Ideal for batch processing
- Reproducible results

**Configuration:**
Edit `config.yaml` to customize:

```yaml
# Data generation
num_patients: 1000
novel_prob: 0.01

# Model architecture
num_hidden_layers: 2
hidden_size: 64

# Training
batch_size: 2
num_epochs: 3
learning_rate: 0.0001

# ClinVec
clinvec_params:
  use_clinvec: true
  clinvec_dir: "ClinVec"
  vocab_types: ["icd10cm", "atc", "cpt"]
  use_hierarchical_init: true
```

---

## Training Pipeline

### Step 1: Data Generation

Generate synthetic patient data with medical codes:

```bash
python scripts/generate_synthetic_patients.py \
    --num-patients 1000 \
    --novel-prob 0.01 \
    --output data/synthetic_patients.json
```

**Parameters:**
- `--num-patients` (default: 1000) - Number of patients to generate
- `--novel-prob` (default: 0.01) - Probability of novel codes (1%)
- `--output` (default: data/synthetic_patients.json) - Output file path
- `--seed` (optional) - Random seed for reproducibility

**Novel Codes:**
Novel codes are medical codes not in the ClinVec embedding files. The system uses hierarchical initialization to find parent codes and initialize novel code embeddings.

**Output:**
```json
{
  "patients": [
    {
      "patient_id": "P0001",
      "diagnoses": [
        {"code": "E11.65", "date": "2020-01-15"},
        {"code": "I10", "date": "2020-02-20"}
      ],
      "prescriptions": [
        {"code": "N02BE01", "date": "2020-01-16"}
      ]
    }
  ]
}
```

---

### Step 2: Dataset Conversion

Convert JSON to PyTorch dataset format:

```bash
python scripts/convert_synthetic_to_dataset.py \
    --input data/synthetic_patients.json \
    --output pretrain_stuff/synthetic.pt \
    --split all \
    --max-length 512 \
    --train-ratio 0.8 \
    --seed 42
```

**Parameters:**
- `--input` - Path to JSON file from Step 1
- `--output` - Output PyTorch dataset path
- `--split` - Which split to create: `all`, `train`, `val`, `test`
- `--max-length` - Maximum sequence length (default: 512)
- `--train-ratio` - Train/val split ratio (default: 0.8)
- `--seed` - Random seed

**Output:**
Creates `*.pt` files containing PatientDataset objects:
- `synthetic_train.pt` - Training data
- `synthetic_val.pt` - Validation data

---

### Step 3: Model Training

#### Option A: With ClinVec Embeddings (Recommended)

```bash
python scripts/pretrain-exmed-bert-clinvec.py \
    --data-dir pretrain_stuff \
    --output-dir output/with_clinvec \
    --use-clinvec \
    --clinvec-dir ClinVec \
    --vocab-types icd10cm,atc,cpt \
    --use-hierarchical-init \
    --resize-method auto \
    --num-hidden-layers 2 \
    --hidden-size 64 \
    --num-epochs 3 \
    --batch-size 2 \
    --learning-rate 1e-4
```

**ClinVec Parameters:**
- `--use-clinvec` - Enable ClinVec pre-trained embeddings
- `--clinvec-dir` - Path to ClinVec directory
- `--vocab-types` - Comma-separated vocabulary types: `icd10cm`, `atc`, `cpt`, `rxnorm`, `phecode`
- `--use-hierarchical-init` - Use parent embeddings for novel codes
- `--resize-method` - How to handle dimension mismatches:
  - `auto` - Automatically choose best method
  - `truncate` - Cut off extra dimensions
  - `pca` - PCA dimensionality reduction
  - `learned_projection` - Learned projection layer
  - `pad_smart` - Pad with learned values
  - `pad_random` - Pad with random values

**Hierarchical Initialization:**
For novel codes not in ClinVec:
- **ICD-10:** E11.65 → E11.6 → E11 (truncate digits)
- **ATC5:** N02BE01 → N02BE → N02B → N02 → N (5-level hierarchy)
- **CPT:** No hierarchy (random initialization if not found)

#### Option B: Without ClinVec (Random Initialization)

```bash
python scripts/pretrain-exmed-bert.py \
    --data-dir pretrain_stuff \
    --output-dir output/random_init \
    --num-hidden-layers 2 \
    --hidden-size 64 \
    --num-epochs 3 \
    --batch-size 2 \
    --learning-rate 1e-4
```

---

## Model Architecture

### Model Size Presets

| Preset | Layers | Hidden Size | Intermediate | Attention Heads | Parameters |
|--------|--------|-------------|--------------|-----------------|------------|
| Tiny | 1 | 32 | 64 | 1 | ~10K |
| Small | 2 | 64 | 128 | 2 | ~40K |
| Medium | 4 | 128 | 256 | 4 | ~200K |
| Large | 6 | 288 | 576 | 6 | ~1M |

### Custom Architecture

```bash
python scripts/pretrain-exmed-bert-clinvec.py \
    --num-hidden-layers 4 \
    --hidden-size 256 \
    --intermediate-size 512 \
    --num-attention-heads 8 \
    --max-position-embeddings 512 \
    --hidden-dropout-prob 0.1 \
    --attention-probs-dropout-prob 0.1
```

---

## Training Parameters

### Batch Size and Memory

| Batch Size | GPU Memory | Speed | Recommendation |
|------------|------------|-------|----------------|
| 1 | ~2 GB | Slow | Testing only |
| 2 | ~4 GB | Moderate | Default, good for most GPUs |
| 4 | ~8 GB | Fast | If you have 8GB+ GPU |
| 8 | ~16 GB | Very fast | High-end GPUs only |

**Out of memory?**
- Reduce batch size
- Use smaller model (Tiny or Small)
- Reduce max sequence length
- Use gradient accumulation

### Learning Rate

**Recommended values:**
- `1e-4` - Default, good starting point
- `5e-5` - More stable, slower convergence
- `2e-4` - Faster convergence, risk of instability
- `1e-5` - Very stable, very slow (fine-tuning)

**Learning rate schedule:**
```bash
--warmup-ratio 0.1  # 10% of training for warmup
```

### Epochs

**Recommendation:**
- **Testing:** 1-2 epochs
- **Development:** 3-5 epochs
- **Production:** 10-20 epochs

---

## Output and Checkpoints

### Training Output

```
output/
├── checkpoint-100/      # Checkpoint every N steps
│   ├── pytorch_model.bin
│   ├── config.json
│   └── training_args.bin
├── checkpoint-200/
├── runs/                # TensorBoard logs
│   └── ...
├── trainer_state.json   # Training state
└── training_args.json   # Training arguments
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir output/runs

# Check training logs
tail -f output/training.log
```

---

## Advanced Features

### MLflow Tracking

```bash
python scripts/pretrain-exmed-bert-clinvec.py \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-experiment-name exmed-bert-pretraining
```

### Dynamic Masking

```bash
--dynamic-masking  # Mask different tokens each epoch
```

### PLOS Prediction

```bash
--plos              # Predict prolonged length of stay
--plos-threshold 7  # Days threshold for PLOS
```

### Initialization Methods

```bash
--initialization orthogonal  # Orthogonal initialization (default)
--initialization xavier      # Xavier initialization
--initialization kaiming     # Kaiming initialization
```

---

## Fine-Tuning

After pre-training, fine-tune for specific tasks:

```bash
python scripts/finetune-exmed-bert.py \
    --model-path output/with_clinvec/checkpoint-best \
    --task-data data/task_specific.csv \
    --output-dir output/finetuned \
    --num-labels 2 \
    --metric-for-best-model f1
```

---

## Baseline Models

Compare ExMed-BERT against traditional ML models:

### Random Forest

```bash
python scripts/train-rf.py \
    --data-dir pretrain_stuff \
    --output-dir output/rf
```

### XGBoost

```bash
python scripts/train-xgboost.py \
    --data-dir pretrain_stuff \
    --output-dir output/xgboost
```

---

## Troubleshooting

### Training is very slow
- **Solution:** Increase batch size if GPU memory allows
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is enabled: `python -c "import torch; print(torch.cuda.is_available())"`

### Loss is not decreasing
- **Solution:** Lower learning rate (try 1e-5)
- Check data quality with `scripts/example_load_synthetic_data.py`
- Ensure sufficient training epochs

### Out of memory errors
- **Solution:** Reduce batch size to 1
- Use smaller model size (Tiny)
- Reduce max sequence length to 256 or 128

### ClinVec embeddings not loading
- **Solution:** Verify file paths
- Check vocabulary types match file names
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#clinvec-issues)

---

## Best Practices

1. **Start small:** Test with Tiny model and 100 patients first
2. **Use ClinVec:** Pre-trained embeddings significantly improve performance
3. **Enable hierarchical init:** Helps with novel codes
4. **Monitor training:** Use TensorBoard to watch loss curves
5. **Save checkpoints:** Enable checkpoint saving every N steps
6. **Validate early:** Use validation set to catch overfitting
7. **Document experiments:** Use MLflow or similar tracking

---

## Next Steps

- **Examples:** See `examples/` folder for usage patterns
- **Tests:** Run `pytest` in `tests/` folder to verify setup
- **ClinVec Guide:** [docs/ClinVec_Integration.md](ClinVec_Integration.md) for details
- **Synthetic Data:** [docs/SYNTHETIC_DATA_GUIDE.md](SYNTHETIC_DATA_GUIDE.md) for technical info

---

**Ready to train?** Run `.\train_interactive.bat` to get started!
