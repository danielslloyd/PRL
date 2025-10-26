# Training ExMed-BERT with Synthetic Data

This guide shows you how to train ExMed-BERT using the synthetic patient data generator.

## Quick Start (3 Steps)

### Step 1: Generate Synthetic Patients

```bash
py scripts/generate_synthetic_patients.py -n 1000 -o data/synthetic_patients.json
```

This creates 1000 synthetic patients with realistic demographics and medical codes.

### Step 2: Convert to PatientDataset Format

```bash
py scripts/convert_synthetic_to_dataset.py \
    --input data/synthetic_patients.json \
    --output pretrain_stuff/synthetic.pt \
    --split all \
    --max-length 50
```

This converts the JSON data to the PyTorch format expected by ExMed-BERT.

### Step 3: Run Pretraining

```bash
py scripts/pretrain-exmed-bert-clinvec.py \
    --training-data pretrain_stuff/synthetic_train.pt \
    --validation-data pretrain_stuff/synthetic_val.pt \
    --output-dir output/synthetic_pretrain \
    --output-data-dir output/synthetic_pretrain_data \
    --train-batch-size 2 \
    --eval-batch-size 2 \
    --num-attention-heads 2 \
    --num-hidden-layers 2 \
    --hidden-size 64 \
    --intermediate-size 128 \
    --epochs 10 \
    --max-seq-length 50 \
    --dynamic-masking \
    --no-plos
```

---

## One-Command Pipeline (Windows)

We've created automated scripts that do all 3 steps:

```bash
# Windows (using batch file)
scripts\train_on_synthetic_data.bat

# With custom parameters
scripts\train_on_synthetic_data.bat 2000 0.02 100 42
#                                    ^    ^    ^   ^
#                               patients novel max seed
#                                       prob  len
```

---

## Understanding the Pipeline

### 1. Generate Synthetic Patients

**Script:** [scripts/generate_synthetic_patients.py](scripts/generate_synthetic_patients.py)

**What it does:**
- Creates N patients with realistic US demographics
- Generates medical events (ICD-10 diagnoses, ATC5 drugs)
- Adds 1% novel codes that are hierarchical cousins
- Saves to JSON format

**Key options:**
```bash
-n 1000         # Number of patients
-o data/out.json # Output path
-p 0.01         # Novel code probability (1%)
-s 42           # Random seed
```

**Output:** JSON file with patient records

### 2. Convert to PatientDataset

**Script:** [scripts/convert_synthetic_to_dataset.py](scripts/convert_synthetic_to_dataset.py)

**What it does:**
- Reads JSON patient data
- Creates vocabulary dictionaries (codes, age, sex, state)
- Converts each patient to a `Patient` object
- Splits into training (80%) and validation (20%) sets
- Saves as PyTorch `.pt` files

**Key options:**
```bash
--input data/synthetic_patients.json  # Input JSON
--output pretrain_stuff/synthetic.pt  # Output base path
--split all                           # Create train + val splits
--max-length 50                       # Max sequence length
--train-ratio 0.8                     # 80% train, 20% val
--dynamic-masking                     # Use dynamic masking
```

**Output:** Two `.pt` files:
- `pretrain_stuff/synthetic_train.pt` - Training dataset
- `pretrain_stuff/synthetic_val.pt` - Validation dataset

### 3. Run Pretraining

**Script:** [scripts/pretrain-exmed-bert-clinvec.py](scripts/pretrain-exmed-bert-clinvec.py)

**What it does:**
- Loads PatientDataset files
- Initializes ExMed-BERT model
- Optionally loads ClinVec embeddings
- Trains with masked language modeling
- Saves checkpoints and final model

**Key options:**
```bash
--training-data pretrain_stuff/synthetic_train.pt
--validation-data pretrain_stuff/synthetic_val.pt
--output-dir output/model             # Where to save model
--output-data-dir output/data         # Where to save logs
--max-seq-length 50                   # Must match conversion step
--dynamic-masking                     # Must match conversion step
--epochs 10                           # Number of training epochs
--train-batch-size 2                  # Batch size (small for demo)
--hidden-size 64                      # Model size (small for demo)
--num-hidden-layers 2                 # Number of layers
```

---

## Configuration Examples

### Small Test Run (Fast)
```bash
# Generate 100 patients
py scripts/generate_synthetic_patients.py -n 100

# Convert with short sequences
py scripts/convert_synthetic_to_dataset.py \
    --input data/synthetic_patients.json \
    --output pretrain_stuff/test.pt \
    --split all \
    --max-length 30

# Quick training run
py scripts/pretrain-exmed-bert-clinvec.py \
    --training-data pretrain_stuff/test_train.pt \
    --validation-data pretrain_stuff/test_val.pt \
    --output-dir output/test \
    --output-data-dir output/test_data \
    --max-seq-length 30 \
    --epochs 5 \
    --train-batch-size 4 \
    --hidden-size 32 \
    --num-hidden-layers 2
```

### Production Run
```bash
# Generate 10K patients with 2% novel codes
py scripts/generate_synthetic_patients.py -n 10000 -p 0.02

# Convert with full-length sequences
py scripts/convert_synthetic_to_dataset.py \
    --input data/synthetic_patients.json \
    --output pretrain_stuff/prod.pt \
    --split all \
    --max-length 512

# Full training
py scripts/pretrain-exmed-bert-clinvec.py \
    --training-data pretrain_stuff/prod_train.pt \
    --validation-data pretrain_stuff/prod_val.pt \
    --output-dir output/production \
    --output-data-dir output/production_data \
    --max-seq-length 512 \
    --epochs 40 \
    --train-batch-size 256 \
    --hidden-size 288 \
    --num-hidden-layers 6 \
    --learning-rate 3e-5 \
    --warmup-steps 10000
```

---

## Using ClinVec with Synthetic Data

To use ClinVec pre-trained embeddings:

```bash
py scripts/pretrain-exmed-bert-clinvec.py \
    --training-data pretrain_stuff/synthetic_train.pt \
    --validation-data pretrain_stuff/synthetic_val.pt \
    --output-dir output/with_clinvec \
    --output-data-dir output/with_clinvec_data \
    --max-seq-length 50 \
    --use-clinvec \
    --clinvec-dir ../ClinVec \
    --vocab-types "icd10cm" \
    --use-hierarchical-init \
    --resize-if-needed \
    --resize-method "pca"
```

**Benefits:**
- Novel codes get hierarchical initialization from their parent codes
- Better starting point for medical code embeddings
- Faster convergence

---

## Important: Matching Configurations

⚠️ **CRITICAL:** The `max_length` must match between steps 2 and 3!

```bash
# Step 2: Set max-length
py scripts/convert_synthetic_to_dataset.py --max-length 50 ...

# Step 3: Must use same max-seq-length
py scripts/pretrain-exmed-bert-clinvec.py --max-seq-length 50 ...
```

Also ensure `dynamic_masking` setting matches:
```bash
# Step 2: Use dynamic masking
py scripts/convert_synthetic_to_dataset.py --dynamic-masking ...

# Step 3: Must also use dynamic masking
py scripts/pretrain-exmed-bert-clinvec.py --dynamic-masking ...
```

---

## Output Files

After running the complete pipeline, you'll have:

```
data/
  synthetic_patients.json          # Raw synthetic data

pretrain_stuff/
  synthetic_train.pt               # Training dataset
  synthetic_val.pt                 # Validation dataset

output/
  synthetic_pretrain/
    pytorch_model.bin              # Trained model weights
    config.json                    # Model configuration
    checkpoint-*/                  # Training checkpoints
    output.log                     # Training log

  synthetic_pretrain_data/
    logs/                          # TensorBoard logs
    eval_results.txt               # Final evaluation metrics
    mlruns.db                      # MLflow tracking database
```

---

## Troubleshooting

### "No module named 'torch'"
You need to install PyTorch and other dependencies:
```bash
# Using conda (recommended)
conda env create -f environment.yaml
conda activate exmed-bert

# Or using pip
py -m pip install torch transformers typer mlflow joblib
```

### "max_length mismatch error"
Make sure the same `max_length` / `max_seq_length` is used in both conversion and training steps.

### Out of memory errors
Reduce batch size:
```bash
--train-batch-size 1 --eval-batch-size 1
```

Or reduce model size:
```bash
--hidden-size 32 --num-hidden-layers 2
```

### "File not found" errors
Make sure you run the scripts from the PRL directory:
```bash
cd c:\Users\danie\Desktop\Git\PRL
py scripts\...
```

---

## Next Steps

After training:

1. **Evaluate the model:**
   - Check `output/synthetic_pretrain_data/eval_results.txt`
   - View training curves in TensorBoard

2. **Fine-tune for specific tasks:**
   - Use the pretrained model for downstream tasks
   - See [scripts/finetune-exmed-bert.py](scripts/finetune-exmed-bert.py)

3. **Analyze novel code embeddings:**
   - Extract embeddings and visualize hierarchical structure
   - Compare novel codes with their parent codes

---

## Full Example Session

```bash
# Navigate to project directory
cd c:\Users\danie\Desktop\Git\PRL

# Run the complete pipeline with one command
scripts\train_on_synthetic_data.bat 1000 0.01 50 42

# The script will:
# 1. Generate 1000 patients
# 2. Convert to PatientDataset format
# 3. Train ExMed-BERT for 10 epochs
# 4. Save model to output/synthetic_pretrain

# Total time: ~10-30 minutes (depending on your hardware)
```

---

## See Also

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Basic Python setup
- [SYNTHETIC_DATA_README.md](SYNTHETIC_DATA_README.md) - Details on synthetic data generation
- [CLAUDE.md](CLAUDE.md) - Project overview and setup
- [config.yaml](config.yaml) - Configuration file reference
