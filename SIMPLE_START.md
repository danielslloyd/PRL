# Simple Start Guide (Avoiding Batch File Issues)

Having issues with the automated script? Here's a simplified 2-step approach:

---

## Step 1: Set Up Environment (One Time Only)

From **Anaconda Prompt**:

```bash
cd c:\Users\danie\Desktop\Git\PRL
scripts\setup_environment.bat
```

This creates the `exmed-bert` conda environment (takes 5-10 minutes).

---

## Step 2: Run Training

From **Anaconda Prompt**:

```bash
cd c:\Users\danie\Desktop\Git\PRL
conda activate exmed-bert
scripts\train_on_synthetic_data_simple.bat
```

**That's it!** The simplified script will:
1. Generate 1000 synthetic patients
2. Convert to dataset format
3. Train ExMed-BERT

---

## Even Simpler: Manual Commands

If the batch files don't work, run commands manually:

### One-Time Setup:
```bash
cd c:\Users\danie\Desktop\Git\PRL
conda env create -f environment.yaml
```

### Every Time You Train:
```bash
conda activate exmed-bert

# Step 1: Generate data
python scripts\generate_synthetic_patients.py -n 1000

# Step 2: Convert to dataset
python scripts\convert_synthetic_to_dataset.py --input data\synthetic_patients.json --output pretrain_stuff\synthetic.pt --split all --max-length 50

# Step 3: Train
python scripts\pretrain-exmed-bert-clinvec.py --training-data pretrain_stuff\synthetic_train.pt --validation-data pretrain_stuff\synthetic_val.pt --output-dir output\synthetic_pretrain --output-data-dir output\synthetic_pretrain_data --max-seq-length 50 --train-batch-size 2 --epochs 10 --dynamic-masking
```

---

## Why This Approach?

The original automated script (`train_on_synthetic_data.bat`) tried to be too smart:
- Auto-detect conda
- Auto-create environment
- Complex error checking

This caused issues with special characters and pipes in Anaconda Prompt.

**The simple approach:**
- Assumes you're in Anaconda Prompt (you are!)
- Assumes you manually activate the environment
- Minimal error checking = fewer compatibility issues

---

## Files You Can Use

1. **`scripts\setup_environment.bat`** - Creates conda environment
2. **`scripts\train_on_synthetic_data_simple.bat`** - Runs training (no auto-setup)
3. **Manual commands** - Copy-paste from above if batch files don't work

---

## Quick Copy-Paste (Recommended)

```bash
# Open Anaconda Prompt, then paste these commands:

cd c:\Users\danie\Desktop\Git\PRL

# First time only:
conda env create -f environment.yaml

# Every time you train:
conda activate exmed-bert
python scripts\generate_synthetic_patients.py -n 1000
python scripts\convert_synthetic_to_dataset.py --input data\synthetic_patients.json --output pretrain_stuff\synthetic.pt --split all --max-length 50
python scripts\pretrain-exmed-bert-clinvec.py --training-data pretrain_stuff\synthetic_train.pt --validation-data pretrain_stuff\synthetic_val.pt --output-dir output\synthetic_pretrain --output-data-dir output\synthetic_pretrain_data --max-seq-length 50 --train-batch-size 2 --epochs 10 --dynamic-masking
```

---

## Success Looks Like

```
Generating 1000 synthetic patients...
  Generated 100/1000 patients
  Generated 200/1000 patients
  ...

Converting JSON to PatientDataset format...
  Found 50 unique ICD-10 codes
  Found 50 unique ATC5 codes
  ...

Training ExMed-BERT...
  Epoch 1/10
  [Training progress...]
```

Your trained model will be in: `output\synthetic_pretrain\`
