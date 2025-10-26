# Installation Guide - PyTorch Dependencies

## Issue: PyTorch Not Available for Python 3.14 (32-bit)

You currently have **Python 3.14 (32-bit)** installed, but PyTorch requires:
- ✅ **64-bit Python**
- ✅ **Python 3.8 - 3.12** (PyTorch doesn't support 3.14 yet)

## Solution: Install Compatible Python Version

You have **two options**:

---

## Option 1: Use Conda (Recommended for ExMed-BERT)

This is the recommended approach because the project uses Conda.

### Step 1: Install Miniconda (if not already installed)

1. Download **Miniconda** (Windows 64-bit):
   - Go to: https://docs.conda.io/en/latest/miniconda.html
   - Download: **Miniconda3 Windows 64-bit**
   - Run the installer (use default settings)

### Step 2: Create ExMed-BERT Environment

Open **Anaconda Prompt** (or regular Command Prompt after restarting) and run:

```bash
# Navigate to project directory
cd c:\Users\danie\Desktop\Git\PRL

# Create environment from the project's environment.yaml
conda env create -f environment.yaml

# This will install:
# - Python 3.10 (64-bit)
# - PyTorch
# - Transformers
# - All ExMed-BERT dependencies
```

### Step 3: Activate and Use

```bash
# Activate the environment
conda activate exmed-bert

# Now run the training pipeline
scripts\train_on_synthetic_data.bat
```

**Every time you work on this project:**
```bash
conda activate exmed-bert
# Then run your scripts
```

---

## Option 2: Install Python 3.11 (64-bit) Standalone

If you prefer not to use Conda:

### Step 1: Install Python 3.11 (64-bit)

1. Go to: https://www.python.org/downloads/
2. Download **Python 3.11** (make sure it says "Windows installer (64-bit)")
3. Run installer with these options:
   - ✅ **Check "Add Python to PATH"**
   - ✅ Install for all users
   - ✅ **Select 64-bit installation**

### Step 2: Verify Installation

```bash
# Check Python version (should say 64 bit)
py -3.11 --version

# Or if 3.11 is your default:
python --version
```

### Step 3: Install Dependencies

```bash
# Navigate to project
cd c:\Users\danie\Desktop\Git\PRL

# Install PyTorch (CPU version)
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
py -3.11 -m pip install transformers typer mlflow joblib pandas scikit-learn pyyaml
```

### Step 4: Run Scripts

```bash
# Use py -3.11 instead of py
py -3.11 scripts\generate_synthetic_patients.py -n 1000
py -3.11 scripts\convert_synthetic_to_dataset.py --input data\synthetic_patients.json --output pretrain_stuff\synthetic.pt --split all --max-length 50
```

---

## Option 3: Quick Test (Generate Data Only)

If you just want to generate synthetic data (Step 1), you can do that with your current Python:

```bash
# This works with Python 3.14 (only needs numpy, which you have)
py scripts\generate_synthetic_patients.py -n 1000

# This creates: data/synthetic_patients.json
```

Then you can:
- Share the JSON file with someone who has PyTorch
- Install proper Python later for Steps 2 and 3
- Use the JSON for other purposes

---

## Checking Your Installation

### Check if you have 64-bit Python:

```bash
py -c "import struct; print('64-bit' if struct.calcsize('P') * 8 == 64 else '32-bit')"
```

If this says **32-bit**, you need to install 64-bit Python.

### Check Python version:

```bash
py --version
```

PyTorch requires Python **3.8, 3.9, 3.10, 3.11, or 3.12** (3.14 is too new).

### After installing Conda or Python 3.11, verify PyTorch:

```bash
# With Conda
conda activate exmed-bert
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

# Or with Python 3.11
py -3.11 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

---

## Recommended Path (What I Suggest)

**For ExMed-BERT development, use Conda:**

1. ✅ Install Miniconda (5 minutes)
2. ✅ Run `conda env create -f environment.yaml` (10-15 minutes)
3. ✅ Activate with `conda activate exmed-bert`
4. ✅ Run the pipeline

**Why Conda?**
- Project already has `environment.yaml` configured
- Manages all dependencies automatically
- Isolated environment (won't affect other Python projects)
- Matches the development environment used by the original authors

---

## Quick Setup Commands (Conda Path)

```bash
# 1. Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

# 2. Open Anaconda Prompt or Command Prompt (restart after Conda install)

# 3. Create environment
cd c:\Users\danie\Desktop\Git\PRL
conda env create -f environment.yaml

# 4. Activate
conda activate exmed-bert

# 5. Run pipeline
scripts\train_on_synthetic_data.bat

# That's it! 🎉
```

---

## Need Help?

**Error: "conda: command not found"**
- Restart your terminal after installing Miniconda
- Or use Anaconda Prompt (search for it in Start menu)

**Error: "environment.yaml not found"**
- Make sure you're in the PRL directory: `cd c:\Users\danie\Desktop\Git\PRL`

**Still having issues?**
- Check the environment.yaml file exists: `dir environment.yaml`
- Try creating an environment manually:
  ```bash
  conda create -n exmed-bert python=3.10
  conda activate exmed-bert
  conda install pytorch torchvision cpuonly -c pytorch
  pip install transformers typer mlflow
  ```

---

## Next Steps After Installation

Once PyTorch is installed:

```bash
# Activate environment (if using Conda)
conda activate exmed-bert

# Run the complete training pipeline
scripts\train_on_synthetic_data.bat

# Or run steps manually
py scripts\generate_synthetic_patients.py -n 1000
py scripts\convert_synthetic_to_dataset.py --input data\synthetic_patients.json --output pretrain_stuff\synthetic.pt --split all --max-length 50
py scripts\pretrain-exmed-bert-clinvec.py --training-data pretrain_stuff\synthetic_train.pt --validation-data pretrain_stuff\synthetic_val.pt --output-dir output\synthetic_pretrain --output-data-dir output\synthetic_pretrain_data --max-seq-length 50 --train-batch-size 2 --epochs 10 --dynamic-masking
```
