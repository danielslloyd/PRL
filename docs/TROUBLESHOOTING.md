# Troubleshooting Guide

Solutions to common issues when setting up and using ExMed-BERT.

---

## Table of Contents

1. [Conda Setup Issues](#conda-setup-issues)
2. [VS Code Configuration](#vs-code-configuration)
3. [Python Installation](#python-installation)
4. [PyTorch and CUDA](#pytorch-and-cuda)
5. [ClinVec Issues](#clinvec-issues)
6. [Training Problems](#training-problems)
7. [Data Generation Errors](#data-generation-errors)
8. [Module Import Errors](#module-import-errors)

---

## Conda Setup Issues

### Problem: "conda: command not found"

**Symptoms:**
```
'conda' is not recognized as an internal or external command
```

**Solutions:**

#### Option 1: Fix PATH in Current Session
```bash
# Find conda installation
where conda.exe

# Typical locations:
# C:\Users\<username>\Anaconda3\Scripts\conda.exe
# C:\ProgramData\Anaconda3\Scripts\conda.exe
# C:\Users\<username>\Miniconda3\Scripts\conda.exe

# Add to PATH temporarily
set PATH=C:\Users\<username>\Anaconda3\Scripts;%PATH%
```

#### Option 2: Use Full Path
```bash
# Instead of: conda activate exmed-bert
# Use:
C:\Users\<username>\Anaconda3\Scripts\conda.exe activate exmed-bert
```

#### Option 3: Fix PATH Permanently
1. Open "Edit the system environment variables"
2. Click "Environment Variables"
3. Under "User variables", find "Path"
4. Click "Edit"
5. Click "New" and add:
   - `C:\Users\<username>\Anaconda3`
   - `C:\Users\<username>\Anaconda3\Scripts`
   - `C:\Users\<username>\Anaconda3\Library\bin`
6. Click "OK" and restart terminal

#### Option 4: Initialize Conda for Bash/PowerShell
```bash
# For Command Prompt
conda init cmd.exe

# For PowerShell
conda init powershell

# For Git Bash
conda init bash
```

**Verify fix:**
```bash
conda --version
# Should output: conda 23.x.x
```

---

### Problem: Conda environment activation fails

**Symptoms:**
```
EnvironmentNotWritableError
CommandNotFoundError: Your shell has not been properly configured
```

**Solutions:**

#### Initialize Conda
```bash
conda init bash
conda init cmd.exe
```

#### Restart terminal
Close and reopen your terminal after running `conda init`.

#### Check environment exists
```bash
conda env list

# If exmed-bert not listed:
conda env create -f environment.yaml
```

---

## VS Code Configuration

### Problem: VS Code terminal doesn't recognize conda

**Symptoms:**
- Conda works in Command Prompt but not in VS Code terminal
- `conda activate` fails in VS Code

**Solutions:**

#### Option 1: Change Default Terminal
1. Open VS Code
2. Press `Ctrl+Shift+P`
3. Type "Terminal: Select Default Profile"
4. Choose "Command Prompt" (not PowerShell or Git Bash)

#### Option 2: Configure PowerShell for Conda
Add to your PowerShell profile (`$PROFILE`):

```powershell
# Initialize Conda
& 'C:\Users\<username>\Anaconda3\shell\condabin\conda-hook.ps1'
```

#### Option 3: Use VS Code Settings
Add to `.vscode/settings.json`:

```json
{
  "terminal.integrated.env.windows": {
    "PATH": "C:\\Users\\<username>\\Anaconda3\\Scripts;${env:PATH}"
  },
  "python.terminal.activateEnvironment": true,
  "python.condaPath": "C:\\Users\\<username>\\Anaconda3\\Scripts\\conda.exe"
}
```

#### Option 4: Manual Activation
Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Activate Conda",
      "type": "shell",
      "command": "conda activate exmed-bert",
      "problemMatcher": []
    }
  ]
}
```

---

### Problem: VS Code Python interpreter not found

**Symptoms:**
- Import errors in VS Code
- "Python was not found" message

**Solutions:**

#### Select Conda Interpreter
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose conda environment: `Python 3.x.x ('exmed-bert')`

#### Manually specify path
Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "C:\\Users\\<username>\\Anaconda3\\envs\\exmed-bert\\python.exe"
}
```

---

## Python Installation

### Problem: Wrong Python version

**Symptoms:**
```
Python 3.13 or 3.14 installed, but conda creates 3.11 environment
```

**Solution:**
This is correct! The `environment.yaml` specifies Python 3.11 because it's most compatible with all dependencies. The conda environment will use its own Python version, separate from your system Python.

**Verify:**
```bash
conda activate exmed-bert
python --version
# Should show: Python 3.11.x
```

---

### Problem: 32-bit Python instead of 64-bit

**Symptoms:**
```
OSError: [WinError 193] %1 is not a valid Win32 application
```

**Solution:**
1. Uninstall 32-bit Python
2. Download 64-bit Python from python.org
3. Reinstall conda environment:
```bash
conda env remove -n exmed-bert
conda env create -f environment.yaml
```

**Verify:**
```python
import struct
print(struct.calcsize("P") * 8)
# Should output: 64
```

---

## PyTorch and CUDA

### Problem: PyTorch not using GPU

**Symptoms:**
```python
import torch
print(torch.cuda.is_available())  # False
```

**Solutions:**

#### Check NVIDIA GPU exists
```bash
nvidia-smi
# Should show GPU details
```

#### Reinstall PyTorch with CUDA
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Verify CUDA version matches
```bash
# Check installed CUDA version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

#### Test GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
```

---

### Problem: Out of memory during training

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

#### Reduce batch size
```bash
# In config.yaml
batch_size: 1  # Reduce from 2 or 4

# Or command line
python scripts/pretrain-exmed-bert-clinvec.py --batch-size 1
```

#### Use smaller model
```bash
# Tiny model (minimal memory)
python scripts/pretrain-exmed-bert-clinvec.py \
    --num-hidden-layers 1 \
    --hidden-size 32
```

#### Reduce sequence length
```bash
# Convert dataset with shorter sequences
python scripts/convert_synthetic_to_dataset.py \
    --max-length 256  # Instead of 512
```

#### Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

#### Use CPU instead
```bash
# Force CPU training
export CUDA_VISIBLE_DEVICES=""
python scripts/pretrain-exmed-bert-clinvec.py
```

---

## ClinVec Issues

### Problem: ClinVec files not found

**Symptoms:**
```
FileNotFoundError: ClinGraph_nodes.csv not found
```

**Solutions:**

#### Check directory structure
```bash
ClinVec/
├── ClinGraph_nodes.csv      # Required
├── ClinVec_icd10cm.csv      # ICD-10 embeddings
├── ClinVec_atc.csv          # ATC embeddings
└── ClinVec_cpt.csv          # CPT embeddings
```

#### Verify file paths
```bash
# Check if files exist
dir ClinVec\*.csv

# Or in Python:
import os
print(os.listdir('ClinVec'))
```

#### Update config if needed
```yaml
# config.yaml
clinvec_params:
  clinvec_dir: "C:/full/path/to/ClinVec"  # Use full path if needed
```

---

### Problem: ClinVec embeddings dimension mismatch

**Symptoms:**
```
RuntimeError: size mismatch, got 200, expected 64
```

**Solutions:**

#### Use auto resize
```bash
python scripts/pretrain-exmed-bert-clinvec.py \
    --resize-method auto \
    --resize-if-needed
```

#### Match embedding dimensions
```bash
# ClinVec embeddings are 200-dimensional
# Either resize ClinVec to model size, or model to ClinVec size

# Option 1: Resize ClinVec down
--resize-method pca \
--hidden-size 64

# Option 2: Use ClinVec dimensions
--hidden-size 200
```

#### Available resize methods
- `auto` - Automatically choose best method
- `truncate` - Cut off extra dimensions (fast, loses info)
- `pca` - PCA dimensionality reduction (preserves info)
- `learned_projection` - Learn projection (best quality)
- `pad_smart` - Pad with learned values
- `pad_random` - Pad with random values

---

### Problem: Novel codes not getting hierarchical initialization

**Symptoms:**
Novel codes get random initialization instead of parent embeddings.

**Solutions:**

#### Enable hierarchical initialization
```bash
python scripts/pretrain-exmed-bert-clinvec.py \
    --use-hierarchical-init
```

#### Check code format
```python
# ICD-10: Must match pattern [A-Z]\d{2}
# ATC5: Must match pattern [A-Z]\d{2}[A-Z]{2}\d{2}
# CPT: No hierarchical support (5-digit numbers)

# Valid:
"E11.65"  # ICD-10
"N02BE01"  # ATC5

# Invalid for hierarchy:
"250.01"  # Old ICD-9 (no longer supported)
"99213"   # CPT (no hierarchy)
```

#### Verify parent codes exist in ClinVec
```python
# Check if parent codes are in ClinVec
import pandas as pd

# Load ICD-10 embeddings
icd10 = pd.read_csv('ClinVec/ClinVec_icd10cm.csv')
print(icd10['code'].head())

# Check for specific code
print('E11' in icd10['code'].values)  # Should be True
```

---

## Training Problems

### Problem: Training loss not decreasing

**Symptoms:**
Loss stays constant or increases over epochs.

**Solutions:**

#### Lower learning rate
```bash
--learning-rate 1e-5  # Instead of 1e-4
```

#### Check data quality
```bash
python scripts/example_load_synthetic_data.py data/synthetic_patients.json
```

#### Increase training time
```bash
--num-epochs 10  # Instead of 2
```

#### Verify data loading
```python
from exmed_bert.data import PatientDataset
import torch

# Load dataset
dataset = torch.load('pretrain_stuff/synthetic_train.pt')
print(f"Dataset size: {len(dataset)}")
print(f"First sample: {dataset[0]}")
```

---

### Problem: Training extremely slow

**Symptoms:**
Each epoch takes hours.

**Solutions:**

#### Check GPU usage
```bash
nvidia-smi
# Should show GPU utilization near 100%
```

#### Increase batch size
```bash
--batch-size 4  # If GPU memory allows
```

#### Reduce dataset size for testing
```bash
# Generate fewer patients
python scripts/generate_synthetic_patients.py -n 100
```

#### Enable mixed precision training
```bash
--fp16  # Half precision for faster training
```

---

## Data Generation Errors

### Problem: JSON decode error

**Symptoms:**
```
JSONDecodeError: Expecting value: line 1 column 1
```

**Solutions:**

#### Check file exists and is valid JSON
```bash
# Verify file
python -m json.tool data/synthetic_patients.json

# Or regenerate
python scripts/generate_synthetic_patients.py -n 100
```

---

### Problem: No codes generated

**Symptoms:**
Generated patients have no diagnosis or prescription codes.

**Solution:**
This shouldn't happen with current scripts. Check:
```bash
python scripts/example_load_synthetic_data.py data/synthetic_patients.json
```

If codes are missing, regenerate data.

---

## Module Import Errors

### Problem: "ModuleNotFoundError: No module named 'exmed_bert'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'exmed_bert'
```

**Solutions:**

#### Activate conda environment
```bash
conda activate exmed-bert
```

#### Install in development mode
```bash
pip install -e .
```

#### Check PYTHONPATH
```bash
# Add project directory to PYTHONPATH
export PYTHONPATH="/path/to/PRL:$PYTHONPATH"

# Or in Windows:
set PYTHONPATH=C:\Users\<username>\Desktop\Git\PRL;%PYTHONPATH%
```

#### Run from project root
```bash
cd /c/Users/danie/Desktop/Git/PRL
python scripts/pretrain-exmed-bert-clinvec.py
```

---

### Problem: "No module named 'torch'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solutions:**

#### Install PyTorch
```bash
# CPU only
pip install torch torchvision torchaudio

# With CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Verify installation
```bash
python -c "import torch; print(torch.__version__)"
```

---

## Still Having Issues?

### Check System Requirements
- Python 3.8-3.12 (64-bit)
- 8GB+ RAM recommended
- NVIDIA GPU with CUDA support (optional but recommended)
- 10GB+ free disk space

### Verify Installation
```bash
# Run diagnostic
python -c "
import sys
import torch
import pandas
import numpy
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Pandas: {pandas.__version__}')
print(f'NumPy: {numpy.__version__}')
"
```

### Get Help
1. Check [GETTING_STARTED.md](GETTING_STARTED.md) for setup
2. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training
3. Create a GitHub issue with:
   - Error message
   - Steps to reproduce
   - System information
   - Output of diagnostic script above

---

## Common Error Messages Quick Reference

| Error | Quick Fix |
|-------|-----------|
| "conda: command not found" | Add conda to PATH or use full path |
| "CUDA out of memory" | Reduce batch size to 1 |
| "ModuleNotFoundError" | Activate conda environment |
| "FileNotFoundError" | Check paths, run from project root |
| "JSONDecodeError" | Regenerate synthetic data |
| "Size mismatch" | Use `--resize-method auto` |
| "Loss not decreasing" | Lower learning rate to 1e-5 |
| "Training too slow" | Check GPU usage, increase batch size |

---

**Still stuck?** See [GETTING_STARTED.md](GETTING_STARTED.md) or create a GitHub issue.
