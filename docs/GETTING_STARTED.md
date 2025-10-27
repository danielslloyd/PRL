# Getting Started with ExMed-BERT

Welcome! This guide will help you set up and run ExMed-BERT training with synthetic patient data.

## Quick Start (One Command)

If you have Conda installed and just want to get started:

```bash
.\train_interactive.bat
```

This interactive script will guide you through:
1. Generating synthetic patient data
2. Configuring ClinVec embeddings
3. Training the model

**That's it!** The script handles everything with prompts.

---

## Prerequisites

Before you begin, ensure you have:

### Required
- **Python 3.8-3.12** (64-bit)
  - Check: `python --version`
  - Download: https://www.python.org/downloads/

- **Conda or Miniconda**
  - Check: `conda --version`
  - Download: https://docs.conda.io/en/latest/miniconda.html

### Optional
- **CUDA-enabled GPU** (for faster training)
- **ClinVec embeddings** (for better performance)
  - Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Z6H1A8
  - Place in `ClinVec/` folder

---

## Installation

### Method 1: Automated Setup (Recommended)

```bash
# Creates conda environment with all dependencies
scripts\setup_environment.bat
```

### Method 2: Manual Setup

```bash
# Create environment
conda env create -f environment.yaml

# Activate environment
conda activate exmed-bert

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}')"
```

---

## Training Workflow

### Option A: Interactive Mode (Easiest)

```bash
.\train_interactive.bat
```

The script will prompt you for:
- Number of synthetic patients to generate
- Whether to use ClinVec embeddings
- Model architecture (Tiny/Small/Medium/Large)
- Training parameters

### Option B: Automated Mode

```bash
.\train.bat
```

Uses default settings from `config.yaml`. Good for:
- Repeated training runs
- Automated workflows
- CI/CD pipelines

### Option C: Manual Step-by-Step

#### Step 1: Generate Synthetic Data

```bash
# Generate 1000 patients with 1% novel codes
python scripts/generate_synthetic_patients.py ^
    --num-patients 1000 ^
    --novel-prob 0.01 ^
    --output data/synthetic_patients.json
```

#### Step 2: Convert to Dataset

```bash
# Convert JSON to PyTorch dataset
python scripts/convert_synthetic_to_dataset.py ^
    --input data/synthetic_patients.json ^
    --output pretrain_stuff/synthetic.pt ^
    --split all ^
    --max-length 512 ^
    --train-ratio 0.8
```

#### Step 3: Train Model

```bash
# Without ClinVec
python scripts/pretrain-exmed-bert.py ^
    --data-dir pretrain_stuff ^
    --output-dir output

# With ClinVec
python scripts/pretrain-exmed-bert-clinvec.py ^
    --data-dir pretrain_stuff ^
    --output-dir output ^
    --use-clinvec ^
    --clinvec-dir ClinVec ^
    --vocab-types icd10cm,atc,cpt
```

---

## Verify Your Setup

### Quick Test

```bash
# Test data generation (10 patients)
python scripts/generate_synthetic_patients.py -n 10 -o test_data.json

# Check output
python scripts/example_load_synthetic_data.py test_data.json
```

### Run Test Suite

```bash
# Run all tests
cd tests
pytest -v

# Run specific test
pytest test_atc5_hierarchy.py -v
```

---

## Common Issues

### Issue: "conda: command not found"
**Solution:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#conda-setup-issues)

### Issue: PyTorch CUDA not available
**Solution:**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Module 'exmed_bert' not found"
**Solution:**
```bash
# Ensure you're in the project directory
cd /path/to/PRL

# Activate conda environment
conda activate exmed-bert

# Install in development mode
pip install -e .
```

### Issue: Out of memory during training
**Solution:**
- Reduce batch size in config.yaml
- Use smaller model size (Tiny or Small)
- Generate fewer patients for testing

---

## Next Steps

- **Training Guide:** [docs/TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training documentation
- **Troubleshooting:** [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Detailed problem solving
- **ClinVec Integration:** [docs/ClinVec_Integration.md](ClinVec_Integration.md) - Pre-trained embeddings
- **Synthetic Data:** [docs/SYNTHETIC_DATA_GUIDE.md](SYNTHETIC_DATA_GUIDE.md) - Technical details

---

## Project Structure

```
PRL/
├── exmed_bert/          # Core package
│   ├── data/            # Data loading and encoding
│   ├── models/          # Model architecture
│   └── utils/           # Utilities and helpers
│
├── scripts/             # Training and data generation
│   ├── generate_synthetic_patients.py
│   ├── convert_synthetic_to_dataset.py
│   ├── pretrain-exmed-bert-clinvec.py
│   └── ...
│
├── tests/               # Test suite
│   └── pytest.ini
│
├── examples/            # Example scripts
│   └── ...
│
├── ClinVec/             # Pre-trained embeddings (optional)
│   ├── ClinVec_icd10cm.csv
│   ├── ClinVec_atc.csv
│   └── ClinVec_cpt.csv
│
├── train.bat            # Main entry point (automated)
├── train_interactive.bat # Interactive mode
├── config.yaml          # Configuration
└── environment.yaml     # Conda environment
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Interactive training | `.\train_interactive.bat` |
| Automated training | `.\train.bat` |
| Generate 100 patients | `python scripts/generate_synthetic_patients.py -n 100` |
| Run tests | `cd tests && pytest` |
| Setup environment | `scripts\setup_environment.bat` |

---

## Getting Help

- **Issues?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Training questions?** See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Bug reports:** Create a GitHub issue

**Ready to train?** Run `.\train_interactive.bat` and follow the prompts!
