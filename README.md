# ExMed-BERT

Transformer-based model for medical code sequences with ClinVec pre-trained embeddings.

## Quick Start

```bash
.\train_interactive.bat
```

That's it! The interactive script will guide you through data generation, ClinVec configuration, and model training.

---

## Features

- **Transformer Architecture** - BERT-based model for medical code sequences
- **ClinVec Integration** - Pre-trained embeddings for ICD-10, ATC, and CPT codes
- **Hierarchical Initialization** - Novel code handling with parent embeddings
- **Synthetic Data Generation** - Built-in tools for creating test datasets
- **Multiple Training Modes** - Interactive and automated workflows
- **Baseline Comparisons** - Random Forest and XGBoost implementations

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Installation and setup |
| [Training Guide](docs/TRAINING_GUIDE.md) | Complete training documentation |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [ClinVec Integration](docs/ClinVec_Integration.md) | Pre-trained embeddings |
| [Synthetic Data Guide](docs/SYNTHETIC_DATA_GUIDE.md) | Data generation details |

---

## Installation

### Quick Setup

```bash
# Create conda environment
conda env create -f environment.yaml

# Activate environment
conda activate exmed-bert

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

### Requirements

- Python 3.8-3.12 (64-bit)
- Conda or Miniconda
- CUDA-enabled GPU (optional but recommended)
- 8GB+ RAM

For detailed installation instructions, see [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

---

## Training Workflows

### Option 1: Interactive Mode (Recommended)

Perfect for first-time users and experimentation:

```bash
.\train_interactive.bat
```

Features:
- Step-by-step prompts
- Auto-detection of ClinVec files
- Model size presets
- Configuration summary

### Option 2: Automated Mode

Uses configuration from `config.yaml`:

```bash
.\train.bat
```

Ideal for:
- Reproducible experiments
- Batch processing
- CI/CD pipelines

### Option 3: Manual Commands

Full control over each step:

```bash
# Generate synthetic data
python scripts/generate_synthetic_patients.py -n 1000

# Convert to dataset
python scripts/convert_synthetic_to_dataset.py \
    --input data/synthetic_patients.json \
    --output pretrain_stuff/synthetic.pt

# Train with ClinVec
python scripts/pretrain-exmed-bert-clinvec.py \
    --data-dir pretrain_stuff \
    --use-clinvec \
    --clinvec-dir ClinVec \
    --vocab-types icd10cm,atc,cpt
```

---

## ClinVec Pre-trained Embeddings

Download from Harvard Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Z6H1A8

### Required Files

Place in `ClinVec/` directory:

- `ClinGraph_nodes.csv` - Node mapping (required)
- `ClinVec_icd10cm.csv` - ICD-10 diagnosis codes
- `ClinVec_atc.csv` - ATC drug codes
- `ClinVec_cpt.csv` - CPT procedure codes

### Hierarchical Initialization

For novel codes not in ClinVec, the model uses parent code embeddings:

- **ICD-10:** E11.65 → E11.6 → E11
- **ATC5:** N02BE01 → N02BE → N02B → N02 → N

---

## Project Structure

```
PRL/
├── exmed_bert/          # Core package
│   ├── data/            # Data loading and encoding
│   ├── models/          # Model architecture
│   └── utils/           # Utilities (ClinVec, metrics)
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
│
├── docs/                # Documentation
│   ├── GETTING_STARTED.md
│   ├── TRAINING_GUIDE.md
│   └── TROUBLESHOOTING.md
│
├── ClinVec/             # Pre-trained embeddings (optional)
│
├── train.bat            # Automated training
├── train_interactive.bat # Interactive mode
├── config.yaml          # Configuration
└── environment.yaml     # Conda environment
```

---

## Model Architectures

| Size | Layers | Hidden | Params | Use Case |
|------|--------|--------|--------|----------|
| Tiny | 1 | 32 | ~10K | Testing |
| Small | 2 | 64 | ~40K | Development |
| Medium | 4 | 128 | ~200K | Production |
| Large | 6 | 288 | ~1M | Research |

---

## Examples

### Run Tests

```bash
cd tests
pytest -v
```

### Generate Custom Data

```bash
python scripts/generate_synthetic_patients.py \
    --num-patients 5000 \
    --novel-prob 0.02 \
    --seed 42
```

### Train Custom Model

```bash
python scripts/pretrain-exmed-bert-clinvec.py \
    --num-hidden-layers 4 \
    --hidden-size 256 \
    --num-attention-heads 8 \
    --num-epochs 10 \
    --batch-size 4
```

### Fine-Tune for Classification

```bash
python scripts/finetune-exmed-bert.py \
    --model-path output/checkpoint-best \
    --task-data data/classification_task.csv \
    --num-labels 2
```

---

## Pre-trained Model

Our pre-trained model is available at https://doi.org/10.5281/zenodo.7324178

---

## Baseline Models

Compare against traditional ML:

```bash
# Random Forest
python scripts/train-rf.py --data-dir pretrain_stuff

# XGBoost
python scripts/train-xgboost.py --data-dir pretrain_stuff
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{exmed-bert-2024,
  title={ExMed-BERT: A Transformer-Based Model Trained on Large Scale Claims Data},
  author={},
  journal={},
  year={2024}
}
```

---

## Support

- **Documentation:** See `docs/` folder
- **Issues:** Create a GitHub issue
- **Questions:** See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## Contact

Please post a GitHub issue or write an e-mail to manuel.lentzen@scai.fraunhofer.de if you have any questions.

---

**Ready to get started?** Run `.\train_interactive.bat` and follow the prompts!
