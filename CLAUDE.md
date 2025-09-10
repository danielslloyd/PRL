# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the ExMed-BERT codebase, a transformer-based model trained on large-scale claims data for predicting severe COVID disease progression. The project includes pre-training, fine-tuning, and traditional ML baselines (Random Forest, XGBoost).

## Development Setup Commands

**Environment Setup:**
```bash
# Using conda
conda env create -f environment.yaml
conda activate exmed-bert

# Or using Poetry
poetry install
poetry shell
```

**Code Quality:**
```bash
# Formatting
black .

# Linting
flake8 .

# Type checking
mypy exmed_bert/
```

**Data Preparation:**
- Primary configuration is in `config.yaml`
- Use `explain.ipynb` for interactive data preparation and exploration
- Ensure `max_length` and `dynamic_masking` parameters match between data prep and training configs

**Model Training:**
```bash
# Pre-train ExMed-BERT from scratch
python scripts/pretrain-exmed-bert.py

# Fine-tune for classification
python scripts/finetune-exmed-bert.py

# Train baseline models
python scripts/train-rf.py
python scripts/train-xgboost.py

# Calculate IPTW scores for evaluation
python scripts/calculate_iptw_scores.py
```

## Architecture Overview

**Core Components:**
- `exmed_bert/models/`: Model definitions, configurations, and trainer classes
  - `config.py`: ExMedBertConfig and CombinedConfig classes for model configuration
  - `model.py`: Main transformer model implementation
  - `trainer.py`: Custom training logic
- `exmed_bert/data/`: Data processing pipeline
  - `dataset.py`: PatientDataset class for PyTorch data loading
  - `patient.py`: Patient data structure definitions
  - `encoding.py`: Vocabulary dictionaries (CodeDict, AgeDict, SexDict, StateDict)
- `exmed_bert/utils/`: Helper functions for metrics, sequence preparation, and training utilities

**Key Configuration Files:**
- `config.yaml`: Contains data preparation parameters and training hyperparameters
- `pyproject.toml`: Poetry dependencies and development tools configuration
- `environment.yaml`: Conda environment specification

**Data Flow:**
1. Raw patient data → Patient class representation
2. Encoding using vocabulary dictionaries (codes, age, sex, region)
3. PatientDataset for PyTorch data loading
4. Model training with masked language modeling and optional PLOS prediction

**Model Variants:**
- `ExMedBertConfig`: Standard transformer configuration for medical codes
- `CombinedConfig`: Extended configuration that includes quantitative clinical measures
- Classification heads: FFN, LSTM, or GRU options

**Critical Parameters:**
- `max_seq_length`/`max_length`: Must match between data preparation and training (default: 512 for production, 50 for demo)
- `dynamic_masking`: Must be consistent across data prep and training configs
- `hidden_size`, `num_hidden_layers`, `num_attention_heads`: Core transformer architecture params