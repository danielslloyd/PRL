#!/bin/bash
# Script to train ExMed-BERT on synthetic data
# This is a complete end-to-end pipeline

set -e  # Exit on error

echo "=========================================="
echo "ExMed-BERT Training on Synthetic Data"
echo "=========================================="

# Configuration
N_PATIENTS=${1:-1000}
NOVEL_PROB=${2:-0.01}
MAX_LENGTH=${3:-50}
SEED=${4:-42}

# Paths
SYNTHETIC_JSON="data/synthetic_patients.json"
DATASET_DIR="pretrain_stuff"
TRAIN_DATASET="${DATASET_DIR}/synthetic_train.pt"
VAL_DATASET="${DATASET_DIR}/synthetic_val.pt"
OUTPUT_DIR="output/synthetic_pretrain"
OUTPUT_DATA_DIR="output/synthetic_pretrain_data"

# Step 1: Generate synthetic patients
echo ""
echo "Step 1: Generating ${N_PATIENTS} synthetic patients..."
echo "  Novel code probability: ${NOVEL_PROB}"
echo "  Output: ${SYNTHETIC_JSON}"
echo ""
py scripts/generate_synthetic_patients.py \
    -n ${N_PATIENTS} \
    -o ${SYNTHETIC_JSON} \
    -p ${NOVEL_PROB} \
    -s ${SEED}

# Step 2: Convert to PatientDataset format
echo ""
echo "Step 2: Converting JSON to PatientDataset format..."
echo "  Max sequence length: ${MAX_LENGTH}"
echo "  Train/val split: 80/20"
echo ""
python scripts/convert_synthetic_to_dataset.py \
    --input ${SYNTHETIC_JSON} \
    --output ${DATASET_DIR}/synthetic.pt \
    --split all \
    --max-length ${MAX_LENGTH} \
    --train-ratio 0.8 \
    --seed ${SEED}

# Step 3: Run pretraining
echo ""
echo "Step 3: Starting ExMed-BERT pretraining..."
echo "  Training data: ${TRAIN_DATASET}"
echo "  Validation data: ${VAL_DATASET}"
echo "  Output: ${OUTPUT_DIR}"
echo ""
python scripts/pretrain-exmed-bert-clinvec.py \
    --training-data ${TRAIN_DATASET} \
    --validation-data ${VAL_DATASET} \
    --output-dir ${OUTPUT_DIR} \
    --output-data-dir ${OUTPUT_DATA_DIR} \
    --train-batch-size 2 \
    --eval-batch-size 2 \
    --num-attention-heads 2 \
    --num-hidden-layers 2 \
    --hidden-size 64 \
    --intermediate-size 128 \
    --epochs 10 \
    --learning-rate 1e-4 \
    --max-seq-length ${MAX_LENGTH} \
    --seed ${SEED} \
    --logging-steps 5 \
    --eval-steps 50 \
    --save-steps 50 \
    --dynamic-masking \
    --no-plos

echo ""
echo "=========================================="
echo "Training Complete!"
echo "  Model saved to: ${OUTPUT_DIR}"
echo "  Logs saved to: ${OUTPUT_DATA_DIR}"
echo "=========================================="
