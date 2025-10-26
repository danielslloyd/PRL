@echo off
REM Script to train ExMed-BERT on synthetic data
REM This is a complete end-to-end pipeline for Windows
REM Includes automatic conda environment setup

setlocal

echo ==========================================
echo ExMed-BERT Training on Synthetic Data
echo ==========================================

REM Configuration (use defaults if not provided)
set N_PATIENTS=%1
if "%N_PATIENTS%"=="" set N_PATIENTS=1000

set NOVEL_PROB=%2
if "%NOVEL_PROB%"=="" set NOVEL_PROB=0.01

set MAX_LENGTH=%3
if "%MAX_LENGTH%"=="" set MAX_LENGTH=50

set SEED=%4
if "%SEED%"=="" set SEED=42

REM Paths
set SYNTHETIC_JSON=data\synthetic_patients.json
set DATASET_DIR=pretrain_stuff
set TRAIN_DATASET=%DATASET_DIR%\synthetic_train.pt
set VAL_DATASET=%DATASET_DIR%\synthetic_val.pt
set OUTPUT_DIR=output\synthetic_pretrain
set OUTPUT_DATA_DIR=output\synthetic_pretrain_data

REM Step 0: Check and setup conda environment
echo:
echo Step 0: Checking conda environment...
echo:

REM Check if conda is available
where conda >nul 2>nul
if errorlevel 1 (
    echo WARNING: Conda not found in PATH
    echo:
    echo If you just installed conda:
    echo   1. CLOSE this window
    echo   2. Open "Anaconda Prompt" from Start menu
    echo   3. Run this script again from Anaconda Prompt
    echo:
    echo Or initialize conda in your current terminal:
    echo   Run: conda init cmd.exe
    echo   Then restart this terminal
    echo:
    echo If conda is not installed:
    echo   Download from: https://docs.conda.io/en/latest/miniconda.html
    echo   Get: Miniconda3 Windows 64-bit
    echo:
    echo See FIX_CONDA_PATH.md for detailed help
    echo:
    pause
    exit /b 1
)

REM Check if exmed-bert environment exists
conda env list > temp_envs.txt 2>nul
findstr /C:"exmed-bert" temp_envs.txt >nul 2>nul
set ENV_EXISTS=%ERRORLEVEL%
del temp_envs.txt >nul 2>nul

if %ENV_EXISTS% NEQ 0 (
    echo Conda environment 'exmed-bert' not found. Creating it now...
    echo This will take 5-10 minutes ^(one-time setup^).
    echo:

    if not exist environment.yaml (
        echo ERROR: environment.yaml not found!
        echo Make sure you're running this from the PRL directory.
        exit /b 1
    )

    conda env create -f environment.yaml
    if errorlevel 1 (
        echo ERROR: Failed to create conda environment
        exit /b 1
    )
    echo:
    echo Conda environment created successfully!
) else (
    echo Conda environment 'exmed-bert' found.
)

REM Activate conda environment
echo Activating conda environment...
call conda activate exmed-bert
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    echo Try running manually: conda activate exmed-bert
    exit /b 1
)
echo Environment activated successfully!
echo:

REM Step 1: Generate synthetic patients
echo:
echo Step 1: Generating %N_PATIENTS% synthetic patients...
echo   Novel code probability: %NOVEL_PROB%
echo   Output: %SYNTHETIC_JSON%
echo:
python scripts\generate_synthetic_patients.py -n %N_PATIENTS% -o %SYNTHETIC_JSON% -p %NOVEL_PROB% -s %SEED%
if errorlevel 1 (
    echo ERROR: Failed to generate synthetic patients
    echo:
    echo Troubleshooting:
    echo   1. Make sure conda environment is activated
    echo   2. Check that numpy is installed: python -c "import numpy"
    echo:
    exit /b 1
)

REM Step 2: Convert to PatientDataset format
echo:
echo Step 2: Converting JSON to PatientDataset format...
echo   Max sequence length: %MAX_LENGTH%
echo   Train/val split: 80/20
echo:
python scripts\convert_synthetic_to_dataset.py --input %SYNTHETIC_JSON% --output %DATASET_DIR%\synthetic.pt --split all --max-length %MAX_LENGTH% --train-ratio 0.8 --seed %SEED%
if errorlevel 1 (
    echo ERROR: Failed to convert data to PatientDataset format
    echo:
    echo Troubleshooting:
    echo   1. Check that PyTorch is installed: python -c "import torch"
    echo   2. Make sure %SYNTHETIC_JSON% exists
    echo:
    exit /b 1
)

REM Step 3: Run pretraining
echo:
echo Step 3: Starting ExMed-BERT pretraining...
echo   Training data: %TRAIN_DATASET%
echo   Validation data: %VAL_DATASET%
echo   Output: %OUTPUT_DIR%
echo:
python scripts\pretrain-exmed-bert-clinvec.py --training-data %TRAIN_DATASET% --validation-data %VAL_DATASET% --output-dir %OUTPUT_DIR% --output-data-dir %OUTPUT_DATA_DIR% --train-batch-size 2 --eval-batch-size 2 --num-attention-heads 2 --num-hidden-layers 2 --hidden-size 64 --intermediate-size 128 --epochs 10 --learning-rate 1e-4 --max-seq-length %MAX_LENGTH% --seed %SEED% --logging-steps 5 --eval-steps 50 --save-steps 50 --dynamic-masking --no-plos
if errorlevel 1 (
    echo ERROR: Failed to run pretraining
    echo:
    echo Check the logs in %OUTPUT_DIR%\output.log for details
    echo:
    exit /b 1
)

echo:
echo ==========================================
echo Training Complete!
echo   Model saved to: %OUTPUT_DIR%
echo   Logs saved to: %OUTPUT_DATA_DIR%
echo ==========================================

endlocal
