@echo off
REM Simplified script to train ExMed-BERT on synthetic data
REM This version has minimal error checking for maximum compatibility

setlocal EnableDelayedExpansion

echo ==========================================
echo ExMed-BERT Training on Synthetic Data
echo ==========================================

REM Configuration
set N_PATIENTS=%1
if "%N_PATIENTS%"=="" set N_PATIENTS=1000

set NOVEL_PROB=%2
if "%NOVEL_PROB%"=="" set NOVEL_PROB=0.01

set MAX_LENGTH=%3
if "%MAX_LENGTH%"=="" set MAX_LENGTH=50

set SEED=%4
if "%SEED%"=="" set SEED=42

echo.
echo Configuration:
echo   Patients: %N_PATIENTS%
echo   Novel codes: %NOVEL_PROB%
echo   Max length: %MAX_LENGTH%
echo   Seed: %SEED%
echo.

REM Activate conda environment
echo Activating exmed-bert environment...
call conda activate exmed-bert
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate conda environment
    echo.
    echo Please create the environment first:
    echo   conda env create -f environment.yaml
    echo.
    pause
    exit /b 1
)

echo Environment activated!
echo.

REM Step 1: Generate synthetic patients
echo ==========================================
echo Step 1: Generating synthetic patients
echo ==========================================
python scripts\generate_synthetic_patients.py -n %N_PATIENTS% -o data\synthetic_patients.json -p %NOVEL_PROB% -s %SEED%
if errorlevel 1 (
    echo.
    echo ERROR: Failed to generate patients
    pause
    exit /b 1
)
echo.

REM Step 2: Convert to dataset
echo ==========================================
echo Step 2: Converting to dataset format
echo ==========================================
python scripts\convert_synthetic_to_dataset.py --input data\synthetic_patients.json --output pretrain_stuff\synthetic.pt --split all --max-length %MAX_LENGTH% --train-ratio 0.8 --seed %SEED%
if errorlevel 1 (
    echo.
    echo ERROR: Failed to convert to dataset
    pause
    exit /b 1
)
echo.

REM Step 3: Train model
echo ==========================================
echo Step 3: Training ExMed-BERT
echo ==========================================
python scripts\pretrain-exmed-bert-clinvec.py --training-data pretrain_stuff\synthetic_train.pt --validation-data pretrain_stuff\synthetic_val.pt --output-dir output\synthetic_pretrain --output-data-dir output\synthetic_pretrain_data --train-batch-size 2 --eval-batch-size 2 --num-attention-heads 2 --num-hidden-layers 2 --hidden-size 64 --intermediate-size 128 --epochs 10 --learning-rate 1e-4 --max-seq-length %MAX_LENGTH% --seed %SEED% --logging-steps 5 --eval-steps 50 --save-steps 50 --dynamic-masking --no-plos
if errorlevel 1 (
    echo.
    echo ERROR: Training failed
    echo Check logs in output\synthetic_pretrain\output.log
    pause
    exit /b 1
)

echo.
echo ==========================================
echo SUCCESS! Training Complete
echo ==========================================
echo Model: output\synthetic_pretrain\
echo Logs: output\synthetic_pretrain_data\
echo ==========================================
echo.

pause
endlocal
