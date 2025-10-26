@echo off
REM Complete pipeline: Generate synthetic data and train ExMed-BERT
REM Auto-activates exmed-bert environment
REM Usage: train.bat [n_patients] [novel_prob] [max_length] [seed]

setlocal

echo ==========================================
echo ExMed-BERT Synthetic Data Training
echo ==========================================

REM Activate conda environment
echo Activating exmed-bert environment...
call conda activate exmed-bert 2>nul
if errorlevel 1 (
    echo WARNING: Could not activate environment
    echo Make sure the exmed-bert environment exists
    echo Run: conda env create -f environment.yaml
    echo.
)

REM Add current directory to PYTHONPATH so exmed_bert module can be found
set PYTHONPATH=%CD%;%PYTHONPATH%
echo.

REM Check and install dependencies
echo Checking dependencies...

REM Check PyTorch
python -c "import torch; print('  PyTorch:', torch.__version__, '- CUDA:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo   WARNING: PyTorch not properly installed
    echo   Installing CPU-only PyTorch...
    pip uninstall torch torchvision -y >nul 2>nul
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo   ERROR: Failed to install PyTorch
        pause
        exit /b 1
    )
    echo   PyTorch installed successfully!
)

REM Check other required packages
python -c "import typer" 2>nul
if errorlevel 1 (
    echo   Installing typer...
    pip install typer >nul 2>nul
)

python -c "import transformers" 2>nul
if errorlevel 1 (
    echo   Installing transformers...
    pip install transformers >nul 2>nul
)

python -c "import mlflow" 2>nul
if errorlevel 1 (
    echo   Installing mlflow...
    pip install mlflow >nul 2>nul
)

python -c "import joblib" 2>nul
if errorlevel 1 (
    echo   Installing joblib...
    pip install joblib >nul 2>nul
)

python -c "import pandas" 2>nul
if errorlevel 1 (
    echo   Installing pandas...
    pip install pandas >nul 2>nul
)

python -c "import yaml" 2>nul
if errorlevel 1 (
    echo   Installing pyyaml...
    pip install pyyaml >nul 2>nul
)

python -c "import multipledispatch" 2>nul
if errorlevel 1 (
    echo   Installing multipledispatch...
    pip install multipledispatch >nul 2>nul
)

python -c "import sklearn" 2>nul
if errorlevel 1 (
    echo   Installing scikit-learn...
    pip install scikit-learn >nul 2>nul
)

python -c "import psmpy" 2>nul
if errorlevel 1 (
    echo   Installing psmpy...
    pip install psmpy >nul 2>nul
)

echo Dependencies checked!
echo.

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
echo   Novel code probability: %NOVEL_PROB% (1%%)
echo   Max sequence length: %MAX_LENGTH%
echo   Random seed: %SEED%
echo.

REM Step 1: Generate synthetic patients
echo ==========================================
echo STEP 1: Generating Synthetic Patients
echo ==========================================
echo.
python scripts\generate_synthetic_patients.py -n %N_PATIENTS% -o data\synthetic_patients.json -p %NOVEL_PROB% -s %SEED%
if errorlevel 1 (
    echo.
    echo ERROR: Failed to generate synthetic patients
    echo Make sure you activated the conda environment:
    echo   conda activate exmed-bert
    pause
    exit /b 1
)

REM Step 2: Convert to PatientDataset format
echo.
echo ==========================================
echo STEP 2: Converting to Dataset Format
echo ==========================================
echo.
python scripts\convert_synthetic_to_dataset.py --input data\synthetic_patients.json --output pretrain_stuff\synthetic.pt --split all --max-length %MAX_LENGTH% --train-ratio 0.8 --seed %SEED%
if errorlevel 1 (
    echo.
    echo ERROR: Failed to convert to dataset format
    pause
    exit /b 1
)

REM Step 3: Train ExMed-BERT
echo.
echo ==========================================
echo STEP 3: Training ExMed-BERT
echo ==========================================
echo.
echo Training configuration:
echo   Training data: pretrain_stuff\synthetic_train.pt
echo   Validation data: pretrain_stuff\synthetic_val.pt
echo   Output directory: output\synthetic_pretrain
echo   Model: Small (2 layers, 64 hidden size)
echo   Epochs: 10
echo.
python scripts\pretrain-exmed-bert-clinvec.py pretrain_stuff\synthetic_train.pt pretrain_stuff\synthetic_val.pt output\synthetic_pretrain output\synthetic_pretrain_data --train-batch-size 1 --eval-batch-size 1 --num-attention-heads 2 --num-hidden-layers 2 --hidden-size 64 --intermediate-size 128 --epochs 2 --learning-rate 1e-4 --max-seq-length %MAX_LENGTH% --seed %SEED% --logging-steps 5 --eval-steps 50 --save-steps 50 --dynamic-masking
if errorlevel 1 (
    echo.
    echo ERROR: Training failed
    echo Check logs: output\synthetic_pretrain\output.log
    pause
    exit /b 1
)

echo.
echo ==========================================
echo SUCCESS! Training Complete
echo ==========================================
echo.
echo Generated files:
echo   Synthetic data: data\synthetic_patients.json
echo   Training dataset: pretrain_stuff\synthetic_train.pt
echo   Validation dataset: pretrain_stuff\synthetic_val.pt
echo   Trained model: output\synthetic_pretrain\
echo   Logs: output\synthetic_pretrain_data\
echo.
echo To view results:
echo   type output\synthetic_pretrain\output.log
echo   type output\synthetic_pretrain_data\eval_results.txt
echo.
echo ==========================================

pause
endlocal
