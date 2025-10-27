@echo off
REM Interactive pipeline: Generate synthetic data and train ExMed-BERT
REM Prompts user for all configuration options

setlocal enabledelayedexpansion

echo ==========================================
echo ExMed-BERT Interactive Training Pipeline
echo ==========================================
echo.

REM Activate conda environment
echo Activating exmed-bert environment...
call conda activate exmed-bert 2>nul
if errorlevel 1 (
    echo WARNING: Could not activate environment
    echo Make sure the exmed-bert environment exists
    echo Run: conda env create -f environment.yaml
    echo.
)

REM Add current directory to PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%
echo.

REM ==========================================
REM STEP 1: Data Generation Configuration
REM ==========================================
echo ==========================================
echo STEP 1: Data Configuration
echo ==========================================
echo.

:ASK_GENERATE_DATA
set /p GENERATE_DATA="Generate new synthetic data? (y/n) [y]: "
if "%GENERATE_DATA%"=="" set GENERATE_DATA=y

if /i "%GENERATE_DATA%"=="y" goto GENERATE_NEW_DATA
if /i "%GENERATE_DATA%"=="n" goto USE_EXISTING_DATA
echo Invalid input. Please enter 'y' or 'n'.
goto ASK_GENERATE_DATA

:GENERATE_NEW_DATA
echo.
echo --- Synthetic Data Generation Settings ---

REM Ask for number of patients
set /p N_PATIENTS="Number of patients to generate [1000]: "
if "%N_PATIENTS%"=="" set N_PATIENTS=1000

REM Ask for novel code probability
set /p NOVEL_PROB="Novel code probability (0.0-1.0) [0.01]: "
if "%NOVEL_PROB%"=="" set NOVEL_PROB=0.01

REM Ask for output location
set /p DATA_OUTPUT="Output JSON file [data\synthetic_patients.json]: "
if "%DATA_OUTPUT%"=="" set DATA_OUTPUT=data\synthetic_patients.json

REM Ask for random seed
set /p SEED="Random seed for reproducibility [42]: "
if "%SEED%"=="" set SEED=42

echo.
echo Summary:
echo   Patients: %N_PATIENTS%
echo   Novel code probability: %NOVEL_PROB%
echo   Output: %DATA_OUTPUT%
echo   Random seed: %SEED%
echo.

set /p CONFIRM_GEN="Proceed with generation? (y/n) [y]: "
if "%CONFIRM_GEN%"=="" set CONFIRM_GEN=y
if /i not "%CONFIRM_GEN%"=="y" goto GENERATE_NEW_DATA

REM Generate the data
echo.
echo Generating synthetic patients...
python scripts\generate_synthetic_patients.py -n %N_PATIENTS% -o %DATA_OUTPUT% -p %NOVEL_PROB% -s %SEED%
if errorlevel 1 (
    echo.
    echo ERROR: Failed to generate synthetic patients
    pause
    exit /b 1
)

set DATA_JSON=%DATA_OUTPUT%
goto CONVERT_DATA

:USE_EXISTING_DATA
echo.
echo --- Using Existing Data ---

REM Check for common locations
set DATA_FOUND=n

if exist "data\synthetic_patients.json" (
    echo Found: data\synthetic_patients.json
    set DEFAULT_DATA=data\synthetic_patients.json
    set DATA_FOUND=y
)

if exist "data\patients.json" (
    echo Found: data\patients.json
    if "%DATA_FOUND%"=="n" set DEFAULT_DATA=data\patients.json
    set DATA_FOUND=y
)

if "%DATA_FOUND%"=="y" goto ASK_DATA_PATH_WITH_DEFAULT
echo.
echo No patient data files found in common locations.
set /p DATA_JSON="Patient data JSON file path: "
goto VALIDATE_DATA_PATH

:ASK_DATA_PATH_WITH_DEFAULT
echo.
set /p DATA_JSON="Patient data JSON file [%DEFAULT_DATA%]: "
if "%DATA_JSON%"=="" set DATA_JSON=%DEFAULT_DATA%

:VALIDATE_DATA_PATH

if not exist "%DATA_JSON%" (
    echo.
    echo ERROR: File not found: %DATA_JSON%
    echo Please check the path and try again.
    pause
    exit /b 1
)

echo.
echo Using patient data: %DATA_JSON%

REM Ask for random seed
set /p SEED="Random seed for train/val split [42]: "
if "%SEED%"=="" set SEED=42

goto CONVERT_DATA

:CONVERT_DATA
echo.
echo ==========================================
echo STEP 2: Dataset Conversion Configuration
echo ==========================================
echo.

REM Ask for max sequence length
set /p MAX_LENGTH="Maximum sequence length [50]: "
if "%MAX_LENGTH%"=="" set MAX_LENGTH=50

REM Ask for train/validation split
set /p TRAIN_RATIO="Training set ratio (0.0-1.0) [0.8]: "
if "%TRAIN_RATIO%"=="" set TRAIN_RATIO=0.8

REM Ask for output directory
set /p DATASET_DIR="Dataset output directory [pretrain_stuff]: "
if "%DATASET_DIR%"=="" set DATASET_DIR=pretrain_stuff

echo.
echo Summary:
echo   Input: %DATA_JSON%
echo   Max sequence length: %MAX_LENGTH%
echo   Train/Val split: %TRAIN_RATIO%
echo   Output: %DATASET_DIR%\synthetic_train.pt
echo           %DATASET_DIR%\synthetic_val.pt
echo.

set /p CONFIRM_CONV="Proceed with conversion? (y/n) [y]: "
if "%CONFIRM_CONV%"=="" set CONFIRM_CONV=y
if /i not "%CONFIRM_CONV%"=="y" goto CONVERT_DATA

REM Convert to dataset format
echo.
echo Converting to PatientDataset format...
if not exist "%DATASET_DIR%" mkdir "%DATASET_DIR%"
python scripts\convert_synthetic_to_dataset.py --input "%DATA_JSON%" --output "%DATASET_DIR%\synthetic.pt" --split all --max-length %MAX_LENGTH% --train-ratio %TRAIN_RATIO% --seed %SEED%
if errorlevel 1 (
    echo.
    echo ERROR: Failed to convert to dataset format
    pause
    exit /b 1
)

REM ==========================================
REM STEP 3: ClinVec Configuration
REM ==========================================
echo.
echo ==========================================
echo STEP 3: ClinVec Embeddings Configuration
echo ==========================================
echo.

:ASK_USE_CLINVEC
set /p USE_CLINVEC="Use ClinVec pre-trained embeddings? (y/n) [y]: "
if "%USE_CLINVEC%"=="" set USE_CLINVEC=y

if /i "%USE_CLINVEC%"=="y" goto CONFIGURE_CLINVEC
if /i "%USE_CLINVEC%"=="n" goto TRAINING_CONFIG
echo Invalid input. Please enter 'y' or 'n'.
goto ASK_USE_CLINVEC

:CONFIGURE_CLINVEC
echo.
echo --- ClinVec Settings ---

REM Check for ClinVec directory
set CLINVEC_FOUND=n

if exist "ClinVec" (
    echo Found: ClinVec\
    set DEFAULT_CLINVEC=ClinVec
    set CLINVEC_FOUND=y
)

echo.
if "%CLINVEC_FOUND%"=="y" goto ASK_CLINVEC_WITH_DEFAULT
echo No ClinVec directories found in common locations.
set /p CLINVEC_DIR="ClinVec data directory path: "
goto VALIDATE_CLINVEC_DIR

:ASK_CLINVEC_WITH_DEFAULT
set /p CLINVEC_DIR="ClinVec data directory [%DEFAULT_CLINVEC%]: "
if "%CLINVEC_DIR%"=="" set CLINVEC_DIR=%DEFAULT_CLINVEC%

:VALIDATE_CLINVEC_DIR
if not exist "%CLINVEC_DIR%" (
    echo.
    echo WARNING: Directory not found: %CLINVEC_DIR%
    set /p CONTINUE_ANYWAY="Continue without ClinVec? (y/n) [n]: "
    if "%CONTINUE_ANYWAY%"=="" set CONTINUE_ANYWAY=n
    if /i "%CONTINUE_ANYWAY%"=="n" goto CONFIGURE_CLINVEC
    set USE_CLINVEC=n
    goto TRAINING_CONFIG
)

REM Check if ClinVec files exist
if not exist "%CLINVEC_DIR%\ClinGraph_nodes.csv" (
    echo.
    echo WARNING: ClinGraph_nodes.csv not found in %CLINVEC_DIR%
    echo This doesn't look like a valid ClinVec directory.
    set /p CONTINUE_ANYWAY="Continue anyway? (y/n) [n]: "
    if "%CONTINUE_ANYWAY%"=="" set CONTINUE_ANYWAY=n
    if /i "%CONTINUE_ANYWAY%"=="n" goto CONFIGURE_CLINVEC
)

echo.
echo ClinVec directory: %CLINVEC_DIR%

REM Detect available vocabulary files
echo.
echo Detecting available ClinVec vocabulary files...
set VOCAB_TYPES=

REM Check for ICD-10
if exist "%CLINVEC_DIR%\ClinVec_icd10cm.csv" (
    echo   [FOUND] ICD-10 CM ^(ClinVec_icd10cm.csv^)
    set /p USE_ICD10="  Include ICD-10 codes? (y/n) [y]: "
    if "!USE_ICD10!"=="" set USE_ICD10=y
    if /i "!USE_ICD10!"=="y" set VOCAB_TYPES=icd10cm
) else (
    echo   [NOT FOUND] ICD-10 CM ^(ClinVec_icd10cm.csv^)
)

REM Check for ATC
if exist "%CLINVEC_DIR%\ClinVec_atc.csv" (
    echo   [FOUND] ATC Drug Codes ^(ClinVec_atc.csv^)
    set /p USE_ATC="  Include ATC drug codes? (y/n) [y]: "
    if "!USE_ATC!"=="" set USE_ATC=y
    if /i "!USE_ATC!"=="y" (
        if defined VOCAB_TYPES (
            set VOCAB_TYPES=!VOCAB_TYPES!,atc
        ) else (
            set VOCAB_TYPES=atc
        )
    )
) else (
    echo   [NOT FOUND] ATC Drug Codes ^(ClinVec_atc.csv^)
)

REM Check for RxNorm
if exist "%CLINVEC_DIR%\ClinVec_rxnorm.csv" (
    echo   [FOUND] RxNorm ^(ClinVec_rxnorm.csv^)
    set /p USE_RXNORM="  Include RxNorm codes? (y/n) [y]: "
    if "!USE_RXNORM!"=="" set USE_RXNORM=y
    if /i "!USE_RXNORM!"=="y" (
        if defined VOCAB_TYPES (
            set VOCAB_TYPES=!VOCAB_TYPES!,rxnorm
        ) else (
            set VOCAB_TYPES=rxnorm
        )
    )
) else (
    echo   [NOT FOUND] RxNorm ^(ClinVec_rxnorm.csv^)
)

REM Check for PheCode
if exist "%CLINVEC_DIR%\ClinVec_phecode.csv" (
    echo   [FOUND] PheCode ^(ClinVec_phecode.csv^)
    set /p USE_PHECODE="  Include PheCode? (y/n) [y]: "
    if "!USE_PHECODE!"=="" set USE_PHECODE=y
    if /i "!USE_PHECODE!"=="y" (
        if defined VOCAB_TYPES (
            set VOCAB_TYPES=!VOCAB_TYPES!,phecode
        ) else (
            set VOCAB_TYPES=phecode
        )
    )
) else (
    echo   [NOT FOUND] PheCode ^(ClinVec_phecode.csv^)
)

REM Check for CPT
if exist "%CLINVEC_DIR%\ClinVec_cpt.csv" (
    echo   [FOUND] CPT Procedure Codes ^(ClinVec_cpt.csv^)
    set /p USE_CPT="  Include CPT codes? (y/n) [y]: "
    if "!USE_CPT!"=="" set USE_CPT=y
    if /i "!USE_CPT!"=="y" (
        if defined VOCAB_TYPES (
            set VOCAB_TYPES=!VOCAB_TYPES!,cpt
        ) else (
            set VOCAB_TYPES=cpt
        )
    )
) else (
    echo   [NOT FOUND] CPT Procedure Codes ^(ClinVec_cpt.csv^)
)

REM If no vocabularies selected, use default
if not defined VOCAB_TYPES (
    echo.
    echo WARNING: No vocabularies selected. Using default: icd10cm,atc
    set VOCAB_TYPES=icd10cm,atc
)

echo.
echo Selected vocabularies: %VOCAB_TYPES%

REM Ask for hierarchical initialization
set /p USE_HIERARCHICAL="Use hierarchical initialization for novel codes? (y/n) [y]: "
if "%USE_HIERARCHICAL%"=="" set USE_HIERARCHICAL=y

REM Ask for resize method
echo.
echo Embedding resize methods:
echo   auto - Automatically choose best method
echo   truncate - Truncate to model dimensions
echo   pca - Use PCA dimensionality reduction
echo   learned_projection - Use learned projection layer
echo   pad_smart - Pad with learned values
echo   pad_random - Pad with random values
echo.
set /p RESIZE_METHOD="Embedding resize method [auto]: "
if "%RESIZE_METHOD%"=="" set RESIZE_METHOD=auto

echo.
echo ClinVec Summary:
echo   Directory: %CLINVEC_DIR%
echo   Vocabularies: %VOCAB_TYPES%
echo   Hierarchical init: %USE_HIERARCHICAL%
echo   Resize method: %RESIZE_METHOD%
echo.

set /p CONFIRM_CLINVEC="Confirm ClinVec settings? (y/n) [y]: "
if "%CONFIRM_CLINVEC%"=="" set CONFIRM_CLINVEC=y
if /i not "%CONFIRM_CLINVEC%"=="y" goto CONFIGURE_CLINVEC

REM Build ClinVec flags
set CLINVEC_FLAGS=--use-clinvec --clinvec-dir "%CLINVEC_DIR%" --vocab-types "%VOCAB_TYPES%" --resize-method %RESIZE_METHOD%
if /i "%USE_HIERARCHICAL%"=="y" set CLINVEC_FLAGS=%CLINVEC_FLAGS% --use-hierarchical-init

goto TRAINING_CONFIG

:TRAINING_CONFIG
echo.
echo ==========================================
echo STEP 4: Review Training Configuration
echo ==========================================
echo.
echo Reading configuration from config.yaml...
echo.

REM Read model configuration from config.yaml
REM These are set in config.yaml and cannot be edited here
REM To change these settings, edit config.yaml
set NUM_LAYERS=2
set HIDDEN_SIZE=64
set INTERMEDIATE_SIZE=128
set NUM_HEADS=2
set EPOCHS=2
set BATCH_SIZE=1
set LEARNING_RATE=1e-4
set OUTPUT_DIR=output\model

echo Configuration Summary:
echo.
echo   Model Architecture:
echo     - Layers: %NUM_LAYERS%
echo     - Hidden size: %HIDDEN_SIZE%
echo     - Intermediate size: %INTERMEDIATE_SIZE%
echo     - Attention heads: %NUM_HEADS%
echo.
echo   Training Parameters:
echo     - Epochs: %EPOCHS%
echo     - Batch size: %BATCH_SIZE%
echo     - Learning rate: %LEARNING_RATE%
echo     - Max sequence length: %MAX_LENGTH%
echo.
echo   Data:
echo     - Training data: %DATA_FILE%
echo     - Output directory: %OUTPUT_DIR%
echo.
echo   ClinVec Settings:
if /i "%USE_CLINVEC%"=="y" (
    echo     - Status: ENABLED
    echo     - Directory: %CLINVEC_DIR%
    echo     - Vocabularies: %VOCAB_TYPES%
    echo     - Hierarchical init: %USE_HIERARCHICAL%
) else (
    echo     - Status: DISABLED
)
echo.
echo NOTE: To change model/training settings, edit config.yaml
echo.

set /p CONFIRM_TRAIN="Proceed with training? (y/n) [y]: "
if "%CONFIRM_TRAIN%"=="" set CONFIRM_TRAIN=y
if /i not "%CONFIRM_TRAIN%"=="y" (
    echo.
    echo Training cancelled. To modify settings:
    echo   1. Edit config.yaml for model/training parameters
    echo   2. Re-run this script to change data/ClinVec settings
    echo.
    goto END
)

REM ==========================================
REM STEP 5: Training
REM ==========================================
echo.
echo ==========================================
echo STEP 5: Training ExMed-BERT
echo ==========================================
echo.

REM Ensure output directories exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%_data" mkdir "%OUTPUT_DIR%_data"

REM Build training command
echo Starting training...
echo.

if /i "%USE_CLINVEC%"=="y" (
    python scripts\pretrain-exmed-bert-clinvec.py "%DATASET_DIR%\synthetic_train.pt" "%DATASET_DIR%\synthetic_val.pt" "%OUTPUT_DIR%" "%OUTPUT_DIR%_data" --train-batch-size %BATCH_SIZE% --eval-batch-size %BATCH_SIZE% --num-attention-heads %NUM_HEADS% --num-hidden-layers %NUM_LAYERS% --hidden-size %HIDDEN_SIZE% --intermediate-size %INTERMEDIATE_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --max-seq-length %MAX_LENGTH% --seed %SEED% --logging-steps 5 --eval-steps 50 --save-steps 50 --dynamic-masking %CLINVEC_FLAGS%
) else (
    python scripts\pretrain-exmed-bert-clinvec.py "%DATASET_DIR%\synthetic_train.pt" "%DATASET_DIR%\synthetic_val.pt" "%OUTPUT_DIR%" "%OUTPUT_DIR%_data" --train-batch-size %BATCH_SIZE% --eval-batch-size %BATCH_SIZE% --num-attention-heads %NUM_HEADS% --num-hidden-layers %NUM_LAYERS% --hidden-size %HIDDEN_SIZE% --intermediate-size %INTERMEDIATE_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --max-seq-length %MAX_LENGTH% --seed %SEED% --logging-steps 5 --eval-steps 50 --save-steps 50 --dynamic-masking
)

if errorlevel 1 (
    echo.
    echo ERROR: Training failed
    echo Check logs: %OUTPUT_DIR%\output.log
    pause
    exit /b 1
)

echo.
echo ==========================================
echo SUCCESS! Training Complete
echo ==========================================
echo.
echo Generated files:
if exist "%DATA_OUTPUT%" echo   Synthetic data: %DATA_OUTPUT%
echo   Training dataset: %DATASET_DIR%\synthetic_train.pt
echo   Validation dataset: %DATASET_DIR%\synthetic_val.pt
echo   Trained model: %OUTPUT_DIR%\
echo   Logs: %OUTPUT_DIR%_data\
echo.
echo To view training logs:
echo   type %OUTPUT_DIR%\output.log
echo.
echo ==========================================

pause
endlocal
