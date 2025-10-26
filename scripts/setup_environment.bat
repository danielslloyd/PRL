@echo off
REM One-time setup script to create the conda environment

echo ==========================================
echo ExMed-BERT Environment Setup
echo ==========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: Conda not found!
    echo.
    echo Please ensure you are running this from Anaconda Prompt
    echo.
    pause
    exit /b 1
)

echo Conda found!
echo.

REM Check for environment.yaml
if not exist environment.yaml (
    echo ERROR: environment.yaml not found!
    echo Make sure you are in the PRL directory
    echo.
    pause
    exit /b 1
)

echo Creating exmed-bert conda environment...
echo This will take 5-10 minutes (one-time setup)
echo.

conda env create -f environment.yaml

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create environment
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo SUCCESS! Environment created
echo ==========================================
echo.
echo To use it:
echo   1. conda activate exmed-bert
echo   2. scripts\train_on_synthetic_data_simple.bat
echo.

pause
