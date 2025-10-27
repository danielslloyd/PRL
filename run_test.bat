@echo off
REM Run ATC5 hierarchy test using conda environment

echo Running ATC5 Hierarchy Tests...
echo:

REM Try to activate conda environment
call conda activate exmed-bert 2>nul

REM Set PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Run the test
python test_atc5_hierarchy.py

if errorlevel 1 (
    echo:
    echo ERROR: Tests failed
    exit /b 1
)

echo:
echo Tests completed successfully!
exit /b 0
