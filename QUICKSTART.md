# Quick Start Guide - Running the Synthetic Data Generator

Welcome! This guide will help you run the synthetic patient data generator script.

## Option 1: Quick Start (Easiest)

You already have Python 3.14 installed! Here's how to run the script:

### Step 1: Install required packages

Open a terminal (Command Prompt or PowerShell) in the PRL folder and run:

```bash
py -m pip install numpy
```

That's the only package needed for the synthetic data generator!

### Step 2: Run the script

```bash
# Generate 100 patients (good for testing)
py scripts/generate_synthetic_patients.py -n 100

# Generate 1000 patients (default)
py scripts/generate_synthetic_patients.py

# Generate 5000 patients with custom output location
py scripts/generate_synthetic_patients.py -n 5000 -o data/my_patients.json
```

### Step 3: Check the output

The script will create a JSON file at `data/synthetic_patients.json` (or wherever you specified with `-o`).

You'll see output like:
```
Generating 100 synthetic patients...
Novel code probability: 1.0%
  Generated 100/100 patients

Successfully generated 100 patients
Saved to: data\synthetic_patients.json

Dataset Statistics:
  Total diagnoses: 1543
  Total prescriptions: 1876
  ...
```

---

## Option 2: Using Conda (For Full ExMed-BERT Project)

If you want to work with the full ExMed-BERT codebase, you'll need to set up conda:

### Step 1: Install Anaconda or Miniconda

1. Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
2. Install it (use default settings)
3. Restart your terminal

### Step 2: Create the conda environment

```bash
cd c:\Users\danie\Desktop\Git\PRL
conda env create -f environment.yaml
```

### Step 3: Activate the environment

```bash
conda activate exmed-bert
```

### Step 4: Run the script

```bash
python scripts/generate_synthetic_patients.py -n 100
```

---

## Command-Line Options

The script supports these options:

| Option | Description | Default |
|--------|-------------|---------|
| `-n`, `--n_patients` | Number of patients to generate | 1000 |
| `-o`, `--output` | Output JSON file path | `data/synthetic_patients.json` |
| `-s`, `--seed` | Random seed for reproducibility | 42 |
| `-p`, `--novel_prob` | Probability of novel codes (0.0 to 1.0) | 0.01 (1%) |

### Examples

```bash
# Generate 500 patients with 2% novel codes
py scripts/generate_synthetic_patients.py -n 500 -p 0.02

# Generate with specific seed for reproducibility
py scripts/generate_synthetic_patients.py -n 1000 -s 123

# No novel codes, just common codes
py scripts/generate_synthetic_patients.py -n 1000 -p 0.0

# Everything custom
py scripts/generate_synthetic_patients.py -n 5000 -o data/train_patients.json -s 42 -p 0.015
```

---

## Analyzing the Generated Data

After generating data, you can analyze it:

```bash
# Make sure numpy is installed first
py -m pip install numpy

# Run the analysis script
py scripts/example_load_synthetic_data.py
```

This will show you:
- Patient demographics (age, sex, state distributions)
- Temporal patterns (dates of care)
- Novel code statistics
- Hierarchical structure analysis

---

## Troubleshooting

### "Python was not found"
- Use `py` instead of `python` or `python3`
- Or try: `py -3` to explicitly use Python 3

### "No module named 'numpy'"
```bash
py -m pip install numpy
```

### "Permission denied" or file access errors
- Make sure you're running from the PRL directory
- Check that the `data/` folder exists (the script will create it if needed)

### Script runs but no output
- Check the terminal for error messages
- Verify the output path exists: `py scripts/generate_synthetic_patients.py -o test.json`

---

## Next Steps

Once you have synthetic data:

1. **Explore the data**: Open `data/synthetic_patients.json` in a text editor
2. **Analyze it**: Run `py scripts/example_load_synthetic_data.py`
3. **Use with ExMed-BERT**: Load the data into Patient objects (see SYNTHETIC_DATA_README.md)

For more details about the novel code generation, see [SYNTHETIC_DATA_README.md](SYNTHETIC_DATA_README.md).
