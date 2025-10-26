# ✅ How to Run the Synthetic Data Generator

## ⚠️ Important: Full Pipeline Requires PyTorch

**For Step 1 (Generate Data):** Your current Python 3.14 works! ✅

**For Steps 2-3 (Convert & Train):** You need PyTorch, which requires:
- 64-bit Python (you have 32-bit)
- Python 3.8-3.12 (you have 3.14)

**👉 See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for setup instructions**

**Quick fix:** Install Miniconda and run `conda env create -f environment.yaml`

---

## You're Ready to Go! 🎉

I've installed NumPy and verified the script works. Here's how to use it:

---

## Step 1: Open a Terminal

1. Press `Windows Key + R`
2. Type `cmd` and press Enter
3. Navigate to the project folder:
   ```bash
   cd c:\Users\danie\Desktop\Git\PRL
   ```

---

## Step 2: Run the Script

### Generate 100 patients (recommended for first test)
```bash
py scripts/generate_synthetic_patients.py -n 100
```

### Generate 1000 patients (default)
```bash
py scripts/generate_synthetic_patients.py
```

### Generate with custom settings
```bash
py scripts/generate_synthetic_patients.py -n 5000 -o data/my_data.json -p 0.02
```

---

## What You'll See

The script will display progress and statistics:

```
Generating 100 synthetic patients...
Novel code probability: 1.0%
  Generated 100/100 patients

Successfully generated 100 patients
Saved to: data\synthetic_patients.json

Dataset Statistics:
  Total diagnoses: 1543
  Total prescriptions: 1876
  Avg diagnoses per patient: 15.4
  Avg prescriptions per patient: 18.8

Novel Codes:
  Unique novel ICD-10 codes generated: 12
  Total novel ICD-10 code instances: 15 (0.97%)
  Unique novel ATC5 codes generated: 14
  Total novel ATC5 code instances: 18 (0.96%)

  Example novel ICD-10 codes (hierarchical cousins):
    E11.73
    I10.12
    M54.88

  Example novel ATC5 codes (hierarchical cousins):
    C10AA88
    N02BE67
    A10BA75
```

---

## Command Options Explained

| Command | What it does |
|---------|-------------|
| `-n 100` | Generate 100 patients |
| `-o data/my_data.json` | Save to a specific file |
| `-s 123` | Set random seed (for reproducibility) |
| `-p 0.02` | 2% novel codes instead of 1% |

### Examples

```bash
# Small test dataset
py scripts/generate_synthetic_patients.py -n 50

# Medium dataset with 2% novel codes
py scripts/generate_synthetic_patients.py -n 1000 -p 0.02

# Large dataset
py scripts/generate_synthetic_patients.py -n 10000

# No novel codes (only common codes)
py scripts/generate_synthetic_patients.py -n 1000 -p 0.0

# Custom output location
py scripts/generate_synthetic_patients.py -o data/training_set.json
```

---

## Where is My Data?

By default, the data is saved to:
```
c:\Users\danie\Desktop\Git\PRL\data\synthetic_patients.json
```

You can open this file in:
- **Notepad** (or any text editor)
- **VS Code**
- **Python** (for processing)

---

## Viewing Your Generated Data

### Option 1: Quick peek in a text editor
```bash
notepad data\synthetic_patients.json
```

### Option 2: Analyze with Python script
```bash
py scripts/example_load_synthetic_data.py
```

This will show detailed statistics about your generated patients!

---

## What Makes This Data Special?

✅ **Realistic demographics**: US population-based age, sex, and state distributions
✅ **Common medical codes**: 50 most frequent ICD-10 and ATC5 codes
✅ **Novel codes with cousins**: 1% of codes are made-up but hierarchically valid
✅ **Recent dates**: All care dates are within the last 5 years
✅ **Ready to use**: Compatible with the ExMed-BERT Patient class

---

## Example: What the Data Looks Like

```json
{
  "patient_id": 0,
  "diagnoses": ["I10", "E78.5", "E11.9", "Z23"],
  "drugs": ["N02BE01", "C10AA05", "A10BA02"],
  "diagnosis_dates": ["20210315", "20220622", "20231104", "20241015"],
  "prescription_dates": ["20210320", "20220625", "20231108"],
  "birth_year": 1965,
  "sex": "FEMALE",
  "patient_state": "CA"
}
```

---

## Troubleshooting

### "No module named 'numpy'"
✅ **Already fixed!** I installed it for you.

If you need to reinstall: `py -m pip install numpy`

### "Python was not found"
Use `py` instead of `python`:
```bash
py scripts/generate_synthetic_patients.py -n 100
```

### Want to see all options?
```bash
py scripts/generate_synthetic_patients.py --help
```

---

## Next Steps

1. ✅ **Generate your first dataset**
   ```bash
   py scripts/generate_synthetic_patients.py -n 100
   ```

2. 📊 **Analyze it**
   ```bash
   py scripts/example_load_synthetic_data.py
   ```

3. 📖 **Learn more**
   - See [SYNTHETIC_DATA_README.md](SYNTHETIC_DATA_README.md) for technical details
   - See [QUICKSTART.md](QUICKSTART.md) for conda setup

---

## Already Tested! ✓

I already ran this command successfully:
```bash
py scripts/generate_synthetic_patients.py -n 10 -o data/test_patients.json -s 123
```

Results:
- ✅ Generated 10 patients
- ✅ Created 204 diagnoses and 200 prescriptions
- ✅ Generated 1 novel ICD-10 code: `I10.12`
- ✅ Generated 1 novel ATC5 code: `C10AA88`
- ✅ File saved: `data\test_patients.json` (17 KB)

**You're all set! Just copy the commands above and run them.** 🚀
