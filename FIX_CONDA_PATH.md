# Fix: "Conda not found in PATH"

You've installed conda, but your terminal can't find it yet. Here are 3 solutions (try them in order):

---

## Solution 1: Use Anaconda Prompt (Easiest)

Instead of regular Command Prompt, use **Anaconda Prompt**:

1. Press `Windows Key`
2. Type: **"Anaconda Prompt"**
3. Click on it (should say "Anaconda Prompt (Miniconda3)" or similar)
4. Navigate to project:
   ```bash
   cd c:\Users\danie\Desktop\Git\PRL
   ```
5. Run the script:
   ```bash
   scripts\train_on_synthetic_data.bat
   ```

**This should work immediately!** Anaconda Prompt has conda pre-configured.

---

## Solution 2: Initialize Conda in Your Terminal

Run this **once** in regular Command Prompt:

```bash
# Find your conda installation (adjust path if needed)
C:\Users\danie\miniconda3\Scripts\conda.exe init cmd.exe

# Or if you installed for all users:
C:\ProgramData\miniconda3\Scripts\conda.exe init cmd.exe
```

Then:
1. **Close and reopen** Command Prompt
2. Try the script again:
   ```bash
   cd c:\Users\danie\Desktop\Git\PRL
   scripts\train_on_synthetic_data.bat
   ```

---

## Solution 3: Find Your Conda Installation

If you're not sure where conda is installed:

### Step 1: Find conda.exe

Open File Explorer and search for `conda.exe` in these locations:
- `C:\Users\danie\miniconda3\Scripts\conda.exe`
- `C:\Users\danie\Anaconda3\Scripts\conda.exe`
- `C:\ProgramData\miniconda3\Scripts\conda.exe`
- `C:\ProgramData\Anaconda3\Scripts\conda.exe`

### Step 2: Initialize conda

Once you find it, run (replace with your actual path):

```bash
"C:\Users\danie\miniconda3\Scripts\conda.exe" init cmd.exe
```

### Step 3: Restart terminal and try again

---

## Solution 4: Manual Environment Creation

If conda still isn't working, you can manually create the environment:

### Open Anaconda Prompt (see Solution 1)

Then run these commands:

```bash
# Navigate to project
cd c:\Users\danie\Desktop\Git\PRL

# Create environment
conda env create -f environment.yaml

# Activate it
conda activate exmed-bert

# Now run the script
scripts\train_on_synthetic_data.bat
```

The script will detect that the environment already exists and skip creation.

---

## Quick Test

To check if conda is working, run:

```bash
conda --version
```

If you see a version number (e.g., `conda 24.7.1`), conda is working!

If you see an error, conda isn't in your PATH yet - try Solution 1 or 2.

---

## Most Common Issue: Didn't Restart Terminal

After installing conda, you **must**:
1. Close ALL Command Prompt/PowerShell windows
2. Open a new one
3. Try again

Or just use **Anaconda Prompt** which always works!

---

## Still Not Working?

### Check if conda actually installed:

Look for these folders:
- `C:\Users\danie\miniconda3\`
- `C:\ProgramData\miniconda3\`
- `C:\Users\danie\Anaconda3\`

If none exist, conda didn't install properly. Reinstall:
1. Download Miniconda again
2. During installation, check **"Add conda to PATH"** (if option appears)
3. Complete installation
4. **Restart your computer**

---

## Recommended: Just Use Anaconda Prompt

**Easiest solution:**
1. Press Windows Key
2. Search: **"Anaconda Prompt"**
3. Use that terminal instead of Command Prompt

This terminal has conda pre-configured and will work immediately!

---

## After Getting Conda Working

Once `conda --version` works, just run:

```bash
cd c:\Users\danie\Desktop\Git\PRL
scripts\train_on_synthetic_data.bat
```

The script will automatically create the environment and run the training pipeline!
