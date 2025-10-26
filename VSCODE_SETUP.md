# Setting Up Conda in VS Code Terminal

Yes! You can create and use the conda environment directly in VS Code's integrated terminal.

---

## Option 1: Initialize Conda in VS Code Terminal (Recommended)

### Step 1: Open VS Code Terminal
- Press `` Ctrl + ` `` (backtick) or go to **View > Terminal**

### Step 2: Initialize Conda

Run this command (adjust path if needed):

```bash
# If you installed Miniconda in your user directory:
C:\Users\danie\miniconda3\Scripts\conda.exe init cmd.exe

# Or if installed for all users:
C:\ProgramData\miniconda3\Scripts\conda.exe init cmd.exe

# Or if you have Anaconda:
C:\Users\danie\Anaconda3\Scripts\conda.exe init cmd.exe
```

### Step 3: Restart VS Code Terminal

Close and reopen the terminal (or click the trash icon and open a new one).

### Step 4: Verify Conda Works

```bash
conda --version
```

You should see something like `conda 24.7.1`

### Step 5: Create Environment

```bash
cd c:\Users\danie\Desktop\Git\PRL
conda env create -f environment.yaml
```

### Step 6: Activate and Use

```bash
conda activate exmed-bert
train.bat
```

---

## Option 2: Change VS Code Default Terminal to Command Prompt

If VS Code is using PowerShell by default:

### Step 1: Change Default Shell

1. Press `Ctrl + Shift + P` to open Command Palette
2. Type: **"Terminal: Select Default Profile"**
3. Select **"Command Prompt"**

### Step 2: Open New Terminal

- Close current terminal
- Open new terminal (`` Ctrl + ` ``)
- Should now be Command Prompt

### Step 3: Initialize Conda (if needed)

```bash
C:\Users\danie\miniconda3\Scripts\conda.exe init cmd.exe
```

### Step 4: Restart Terminal and Use

```bash
conda env create -f environment.yaml
conda activate exmed-bert
```

---

## Option 3: Use PowerShell (Alternative)

If you prefer PowerShell:

### Initialize Conda for PowerShell

```powershell
C:\Users\danie\miniconda3\Scripts\conda.exe init powershell
```

### Restart Terminal

Close and reopen VS Code terminal.

### Create Environment

```powershell
conda env create -f environment.yaml
conda activate exmed-bert
```

---

## Option 4: Quick Setup (Copy-Paste)

If you're not sure which conda you have, try each of these:

```bash
# Try option 1
C:\Users\danie\miniconda3\Scripts\conda.exe init cmd.exe

# If that doesn't work, try option 2
C:\ProgramData\miniconda3\Scripts\conda.exe init cmd.exe

# If that doesn't work, try option 3
C:\Users\danie\Anaconda3\Scripts\conda.exe init cmd.exe
```

One of these should work. Then:
1. Restart VS Code terminal
2. Run: `conda env create -f environment.yaml`

---

## Verify Everything Works

After setup, test in VS Code terminal:

```bash
# Should show version
conda --version

# Should list environments (base should be there)
conda env list

# Create exmed-bert environment
conda env create -f environment.yaml

# Activate it
conda activate exmed-bert

# Should show (exmed-bert) in your prompt
# Now run training
train.bat
```

---

## Set VS Code to Auto-Activate Environment

Once the environment is created:

### Step 1: Create VS Code Settings

Create `.vscode/settings.json` in your project:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/../../../miniconda3/envs/exmed-bert/python.exe",
    "python.terminal.activateEnvironment": true,
    "terminal.integrated.env.windows": {
        "CONDA_DEFAULT_ENV": "exmed-bert"
    }
}
```

### Step 2: Restart VS Code

The terminal should now auto-activate `exmed-bert` when you open it!

---

## Troubleshooting

### "conda is not recognized"

You need to initialize conda first. Run one of these:

```bash
# Find your conda installation
dir C:\Users\danie\miniconda3\Scripts\conda.exe
dir C:\ProgramData\miniconda3\Scripts\conda.exe

# Then initialize with the path you found
C:\<PATH_TO_CONDA>\Scripts\conda.exe init cmd.exe
```

### "Cannot be loaded because running scripts is disabled"

If using PowerShell, you need to change execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then initialize conda for PowerShell:
```powershell
C:\Users\danie\miniconda3\Scripts\conda.exe init powershell
```

### VS Code terminal still doesn't work

1. Close VS Code completely
2. Open **Anaconda Prompt** separately
3. Run: `conda init cmd.exe`
4. Reopen VS Code
5. Open new terminal

---

## Recommended Workflow

**After one-time setup:**

1. Open VS Code in your project folder
2. Open terminal (`` Ctrl + ` ``)
3. Terminal should show `(base)` or auto-activate `(exmed-bert)`
4. If not, run: `conda activate exmed-bert`
5. Run: `train.bat`

That's it! Everything runs in VS Code's integrated terminal. ✨
