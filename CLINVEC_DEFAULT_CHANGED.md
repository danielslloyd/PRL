# ClinVec Default Changed to 'Yes'

## Overview

Changed the default response for using ClinVec pre-trained embeddings from `[n]` to `[y]` in the interactive training script.

## Rationale

Since the user has ClinVec data available and ClinVec embeddings provide better performance than random initialization, it makes sense to default to using them. This:
- Reduces friction for users who want the best performance
- Makes ClinVec the recommended path by default
- Still allows users to opt-out by typing 'n'

## Changes Made

### [train_interactive.bat](train_interactive.bat:192-193)

**Before:**
```batch
set /p USE_CLINVEC="Use ClinVec pre-trained embeddings? (y/n) [n]: "
if "%USE_CLINVEC%"=="" set USE_CLINVEC=n
```

**After:**
```batch
set /p USE_CLINVEC="Use ClinVec pre-trained embeddings? (y/n) [y]: "
if "%USE_CLINVEC%"=="" set USE_CLINVEC=y
```

### [INTERACTIVE_TRAINING_GUIDE.md](INTERACTIVE_TRAINING_GUIDE.md:71-73)

**Updated section header:**
```markdown
**Question: Use ClinVec pre-trained embeddings? (Default: y)**
- `y` - Use ClinVec embeddings (better performance, requires ClinVec data) **[DEFAULT]**
- `n` - Use random initialization (simpler, no external data needed)
```

**Updated example interaction:**
```
Use ClinVec pre-trained embeddings? (y/n) [y]:
```

## User Experience

### Before
```
Use ClinVec pre-trained embeddings? (y/n) [n]:
```
- User had to type 'y' to use ClinVec
- Defaulted to random initialization (lower performance)

### After
```
Use ClinVec pre-trained embeddings? (y/n) [y]:
```
- User can just press Enter to use ClinVec (recommended)
- Still can type 'n' to use random initialization
- Defaults to better performance option

## Benefits

1. **Better Defaults**: Users get better model performance by default
2. **Less Typing**: Most users will want ClinVec, so they can just press Enter
3. **Clear Recommendation**: Makes it clear that ClinVec is the recommended approach
4. **Easy Opt-Out**: Users can still easily choose random initialization by typing 'n'

## Impact

This change affects the default workflow:
- **Old workflow**: User must type 'y' to use ClinVec
- **New workflow**: User presses Enter to use ClinVec (recommended path)

If a user doesn't have ClinVec data, they can:
1. Type 'n' to skip ClinVec
2. Or see validation warnings if ClinVec directory/files are missing and choose to continue anyway

## Date
2025-10-27

## Related Changes
- [CPT_SUPPORT_ADDED.md](CPT_SUPPORT_ADDED.md) - Added CPT code support
- [VOCAB_AUTO_DETECTION.md](VOCAB_AUTO_DETECTION.md) - Auto-detection feature
- [ICD9_REMOVAL_SUMMARY.md](ICD9_REMOVAL_SUMMARY.md) - ICD-9 removal and standardization
