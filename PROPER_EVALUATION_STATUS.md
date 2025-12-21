# Proper Evaluation Status - Study-Level Split

## ✅ Completed Tasks

### 1. Critical Bug Fixed - Data Split Methodology
**Problem Found**: The original evaluation used a RANDOM split instead of study-level split, causing 99% study overlap between train and test sets.

**Impact**: The reported 81.5% correlation was inflated because the model was tested on studies it had seen during training (just different participants).

**Fix Applied**:
- Updated `scripts/prepare_data.py` to implement proper study-level splitting
- Added `study_level_split` parameter (default: True)
- Implements SOCRATES paper Section 5.3 methodology: 170 train studies / 40 test studies

### 2. Data Regenerated with Proper Split
**Location**: `data/socsci210_1pct_proper_split/`

**Statistics**:
- **Train**: 170 studies, 22,944 examples
- **Val**: 20 studies, 2,780 examples
- **Test**: 20 studies, 3,289 examples
- **Verified**: Zero study overlap between train and test

**Metadata**: All study IDs tracked in `metadata.json` for verification

### 3. Wasserstein Distance Metric Implemented
**Location**: `scripts/evaluate.py` lines 134-205

**Implementation**:
- Groups predictions by (study_id, condition_num)
- Normalizes distributions to [0, 1] range
- Computes Earth Mover's Distance between predicted and actual response distributions
- Averages across all experimental conditions
- Provides per-study breakdown

**Additional Metrics**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation
- Per-study Wasserstein breakdown

### 4. All Changes Committed and Pushed
**GitHub Repository**: https://github.com/tamaratran/socrates-behavioral-prediction

**Commit**: `0ef8509` - "Add proper study-level data split and Wasserstein distance metric"

**Files Updated**:
- `scripts/prepare_data.py` - Study-level split implementation
- `scripts/evaluate.py` - Wasserstein distance metric
- `data/socsci210_1pct_proper_split/` - Properly split dataset
- `RUN_PROPER_EVALUATION_GUIDE.md` - Complete evaluation guide

## 🔄 Next Steps - Run Proper Evaluation

### Option 1: Manual Evaluation (Recommended)
Follow the detailed guide in `RUN_PROPER_EVALUATION_GUIDE.md`:

```bash
# 1. Create T4 GPU instance (from your terminal, not via this tool)
#    Thunder CLI needs interactive input, so run manually

# 2. Connect to instance
python3.11 -m thunder.thunder connect 0

# 3. On GPU, clone the repo
cd /home/ubuntu
git clone https://github.com/tamaratran/socrates-behavioral-prediction prediction-agents-copy
cd prediction-agents-copy

# 4. Install dependencies
pip install torch transformers peft accelerate datasets scikit-learn scipy matplotlib pandas numpy

# 5. Run evaluation
nohup python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct_proper_split/test.jsonl \
  --output results_proper_split \
  > evaluation_proper.log 2>&1 &

# 6. Monitor progress
tail -f evaluation_proper.log
```

**Note**: The model (`models/socrates-qwen-paper-replication/`) also needs to be on the GPU. If it's not there from previous training, you'll need to upload it or re-train.

### Option 2: Quick Test (100 examples)
To quickly verify the implementation works:

```bash
python scripts/evaluate.py \
  --model models/socrates-qwen-paper-replication \
  --test-data data/socsci210_1pct_proper_split/test.jsonl \
  --output results_test_100 \
  --max-examples 100
```

**Time**: ~20 minutes on T4 GPU
**Cost**: ~$0.22

## 📊 Expected Results

### Comparison: Data Leak vs Proper Split

| Metric | Previous (Data Leak) | Expected (Proper) | Paper (100% data) |
|--------|---------------------|-------------------|-------------------|
| **Wasserstein Distance** | N/A | **0.20-0.25** | **0.151** |
| **Correlation** | 0.815 | **0.60-0.70** | N/A |
| **MAE** | 3.72 | **5-7** | N/A |
| **Study Overlap** | 99% (208/210) | **0%** (0/210) | 0% |

**Why Lower Performance is Expected**:
- Truly unseen experimental scenarios (not just unseen participants)
- Harder generalization task
- Only 1% of training data used (paper used 100%)
- Paper reported 71% improvement on unseen conditions with full dataset

### What This Tells Us

**Our Previous Result (81.5% correlation)**:
- ❌ Not comparable to paper
- ❌ Inflated by data leakage
- ❌ Easy task: predict participants in KNOWN studies

**Proper Evaluation**:
- ✅ Matches paper methodology
- ✅ True out-of-distribution generalization
- ✅ Measures ability to predict behavior in UNSEEN experimental conditions

## 🎯 Success Criteria

The proper evaluation is successful if:
1. ✅ Wasserstein distance can be computed (no errors)
2. ✅ All 3,289 test examples evaluated
3. ✅ Results comparable to paper's methodology
4. Expected Wasserstein: 0.15-0.30 (higher than paper's 0.151 due to less training data)

## 💰 Cost Estimate

**Full Evaluation (3,289 examples)**:
- Time: ~4 hours on T4 16GB GPU
- Cost: ~$2.64 (at $0.66/hr)

**Quick Test (100 examples)**:
- Time: ~20 minutes
- Cost: ~$0.22

## 📁 Files Ready for Evaluation

All files are now in the GitHub repository:

```
✅ scripts/evaluate.py           - Updated with Wasserstein metric
✅ scripts/prepare_data.py       - Fixed study-level split
✅ data/socsci210_1pct_proper_split/
   ✅ train.jsonl                - 22,944 examples, 170 studies
   ✅ val.jsonl                  - 2,780 examples, 20 studies
   ✅ test.jsonl                 - 3,289 examples, 20 studies
   ✅ metadata.json              - Split verification
✅ RUN_PROPER_EVALUATION_GUIDE.md - Step-by-step instructions
✅ models/socrates-qwen-paper-replication/ - Trained model (131MB adapter)
```

## 🚨 Important Notes

1. **Model Location**: The trained model needs to be on the GPU. If starting fresh:
   - Option A: Upload `models/socrates-qwen-paper-replication/` (~140MB)
   - Option B: Re-train (2 hours, $2.20)

2. **GitHub Clone**: All code/data can be pulled from GitHub, making setup easier

3. **Thunder Compute Ephemeral Storage**: Data is deleted on shutdown - download results before destroying instance

4. **Evaluation Time**: 3,289 examples × ~4.5 seconds/example ≈ 4 hours

## ✨ Summary

We've successfully:
1. ✅ Identified critical data split bug (99% study overlap)
2. ✅ Fixed prepare_data.py to use proper study-level split
3. ✅ Regenerated dataset with verified 0% overlap
4. ✅ Implemented Wasserstein distance metric (paper's primary metric)
5. ✅ Committed and pushed all changes to GitHub

**Ready to run proper evaluation** that will give TRUE performance metrics comparable to the SOCRATES paper!

The previous 81.5% correlation result should be discarded as it was based on flawed methodology. The new evaluation will show the model's real ability to generalize to completely unseen experimental studies.
