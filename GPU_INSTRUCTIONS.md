# GPU Training Instructions

## ✅ Local Setup Complete!
Your data is ready: **29,013 examples** formatted and split into train/val/test sets.

---

## 🖥️ GPU Setup (You do this part)

### Step 1: Sign Up for Thunder Compute (2 minutes)

1. Go to: https://console.thundercompute.com/signup
2. Create account
3. Add payment method: https://console.thundercompute.com/settings/billing

### Step 2: Install Thunder Compute Extension (1 minute)

**In VS Code:**
1. Open VS Code
2. Go to Extensions (Cmd+Shift+X)
3. Search for "Thunder Compute"
4. Click Install

### Step 3: Connect to Thunder Compute (1 minute)

1. Press `Cmd+Shift+P` (or Ctrl+Shift+P on Windows)
2. Type: "Thunder Compute: Login"
3. Follow authentication prompts

### Step 4: Create GPU Instance (1 minute)

1. In Thunder Compute extension panel, click "Create Instance"
2. **Select GPU:** A100 40GB ($0.66/hour)
3. Click "Create"
4. Click the connect button (two arrows icon) when ready
5. A new VS Code window will open connected to the GPU

### Step 5: Upload Your Code (2 minutes)

**In the new VS Code window (connected to GPU):**

1. Open terminal in VS Code (Cmd+` or Ctrl+`)
2. Run:
```bash
mkdir -p ~/socrates-training
cd ~/socrates-training
```

3. **Upload your local folder:**
   - Drag and drop the entire `prediction-agents-copy` folder into the VS Code file explorer
   - OR use scp/rsync if you prefer command line

### Step 6: Run Training on GPU (~60 minutes, ~$0.66 total)

**Copy and paste these commands into the GPU terminal:**

```bash
# Navigate to uploaded folder
cd ~/socrates-training/prediction-agents-copy

# Install dependencies
pip install torch transformers datasets accelerate peft trl huggingface-hub python-dotenv tqdm pandas "numpy<2" bitsandbytes scipy scikit-learn matplotlib seaborn -q

# Verify installation
python -c "import torch; print(f'✓ PyTorch installed. CUDA available: {torch.cuda.is_available()}')"

# Run finetuning
python scripts/finetune.py \
  --model "Qwen/Qwen2.5-14B-Instruct" \
  --data data/socsci210_1pct \
  --output models/socrates-qwen-1pct \
  --use-qlora \
  --epochs 3 \
  --batch-size 4

# This will take ~30-60 minutes
# Cost: ~$0.33-0.66 at $0.66/hour
```

### Step 7: Evaluate Model (~10 minutes, ~$0.11)

```bash
python scripts/evaluate.py --model models/socrates-qwen-1pct
```

### Step 8: Test Your Model

```bash
# Run a predefined example
python scripts/test_inference.py --model models/socrates-qwen-1pct --example 1

# Or interactive mode
python scripts/test_inference.py --model models/socrates-qwen-1pct
```

### Step 9: Download Results

**Download your trained model back to your Mac:**

1. In VS Code file explorer (GPU window), right-click on `models/socrates-qwen-1pct`
2. Click "Download"
3. Save to your local `prediction-agents-copy/models/` folder

**Or use command line:**
```bash
# On your Mac
scp -r [thunder-compute-instance]:~/socrates-training/prediction-agents-copy/models/socrates-qwen-1pct ./models/
```

### Step 10: ⚠️ STOP THE INSTANCE! ⚠️

**IMPORTANT:** Don't forget to stop your GPU instance when done to stop billing!

1. In Thunder Compute extension, find your instance
2. Click "Stop" or "Terminate"
3. Verify it's stopped in the console: https://console.thundercompute.com

---

## 📊 Expected Results

After training completes, you should see:

```
Training complete!
Model saved to: models/socrates-qwen-1pct

Next steps:
  1. Evaluate model: python scripts/evaluate.py --model models/socrates-qwen-1pct
  2. Test inference: python scripts/test_inference.py --model models/socrates-qwen-1pct
```

Evaluation will show metrics like:
- Mean Absolute Error (MAE)
- Correlation with ground truth
- Sample predictions vs actual responses

---

## 💰 Cost Breakdown

| Task | Time | Cost |
|------|------|------|
| Setup & upload | 10 min | $0.11 |
| Training | 30-60 min | $0.33-0.66 |
| Evaluation | 10 min | $0.11 |
| Testing | 5 min | $0.05 |
| **TOTAL** | **~1-1.5 hours** | **~$0.60-1.00** |

At $0.66/hour, this is incredibly cheap for finetuning a 14B parameter model!

---

## 🚨 Troubleshooting

### "Out of memory" error
```bash
# Reduce batch size
python scripts/finetune.py \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --data data/socsci210_1pct \
  --output models/socrates-qwen-1pct \
  --use-qlora
```

### "CUDA not available"
```bash
# Check GPU
nvidia-smi

# If no GPU, verify you selected A100 when creating instance
```

### Training is slow
```bash
# This is normal! A100 will process ~400-800 examples/minute
# 23,210 training examples × 3 epochs = ~70K examples
# At 600 examples/min = ~120 minutes (2 hours)
# But with optimizations, should be 30-60 minutes
```

---

## 📝 Alternative: RunPod (If Thunder Compute is full)

If Thunder Compute doesn't have A100s available:

1. Go to https://runpod.io
2. Create account
3. Select "GPU Pods"
4. Choose A100 40GB (~$1.39/hour)
5. Select "PyTorch" template
6. Upload code via web interface or SSH
7. Run same commands as above

Cost: ~$1.40-2.80 for full training (still very cheap!)

---

## 🎯 Next Steps After Training

1. **Compare with paper results:** SOCRATES-Qwen-14B showed 26% improvement over base model
2. **Scale up if promising:** Try 10% subset (290K examples, ~$5-10)
3. **Test on your own scenarios:** Use `test_inference.py` with custom demographics and questions
4. **Try different models:** Experiment with Llama 3 8B instead of Qwen 14B

---

Need help? Check the main README.md or QUICKSTART.md for more details!
