# GPU Cluster Setup Guide

## ✅ Auto-Detection Enabled

Your code now **automatically detects and uses GPU** when available!

### How It Works:

```python
# When device=None (default), code auto-detects:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**On GPU cluster:** Uses GPU automatically ✅
**On Mac/CPU:** Falls back to CPU ✅

---

## 🚀 Running on GPU Cluster

### 1. Check GPU is Available

```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

**Expected output on cluster:**
```
GPU: True
```

### 2. Run Your Training

```bash
cd ML
python main.py  # Your normal training script
```

**The code will print:**
```
Auto-detected device: cuda
```

---

## 🔧 Force Specific Device (Optional)

If auto-detection doesn't work, you can force GPU:

### Option A: Set Environment Variable

```bash
# In your cluster job script:
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
python main.py
```

### Option B: Modify Code Directly

```python
# In your training script:
model = NeuralNetReactorModel(
    device='cuda',  # Force GPU
    # ... other params
)
```

---

## 🎯 Verify GPU Usage During Training

Add this to your training script to monitor GPU:

```python
import torch

print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# After training starts:
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## 📊 Expected Performance

| Hardware | Training Speed | Batch Size |
|----------|---------------|------------|
| CPU (Mac) | 1x (baseline) | 32-64 |
| **GPU Cluster** | **10-100x faster** | **128-512** |

---

## ⚠️ Common Cluster Issues & Solutions

### Issue 1: "CUDA out of memory"

**Solution:** Reduce batch size

```python
model = NeuralNetReactorModel(
    batch_size=64,  # Reduce from 128
    # ... other params
)
```

### Issue 2: "No GPU detected" on cluster

**Check:**
```bash
# Verify CUDA is available
nvcc --version
nvidia-smi

# Check PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Fix:** Load CUDA module in your job script:
```bash
module load cuda/11.8
module load cudnn/8.6
```

### Issue 3: Multiple GPUs, want to use specific one

```bash
# Use only GPU 2
export CUDA_VISIBLE_DEVICES=2
python main.py
```

---

## 📝 Sample Cluster Job Script

### SLURM Example:

```bash
#!/bin/bash
#SBATCH --job-name=neural_net_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=training_%j.log

# Load modules
module load cuda/11.8
module load python/3.9

# Activate environment
source activate myenv  # or: conda activate myenv

# Verify GPU
nvidia-smi
python -c "import torch; print(f'PyTorch sees GPU: {torch.cuda.is_available()}')"

# Run training
cd /path/to/ML
python main.py
```

### PBS Example:

```bash
#!/bin/bash
#PBS -N neural_net_training
#PBS -l nodes=1:ppn=4:gpus=1    # Request 1 GPU
#PBS -l walltime=24:00:00
#PBS -l mem=32gb
#PBS -o training.log
#PBS -e training.err

# Load modules
module load cuda/11.8
module load python/3.9

# Change to working directory
cd $PBS_O_WORKDIR
cd ML

# Verify GPU
nvidia-smi
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Run training
python main.py
```

---

## 🧪 Test Script for Cluster

Save as `test_gpu.py` and run on cluster first:

```python
#!/usr/bin/env python
"""Quick GPU test before full training"""
import torch
import numpy as np
from ML_models.neural_net_train import NeuralNetReactorModel

print("="*60)
print("GPU CLUSTER TEST")
print("="*60)

# Check GPU
print(f"\n1. GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠️  WARNING: No GPU detected!")
    print("   Will use CPU (slow)")

# Quick training test
print("\n2. Testing neural network...")
X = np.random.randn(100, 10)
y = np.random.randn(100, 4)

model = NeuralNetReactorModel(
    depth=2,
    width=50,
    max_epochs=3,
    device=None,  # Auto-detect
    verbose=True
)

print("\n3. Training...")
model.fit_flux(X, y)

print("\n4. Checking device used...")
print(f"   Model trained on: {model.params['device']}")

if torch.cuda.is_available():
    print(f"   GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print("\n" + "="*60)
print("✅ TEST COMPLETE!")
print("="*60)
```

Run it:
```bash
python test_gpu.py
```

---

## 📋 Checklist Before Running on Cluster

- [ ] CUDA modules loaded
- [ ] PyTorch installed with CUDA support
- [ ] GPU detected: `torch.cuda.is_available()` returns `True`
- [ ] Test script passes
- [ ] Job script configured with GPU request
- [ ] Sufficient GPU memory for batch size

---

## 🎉 Summary

**Your code is ready for GPU cluster!**

- ✅ Auto-detects GPU when available
- ✅ Falls back to CPU on Mac
- ✅ No code changes needed
- ✅ Works with all optimization methods (Optuna, three-stage)

Just run `python main.py` on your cluster and it will use GPU automatically! 🚀
