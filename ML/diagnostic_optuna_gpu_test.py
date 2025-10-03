#!/usr/bin/env python
"""
Comprehensive Diagnostic Test for Optuna + GPU + Parallelization
This test mimics your actual training setup to diagnose any issues
"""

import sys
import os
import time
import torch
import numpy as np
from datetime import datetime
import multiprocessing

print("="*80)
print("OPTUNA + GPU + PARALLEL DIAGNOSTIC TEST")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# TEST 1: System & Environment Information
# ============================================================================
print("\n[TEST 1] System & Environment Information")
print("-"*80)
import platform
print(f"Hostname: {platform.node()}")
print(f"Python: {sys.version.split()[0]}")
print(f"Working Directory: {os.getcwd()}")
print(f"CPU Cores Available: {multiprocessing.cpu_count()}")

# Check SLURM environment
slurm_vars = ['SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_GPUS',
              'SLURM_NODELIST', 'SLURM_JOB_PARTITION']
print("\nSLURM Environment:")
for var in slurm_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"  {var}: {value}")

# ============================================================================
# TEST 2: GPU Detection & Specs
# ============================================================================
print("\n[TEST 2] GPU Detection & Specifications")
print("-"*80)
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"✅ GPU DETECTED!")
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Total: {props.total_memory / 1e9:.2f} GB")
        print(f"  Memory Free: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("⚠️  NO GPU DETECTED - will use CPU (slow)")

# ============================================================================
# TEST 3: Multiprocessing Setup
# ============================================================================
print("\n[TEST 3] Multiprocessing Configuration")
print("-"*80)

# Apply the fix
if torch.cuda.is_available():
    try:
        import torch.multiprocessing
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("✅ PyTorch multiprocessing set to 'spawn' for CUDA compatibility")
    except RuntimeError as e:
        print(f"⚠️  Multiprocessing already set: {e}")
else:
    print("⚠️  No GPU, skipping multiprocessing setup")

# ============================================================================
# TEST 4: Import Neural Network Classes
# ============================================================================
print("\n[TEST 4] Import Neural Network & Optuna")
print("-"*80)
try:
    from ML_models.neural_net_train import PyTorchRegressorWrapper, NeuralNetReactorModel
    print("✅ Neural network classes imported")

    import optuna
    print(f"✅ Optuna imported (version {optuna.__version__})")
except Exception as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)

# ============================================================================
# TEST 5: Single Neural Network Training (Baseline)
# ============================================================================
print("\n[TEST 5] Single Neural Network Training (Baseline)")
print("-"*80)

# Create dummy data
n_samples = 500
n_features = 15
n_outputs = 4

X_dummy = np.random.randn(n_samples, n_features).astype(np.float32)
y_dummy = np.random.randn(n_samples, n_outputs).astype(np.float32)

print(f"Training data: {n_samples} samples, {n_features} features, {n_outputs} outputs")

# Train a single model
single_start = time.time()
model = PyTorchRegressorWrapper(
    depth=2,
    width=100,
    max_epochs=10,
    device=None,  # Auto-detect
    verbose=False
)

print("Training single model...")
model.fit(X_dummy, y_dummy)
single_time = time.time() - single_start

print(f"✅ Single model trained in {single_time:.2f} seconds")
print(f"   Device used: {model.device}")

if torch.cuda.is_available():
    print(f"   GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
    torch.cuda.empty_cache()

# ============================================================================
# TEST 6: Optuna Parallel Trials (THE CRITICAL TEST)
# ============================================================================
print("\n[TEST 6] Optuna Parallel Optimization")
print("-"*80)

# Get number of cores to use
n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
n_trials = 20  # Small number for quick test
n_jobs = min(n_cores, 8)  # Use up to 8 parallel jobs

print(f"Configuration:")
print(f"  Total CPU cores available: {n_cores}")
print(f"  Parallel jobs (n_jobs): {n_jobs}")
print(f"  Total trials: {n_trials}")
print(f"  Expected behavior: {n_jobs} trials running simultaneously")

# Track trial times
trial_times = []
trial_devices = []
process_ids = []

def objective(trial):
    """Optuna objective function"""
    import os
    import torch
    import time

    trial_start = time.time()
    process_id = os.getpid()

    # Suggest hyperparameters
    depth = trial.suggest_int('depth', 2, 3)
    width = trial.suggest_int('width', 50, 150)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)

    # Create model
    from ML_models.neural_net_train import PyTorchRegressorWrapper
    model = PyTorchRegressorWrapper(
        depth=depth,
        width=width,
        learning_rate=learning_rate,
        max_epochs=5,
        device=None,  # Auto-detect
        verbose=False
    )

    # Create small dataset
    n = 200
    X = np.random.randn(n, 15).astype(np.float32)
    y = np.random.randn(n, 4).astype(np.float32)

    # Train
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)
    mse = np.mean((y - predictions) ** 2)

    trial_time = time.time() - trial_start

    # Store diagnostics
    trial_times.append(trial_time)
    trial_devices.append(model.device)
    process_ids.append(process_id)

    # Print progress
    print(f"  Trial {trial.number}: PID={process_id}, Device={model.device}, "
          f"Time={trial_time:.1f}s, MSE={mse:.4f}")

    return mse

# Run Optuna optimization
print(f"\nStarting Optuna optimization with n_jobs={n_jobs}...")
print(f"If parallelization works, you should see {n_jobs} trials running at once.")
print("-"*80)

optuna_start = time.time()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

optuna_time = time.time() - optuna_start

print("-"*80)
print(f"✅ Optuna optimization complete!")
print(f"   Total time: {optuna_time:.2f} seconds")
print(f"   Average time per trial: {optuna_time/n_trials:.2f} seconds")
print(f"   Best MSE: {study.best_value:.6f}")

# ============================================================================
# TEST 7: Analyze Results
# ============================================================================
print("\n[TEST 7] Parallelization Analysis")
print("-"*80)

unique_pids = len(set(process_ids))
print(f"Unique Process IDs: {unique_pids}")
if unique_pids > 1:
    print(f"✅ PARALLELIZATION WORKING! ({unique_pids} separate processes detected)")
else:
    print(f"⚠️  WARNING: Only 1 process detected. Parallelization may not be working!")

unique_devices = set(trial_devices)
print(f"\nDevices used: {unique_devices}")
if 'cuda' in unique_devices:
    print(f"✅ GPU WAS USED!")
    cuda_trials = sum(1 for d in trial_devices if d == 'cuda')
    print(f"   {cuda_trials}/{n_trials} trials used GPU")
else:
    print(f"⚠️  WARNING: No GPU usage detected!")

print(f"\nTrial time statistics:")
print(f"  Min: {min(trial_times):.2f}s")
print(f"  Max: {max(trial_times):.2f}s")
print(f"  Mean: {np.mean(trial_times):.2f}s")
print(f"  Median: {np.median(trial_times):.2f}s")

# Calculate theoretical vs actual speedup
sequential_time = sum(trial_times)
actual_time = optuna_time
theoretical_speedup = n_jobs
actual_speedup = sequential_time / actual_time

print(f"\nParallel efficiency:")
print(f"  Sequential time (sum of trials): {sequential_time:.2f}s")
print(f"  Actual parallel time: {actual_time:.2f}s")
print(f"  Theoretical speedup (n_jobs={n_jobs}): {theoretical_speedup:.1f}x")
print(f"  Actual speedup: {actual_speedup:.1f}x")
print(f"  Efficiency: {(actual_speedup/theoretical_speedup)*100:.1f}%")

if actual_speedup < 2 and n_jobs > 1:
    print(f"⚠️  WARNING: Low speedup detected! Parallelization may not be working properly.")
elif actual_speedup > 1.5:
    print(f"✅ GOOD SPEEDUP! Parallelization is working!")

# ============================================================================
# TEST 8: GPU Memory Check (if available)
# ============================================================================
if torch.cuda.is_available():
    print("\n[TEST 8] Final GPU Memory Status")
    print("-"*80)
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.4f} GB")
        print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1e9:.4f} GB")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

status = []
if cuda_available:
    status.append("✅ GPU: Available")
    if 'cuda' in unique_devices:
        status.append("✅ GPU Usage: Confirmed")
    else:
        status.append("❌ GPU Usage: Not detected")
else:
    status.append("❌ GPU: Not available")

if unique_pids > 1:
    status.append(f"✅ Parallelization: Working ({unique_pids} processes)")
else:
    status.append("❌ Parallelization: Not working")

if actual_speedup > 1.5:
    status.append(f"✅ Speedup: {actual_speedup:.1f}x")
else:
    status.append(f"⚠️  Speedup: Only {actual_speedup:.1f}x")

print("\n".join(status))

print(f"\n{'='*80}")
print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {time.time() - time.time():.2f} seconds")
print("="*80)

# Exit with appropriate code
if cuda_available and 'cuda' in unique_devices and unique_pids > 1:
    print("\n🎉 ALL SYSTEMS GO! Ready for full training!")
    sys.exit(0)
else:
    print("\n⚠️  Issues detected. Review the summary above.")
    sys.exit(1)
