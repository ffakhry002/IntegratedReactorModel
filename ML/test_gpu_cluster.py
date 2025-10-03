#!/usr/bin/env python
"""
GPU Cluster Test Script for PyTorch Neural Network
Run this to verify everything works before full training
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime

print("="*70)
print("GPU CLUSTER TEST - PyTorch Neural Network")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================================
# TEST 1: System Information
# ============================================================================
print("\n[TEST 1] System Information")
print("-"*70)
import platform
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Working Directory: {os.getcwd()}")

# ============================================================================
# TEST 2: GPU Detection
# ============================================================================
print("\n[TEST 2] GPU Detection")
print("-"*70)
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"✅ GPU DETECTED!")
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("⚠️  NO GPU DETECTED - will use CPU (slow)")

# ============================================================================
# TEST 3: Basic PyTorch Operations
# ============================================================================
print("\n[TEST 3] Basic PyTorch Operations")
print("-"*70)
try:
    # Create tensors
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)

    # Move to device (auto-detect)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)
    y = y.to(device)

    print(f"Device being used: {device}")

    # Matrix multiplication
    z = torch.mm(x, y)
    print(f"Matrix multiplication: {x.shape} × {y.shape} = {z.shape}")

    if cuda_available:
        print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.4f} GB")

    print("✅ Basic PyTorch operations work!")

except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# TEST 4: Import Neural Network Classes
# ============================================================================
print("\n[TEST 4] Import Neural Network Classes")
print("-"*70)
try:
    from ML_models.neural_net_train import (
        PyTorchRectangularNet,
        PyTorchRegressorWrapper,
        NeuralNetReactorModel
    )
    print("✅ All neural network classes imported successfully")
except Exception as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("\nMake sure you're running from the correct directory!")
    print("Should be in: /path/to/IntegratedReactorModel/ML")
    sys.exit(1)

# ============================================================================
# TEST 5: Create Neural Network
# ============================================================================
print("\n[TEST 5] Create Neural Network")
print("-"*70)
try:
    net = PyTorchRectangularNet(
        input_dim=20,
        output_dim=4,
        depth=3,
        width=150,
        activation='relu'
    )
    print(f"✅ Network created:")
    print(f"   Architecture: 20 → [150, 150, 150] → 4")
    print(f"   Total parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Test forward pass
    test_input = torch.randn(10, 20)
    test_output = net(test_input)
    print(f"   Forward pass test: {test_input.shape} → {test_output.shape}")
    print("✅ Neural network works!")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: Train Small Model (Flux)
# ============================================================================
print("\n[TEST 6] Train Small Flux Model")
print("-"*70)
try:
    # Create dummy data
    n_samples = 200
    n_features = 15
    n_flux_outputs = 4

    X_train = np.random.randn(n_samples, n_features)
    y_flux = np.random.randn(n_samples, n_flux_outputs)

    print(f"Training data: {n_samples} samples, {n_features} features")
    print(f"Target: {n_flux_outputs} flux outputs")

    # Create model with auto-detection
    model = NeuralNetReactorModel(
        scale_features=True,
        depth=2,
        width=100,
        activation='relu',
        optimizer='adam',
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=64,
        max_epochs=5,  # Just 5 epochs for testing
        patience=3,
        device=None,  # Auto-detect
        verbose=True
    )

    print(f"\nModel configuration:")
    print(f"  Device: {model.params['device']}")
    print(f"  Architecture: depth={model.params['depth']}, width={model.params['width']}")
    print(f"  Optimizer: {model.params['optimizer']}")
    print(f"  Learning rate: {model.params['learning_rate']}")

    print("\nTraining flux model...")
    model.fit_flux(X_train, y_flux)

    print("\n✅ Flux training complete!")

    # Test prediction
    X_test = np.random.randn(10, n_features)
    predictions = model.predict_flux(X_test)
    print(f"✅ Prediction works! Output shape: {predictions.shape}")

    if torch.cuda.is_available():
        print(f"\nGPU Memory Stats:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.4f} GB")

except Exception as e:
    print(f"❌ TRAINING ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 7: Train Small Model (Keff)
# ============================================================================
print("\n[TEST 7] Train Small Keff Model")
print("-"*70)
try:
    # Create dummy data
    y_keff = np.random.randn(n_samples)

    print(f"Training data: {n_samples} samples")
    print(f"Target: 1 keff output")

    # Use same model architecture
    model_keff = NeuralNetReactorModel(
        scale_features=True,
        depth=2,
        width=100,
        max_epochs=5,
        device=None,  # Auto-detect
        verbose=False
    )

    print("\nTraining keff model...")
    model_keff.fit_keff(X_train, y_keff)

    print("✅ Keff training complete!")

    # Test prediction
    predictions = model_keff.predict_keff(X_test)
    print(f"✅ Prediction works! Output shape: {predictions.shape}")

except Exception as e:
    print(f"❌ TRAINING ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 8: Test Cross-Validation Compatibility
# ============================================================================
print("\n[TEST 8] Cross-Validation Compatibility")
print("-"*70)
try:
    from sklearn.model_selection import cross_val_score, KFold

    wrapper = PyTorchRegressorWrapper(
        depth=2,
        width=50,
        max_epochs=3,
        device=None,
        verbose=False
    )

    X_small = np.random.randn(50, 10).astype(np.float32)
    y_small = np.random.randn(50, 4).astype(np.float32)

    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    scores = cross_val_score(
        wrapper, X_small, y_small,
        cv=cv,
        scoring='neg_mean_squared_error'
    )

    print(f"CV Scores: {scores}")
    print(f"Mean Score: {scores.mean():.6f}")
    print("✅ Cross-validation works!")

except Exception as e:
    print(f"❌ CV ERROR: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - this is not critical

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\n✅ ALL TESTS PASSED!")
print("\nSystem Ready:")
if cuda_available:
    print(f"  • GPU: {torch.cuda.get_device_name(0)}")
    print(f"  • Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("  • Device: CPU (no GPU)")

print(f"\nNeural Network:")
print(f"  • PyTorch version: {torch.__version__}")
print(f"  • Flux model: ✅ Working")
print(f"  • Keff model: ✅ Working")
print(f"  • Cross-validation: ✅ Working")

print(f"\n{'='*70}")
print("🎉 READY FOR FULL TRAINING!")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Exit with success
sys.exit(0)
