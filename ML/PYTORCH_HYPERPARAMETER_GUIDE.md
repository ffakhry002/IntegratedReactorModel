# PyTorch Neural Network Hyperparameter Guide

Quick reference for understanding and tuning your PyTorch neural network hyperparameters.

---

## 🏗️ Architecture Parameters

### `depth` (Number of Hidden Layers)
**Range:** 1-5
**Default:** 2

**What it does:** Controls how many hidden layers the network has.

**Guidelines:**
- **1-2 layers:** Good for simple, linear-ish relationships
- **3-4 layers:** Good for complex, non-linear patterns
- **5+ layers:** Risk of overfitting with small datasets (~2000 samples)

**Trade-offs:**
- ✅ More layers = More expressive power
- ❌ More layers = Longer training, more risk of overfitting

### `width` (Neurons Per Layer)
**Range:** 50-400
**Default:** 100

**What it does:** How many neurons in each hidden layer (uniform across all layers).

**Guidelines:**
- **50-100:** Lightweight, fast, good for simple problems
- **100-200:** Balanced, works well for most cases
- **200-400:** Heavy, powerful, needs more data to avoid overfitting

**Trade-offs:**
- ✅ More width = More capacity to learn patterns
- ❌ More width = More parameters, slower, more GPU memory

**Rule of thumb:** Start with width = 100-150, increase if underfitting.

---

## 🎨 Activation Function

### `activation`
**Options:** 'relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu'
**Default:** 'relu'

**What it does:** Non-linear transformation between layers.

**Comparison:**

| Activation | Speed | Range | Best For | Downsides |
|-----------|-------|-------|----------|-----------|
| **relu** | Fast | [0, ∞) | General purpose, default choice | Dead neurons possible |
| **tanh** | Medium | [-1, 1] | Centered data, symmetric outputs | Gradient vanishing |
| **sigmoid** | Slow | [0, 1] | Probability-like outputs | Strong gradient vanishing |
| **elu** | Medium | (-α, ∞) | Avoiding dead neurons | Slightly slower than ReLU |
| **leaky_relu** | Fast | (-∞, ∞) | Avoiding dead neurons | Extra hyperparameter |

**Recommendation:** Start with **relu**, try **tanh** if relu doesn't work well.

---

## 🚀 Optimizer

### `optimizer`
**Options:** 'adam', 'sgd', 'adamw', 'rmsprop'
**Default:** 'adam'

**What it does:** Algorithm for updating weights during training.

**Comparison:**

| Optimizer | Speed | Memory | Best For | Notes |
|-----------|-------|--------|----------|-------|
| **adam** | Fast convergence | High | General purpose, default | Adaptive learning rate |
| **sgd** | Slow convergence | Low | Fine-tuning, when overfitting | Needs momentum (auto-set) |
| **adamw** | Fast convergence | High | When regularization important | Adam + better weight decay |
| **rmsprop** | Medium | Medium | RNNs, unstable gradients | Less popular now |

**Recommendation:** Start with **adam**, try **adamw** if you need stronger regularization.

---

## 📚 Learning Parameters

### `learning_rate`
**Range:** 0.0001-0.01 (log scale)
**Default:** 0.001

**What it does:** How big of a step to take when updating weights.

**Guidelines:**
- **0.0001-0.0005:** Conservative, slow but stable
- **0.001:** Good default, works for most cases
- **0.005-0.01:** Aggressive, fast but may be unstable

**Signs of bad learning rate:**
- Too high: Loss explodes or oscillates
- Too low: Training is very slow, gets stuck

**Recommendation:** Start with **0.001**, decrease if unstable, increase if too slow.

### `batch_size`
**Options:** 32, 64, 128, 256, 512
**Default:** 128

**What it does:** How many samples to process before updating weights.

**Guidelines:**
- **32-64:** More updates per epoch, better for small datasets
- **128:** Good default balance
- **256-512:** Faster training, needs more memory

**Trade-offs:**
- ✅ Larger batch = Faster training (parallelism)
- ✅ Smaller batch = Better generalization
- ❌ Larger batch = More GPU memory
- ❌ Smaller batch = Noisier gradient updates

**GPU Memory Guide:**
- 2GB GPU: Use 64 or 128
- 4GB GPU: Use 128 or 256
- 8GB+ GPU: Use 256 or 512

---

## 🛡️ Regularization

### `weight_decay` (L2 Regularization)
**Range:** 0.00001-0.1 (log scale)
**Default:** 0.001

**What it does:** Penalizes large weights to prevent overfitting.

**Guidelines:**
- **0.00001-0.0001:** Light regularization
- **0.001-0.01:** Medium regularization (recommended)
- **0.01-0.1:** Strong regularization (if heavily overfitting)

**Signs you need more regularization:**
- Training loss much lower than validation loss
- Model performs well on train set, poorly on test set

**Signs you have too much regularization:**
- Both training and validation loss are high
- Model is underfitting

**Recommendation:** Start with **0.001**, increase if overfitting.

---

## ⏱️ Training Duration

### `max_epochs`
**Range:** 200-1500
**Default:** 1000

**What it does:** Maximum number of times to go through the entire dataset.

**Guidelines:**
- **200-500:** Quick experiments
- **500-1000:** Standard training
- **1000-1500:** Thorough search, with early stopping

**Note:** With early stopping, actual training usually stops much earlier (50-200 epochs).

### `patience`
**Range:** 10-40
**Default:** 20

**What it does:** How many epochs to wait without improvement before stopping early.

**Guidelines:**
- **10-15:** Stop quickly, less thorough
- **20-30:** Good balance (recommended)
- **30-40:** Very patient, thorough search

**Trade-off:**
- ✅ Higher patience = More thorough training
- ❌ Higher patience = Longer training time

---

## 🎯 Hyperparameter Tuning Strategy

### 1. Start with Defaults
```python
model = NeuralNetReactorModel(
    depth=2,
    width=100,
    activation='relu',
    optimizer='adam',
    learning_rate=0.001,
    weight_decay=0.001,
    batch_size=128,
    max_epochs=1000,
    patience=20
)
```

### 2. Check for Overfitting/Underfitting

**If Underfitting (poor train AND test performance):**
1. Increase `width` (100 → 200)
2. Increase `depth` (2 → 3)
3. Decrease `weight_decay` (0.001 → 0.0001)
4. Train longer (increase `max_epochs`)

**If Overfitting (good train, poor test):**
1. Increase `weight_decay` (0.001 → 0.01)
2. Decrease `width` (200 → 100)
3. Decrease `depth` (3 → 2)
4. Use smaller `batch_size` (128 → 64)

### 3. Optimize Learning Rate

**Too slow:**
- Increase `learning_rate` (0.001 → 0.003)

**Unstable (loss oscillating):**
- Decrease `learning_rate` (0.001 → 0.0005)
- Try different `optimizer` (adam → adamw)

### 4. Final Tuning

Once you have rough architecture, use Optuna to fine-tune all parameters together:

```python
from hyperparameter_tuning.optuna_optimization import optimize_flux_model

best_params, study = optimize_flux_model(
    X_train, y_flux_train,
    model_type='neural_net',
    n_trials=500,  # More trials = better results
    n_jobs=-1,     # Use all CPU cores
    groups=groups_train,
    flux_mode='total'
)
```

---

## 📊 Quick Diagnosis Table

| Symptom | Likely Cause | Try This |
|---------|--------------|----------|
| Loss not decreasing | Learning rate too low | Increase learning_rate |
| Loss exploding | Learning rate too high | Decrease learning_rate |
| Train loss << Val loss | Overfitting | Increase weight_decay |
| Train loss ≈ Val loss, both high | Underfitting | Increase depth or width |
| Training very slow | Batch size too small | Increase batch_size |
| GPU out of memory | Batch size too large | Decrease batch_size |
| Stops early, not improving | Patience too low | Increase patience |
| Takes forever to train | Max_epochs too high | Decrease max_epochs (let early stopping handle it) |

---

## 🔬 Advanced Tips

### For Small Datasets (~2000 samples)
- Keep `depth` ≤ 3
- Use moderate `width` (100-150)
- Use strong regularization (`weight_decay` = 0.01)
- Use smaller `batch_size` (64)

### For Large Datasets (>10,000 samples)
- Can use `depth` = 4-5
- Can use larger `width` (200-400)
- Can use lighter regularization (`weight_decay` = 0.0001)
- Use larger `batch_size` (256-512)

### For GPU Optimization
- Maximize `batch_size` within memory limits
- Use `depth` × `width` product < 2000 for memory efficiency
- Example: depth=3, width=200 → 3×200 = 600 neurons total ✅
- Example: depth=5, width=400 → 5×400 = 2000 neurons total (may be slow)

### For Imbalanced Targets
- Your flux values have very different scales
- Using log-transformed data helps
- Consider `tanh` activation instead of `relu`
- Use `weight_decay` = 0.001-0.01

---

## 🎓 Summary

**Most Important Hyperparameters (in order):**
1. **learning_rate** - Directly affects training success
2. **weight_decay** - Controls overfitting
3. **width** - Controls model capacity
4. **depth** - Adds complexity
5. **batch_size** - Affects speed and stability

**Safe Starting Point:**
```python
depth=2, width=150, learning_rate=0.001, weight_decay=0.001,
batch_size=128, optimizer='adam', activation='relu'
```

**When in Doubt:**
- Use Optuna with 250-500 trials
- It will find good hyperparameters automatically
- Trust the MAPE scores from cross-validation

Good luck with your nuclear reactor flux predictions! 🚀☢️
