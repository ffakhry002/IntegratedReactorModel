#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --wckey=edu_class
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06-23:59:59
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ffakhry@mit.edu

# ──────────────────────────────────────────────────────────────────────
# SLURM batch script for thesis Neural Network + Ray Tune HPO jobs.
#
# Runs ONE HPO study for the specified NN_CONFIG and (optionally)
# FLUX_GROUP.  For multi-network configs (2, 3, 5), submit one job per
# flux group via the companion submit_nn_thesis.sh script.
#
# Required environment variables (passed by submit script or --export):
#   NN_CONFIG            Thesis layout 1–5
#   NCI_DISTANCE_CUTOFF  0, 1, or 2
#
# Optional environment variables:
#   FLUX_GROUP           0–3 (required for configs 2/3/5 single-HPO mode)
#   TRAIN_FILE           Training data path (default: data/train.txt)
#   N_TRIALS             Ray Tune trials  (default: 100)
#   N_GPUS               GPUs to use      (default: 3, must match --gres)
#   TEST_C_FILE          Test Set C file  (default: data/test_c.txt)
# ──────────────────────────────────────────────────────────────────────

NN_CONFIG=${NN_CONFIG:?ERROR: NN_CONFIG must be set (1-5)}
NCI_DISTANCE_CUTOFF=${NCI_DISTANCE_CUTOFF:?ERROR: NCI_DISTANCE_CUTOFF must be set}

echo "========================================================"
echo "Job started at: $(date)"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node(s):        $(hostname)"
echo "NN_CONFIG:      $NN_CONFIG"
echo "NCI_CUTOFF:     $NCI_DISTANCE_CUTOFF"
if [ -n "$FLUX_GROUP" ]; then
    echo "FLUX_GROUP:     $FLUX_GROUP (single-HPO-per-job mode)"
fi
echo "Trials:         ${N_TRIALS:-100}"
echo "Train file:     ${TRAIN_FILE:-data/train.txt}"
echo "GPUs:           ${N_GPUS:-3}"
echo "========================================================"

cd $SLURM_SUBMIT_DIR

# ── CUDA ────────────────────────────────────────────────────────────
module purge
module load cuda/11.8.0-gcc-11.5.0-hfmv

echo "CUDA loaded:"
nvcc --version

# ── Python environment ──────────────────────────────────────────────
export PATH=$HOME/.conda/envs/openmc_fresh/bin:$PATH
export PYTHONPATH=$HOME/.conda/envs/openmc_fresh/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/.conda/envs/openmc_fresh/lib:$LD_LIBRARY_PATH

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# ── Threading ───────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export TUNE_DISABLE_AUTO_CALLBACK_LOGGERS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
export OMP_PROC_BIND=false
export OMP_PLACES=threads

NTHREADS=${SLURM_CPUS_PER_TASK:-32}

# ── PyTorch CUDA verification ──────────────────────────────────────
echo "========================================================"
echo "Checking PyTorch installation..."
echo "========================================================"

TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "NONE")

if [ "$TORCH_CUDA_VERSION" != "11.8" ]; then
    echo "PyTorch CUDA version mismatch (found: $TORCH_CUDA_VERSION, need: 11.8)"
    echo "Reinstalling PyTorch with CUDA 11.8..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "PyTorch CUDA 11.8 already installed"
fi

ray stop --force 2>/dev/null || true

echo "========================================================"
echo "Final PyTorch Test:"
echo "========================================================"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "GPU Info:"
nvidia-smi

# ── Config exports (read by interactive_menu.py) ───────────────────
export NCI_DISTANCE_CUTOFF=${NCI_DISTANCE_CUTOFF}
export FLUX_MODE=energy_sixteen
export NN_CONFIG=${NN_CONFIG}
export FLUX_GROUP=${FLUX_GROUP:-}
export TRAIN_FILE=${TRAIN_FILE:-data/train.txt}
export TEST_C_FILE=${TEST_C_FILE:-data/test_c.txt}
export FLUX_LOSS=mse

cd /home/fakhfari/IntegratedReactorModel/ML/

N_GPUS=${N_GPUS:-3}

echo "========================================================"
echo "Starting NN thesis config ${NN_CONFIG} with Ray Tune..."
echo "  GPUs: ${N_GPUS}  |  CPUs: ${NTHREADS}  |  Trials: ${N_TRIALS:-100}"
if [ -n "$FLUX_GROUP" ]; then
    echo "  Flux group: ${FLUX_GROUP} (single-HPO mode)"
fi
echo "========================================================"

# ──────────────────────────────────────────────────────────────────────
# Heredoc for the interactive menu (22 inputs).
#
# FLUX_MODE, NN_CONFIG, FLUX_GROUP, and TRAIN_FILE are set as env vars
# so the menu auto-reads them without prompting.
#
#  1  Train flux?                    → y
#     (FLUX_MODE=energy_sixteen from env — no input needed)
#  2  Train keff?                    → n
#  3  XGBoost?                       → n
#  4  Random Forest?                 → n
#  5  SVM?                           → n
#  6  Neural Net?                    → y
#  7  One-Hot encoding?              → n
#  8  Categorical encoding?          → n
#  9  Physics encoding?              → y
# 10  Spatial encoding?              → n
# 11  Graph encoding?                → n
# 12  Optuna?                        → n
# 13  Three-Stage?                   → n
# 14  Ray Tune?                      → y
# 15  Three-Stage Neural Net?        → n
# 16  No optimization?               → n
#     (position pooling NOT asked — requires xgboost)
# 17  Ray Tune trials                → N_TRIALS
# 18  Parallel computing?            → y
# 19  Number of cores                → NTHREADS
# 20  Number of GPUs                 → N_GPUS  (asked because >1 GPU)
#     (NN_CONFIG from env — no input)
#     (FLUX_GROUP from env — no input)
#     (TRAIN_FILE from env — no input)
# 21  Test data path                 → (empty = default data/test.txt)
# 22  Proceed?                       → y
#
# NOTE: Line 20 (GPU count) is only asked when >1 GPU is detected.
#       If you change --gres to a single GPU, REMOVE line 20.
# ──────────────────────────────────────────────────────────────────────
python -u main.py <<EOF
y
n
n
n
n
y
n
n
y
n
n
n
n
y
n
n
${N_TRIALS:-100}
y
${NTHREADS}
${N_GPUS}

y
EOF

echo "========================================================"
echo "Job completed at: $(date)"
echo "Runtime: $((SECONDS/60)) minutes"
echo "========================================================"
