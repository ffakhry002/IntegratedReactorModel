#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --wckey=edu_class
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06-23:59:59
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ffakhry@mit.edu

# ──────────────────────────────────────────────────────────────────────
# SLURM batch script for thesis Neural Network + Ray Tune HPO jobs.
# *** SAWTOOTH cluster (V100 GPUs, CUDA 12.x, 48 CPUs) ***
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
#   N_GPUS               GPUs to use      (default: 4, must match --gres)
#   TEST_C_FILE          Test Set C file  (default: data/test_c.txt)
# ──────────────────────────────────────────────────────────────────────

NN_CONFIG=${NN_CONFIG:?ERROR: NN_CONFIG must be set (1-5)}
NCI_DISTANCE_CUTOFF=${NCI_DISTANCE_CUTOFF:?ERROR: NCI_DISTANCE_CUTOFF must be set}

echo "========================================================"
echo "Job started at: $(date)"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node(s):        $(hostname)"
echo "Cluster:        SAWTOOTH (V100)"
echo "NN_CONFIG:      $NN_CONFIG"
echo "NCI_CUTOFF:     $NCI_DISTANCE_CUTOFF"
if [ -n "$FLUX_GROUP" ]; then
    echo "FLUX_GROUP:     $FLUX_GROUP (single-HPO-per-job mode)"
fi
echo "Trials:         ${N_TRIALS:-100}"
echo "Train file:     ${TRAIN_FILE:-data/train.txt}"
echo "GPUs:           ${N_GPUS:-4}"
echo "========================================================"

cd $SLURM_SUBMIT_DIR

# ── CUDA ────────────────────────────────────────────────────────────
module purge
module load cuda/12.9.1-er7a

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

# Disable PyTorch dynamo/JIT compiler — avoids 'skip_code' import crash
# in PyTorch 2.5.x on certain CUDA driver combinations.
export TORCHDYNAMO_DISABLE=1

NTHREADS=${SLURM_CPUS_PER_TASK:-48}

# ── PyTorch CUDA verification ──────────────────────────────────────
# NOTE: If PyTorch CUDA version is wrong, fix it on the LOGIN NODE
# (compute nodes may not have internet):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "========================================================"
echo "Checking PyTorch installation..."
echo "========================================================"

TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "NONE")
echo "PyTorch CUDA version: $TORCH_CUDA_VERSION"

if [ "$TORCH_CUDA_VERSION" != "12.1" ] && [ "$TORCH_CUDA_VERSION" != "12.4" ]; then
    echo "WARNING: PyTorch CUDA version mismatch (found: $TORCH_CUDA_VERSION, need: 12.x)"
    echo "Fix this on the login node before resubmitting:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
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
TARGET=${TARGET:-flux}
export NCI_DISTANCE_CUTOFF=${NCI_DISTANCE_CUTOFF}
export TRAIN_FILE=${TRAIN_FILE:-data/train.txt}
export TEST_C_FILE=${TEST_C_FILE:-data/test_c.txt}

cd /home/fakhfari/IntegratedReactorModel/ML/

N_GPUS=${N_GPUS:-4}

if [ "$TARGET" = "keff" ]; then
    echo "========================================================"
    echo "Starting NN keff with Ray Tune..."
    echo "  GPUs: ${N_GPUS}  |  CPUs: ${NTHREADS}  |  Trials: ${N_TRIALS:-100}"
    echo "========================================================"

    python -u main.py <<EOF
n
y
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

else
    export FLUX_MODE=energy_sixteen
    export NN_CONFIG=${NN_CONFIG}
    export FLUX_GROUP=${FLUX_GROUP:-}
    export FLUX_LOSS=mse

    echo "========================================================"
    echo "Starting NN thesis config ${NN_CONFIG} with Ray Tune..."
    echo "  GPUs: ${N_GPUS}  |  CPUs: ${NTHREADS}  |  Trials: ${N_TRIALS:-100}"
    if [ -n "$FLUX_GROUP" ]; then
        echo "  Flux group: ${FLUX_GROUP} (single-HPO mode)"
    fi
    echo "========================================================"

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

fi

echo "========================================================"
echo "Job completed at: $(date)"
echo "Runtime: $((SECONDS/60)) minutes"
echo "========================================================"
