#!/bin/bash
#SBATCH --time=06-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --partition=general
#SBATCH --wckey=edu_class
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ffakhry@mit.edu
#SBATCH --exclusive
#SBATCH --mem=0

# ──────────────────────────────────────────────────────────────────────
# Environment variables expected from submit_all.sh:
#   NCI_DISTANCE_CUTOFF  (0, 1, or 2)
#   N_TRIALS             (Optuna trials, default 5000)
# ──────────────────────────────────────────────────────────────────────

echo "=========================================================================="
echo "Job:        $SLURM_JOB_NAME"
echo "Job ID:     $SLURM_JOB_ID"
echo "Started:    $(date)"
echo "NCI mode:   CUTOFF=$NCI_DISTANCE_CUTOFF"
echo "Target:     keff"
echo "Trials:     ${N_TRIALS:-5000}"
echo "=========================================================================="

# ── Conda / module setup ────────────────────────────────────────────
module purge
export PATH=$HOME/.conda/envs/openmc_fresh/bin:$PATH
export PYTHONPATH=$HOME/.conda/envs/openmc_fresh/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/.conda/envs/openmc_fresh/lib:$LD_LIBRARY_PATH

# ── Thread control ──────────────────────────────────────────────────
NTHREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
export MPLBACKEND=Agg

# ── NCI + loss config (set before Python imports encoding_methods) ──
export NCI_DISTANCE_CUTOFF=${NCI_DISTANCE_CUTOFF}
export FLUX_LOSS=mse

cd "$HOME/IntegratedReactorModel/ML/" || exit 1

echo "Training keff with $NTHREADS threads..."
echo "=========================================================================="

# ── Feed answers to the interactive menu ────────────────────────────
# Prompt sequence (23 inputs — no flux mode or position pooling):
#  1  Train flux?                          → n
#     (flux mode prompt skipped)
#  2  Train keff?                          → y
#  3  XGBoost?                             → y
#  4  Random Forest?                       → n
#  5  SVM?                                 → n
#  6  Neural Net?                          → n
#  7  One-Hot encoding?                    → n
#  8  Categorical encoding?                → n
#  9  Physics encoding?                    → y
# 10  Spatial encoding?                    → n
# 11  Graph encoding?                      → n
# 12  Optuna?                              → y
# 13  Three-Stage?                         → n
# 14  Ray Tune?                            → n
# 15  Three-Stage Neural Net?              → n
# 16  No optimization?                     → n
#     (no position pooling — only for flux + physics + xgboost)
# 17  Optuna trials                        → $N_TRIALS
# 18  Parallel computing?                  → y
# 19  Number of cores                      → $NTHREADS
# 20  Training data path                   → (empty = default data/train.txt)
# 21  Test data path                       → (empty = default data/test.txt)
# 22  Proceed?                             → y

python -u main.py <<EOF
n
y
y
n
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
${N_TRIALS:-5000}
y
${NTHREADS}


y
EOF

echo "=========================================================================="
echo "Completed: $(date)"
echo "Runtime: $((SECONDS/60)) minutes"
echo "=========================================================================="
