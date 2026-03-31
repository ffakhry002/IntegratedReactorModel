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
# Consolidated SLURM script for both flux and keff training.
#
# Environment variables expected:
#   TARGET               (flux or keff)
#   NCI_DISTANCE_CUTOFF  (0, 1, or 2)
#   N_GEOMETRIES         (optional: number of geometric configs to use, 1-256)
#   TRAIN_FILE           (optional: path to training data file, e.g. data/train_lc_set_195.txt)
#   FLUX_MODE            (optional: set to energy_sixteen for 16-channel thesis NN + Ray Tune)
#   NN_CONFIG            (optional: thesis layout 1–5 when using neural_net + raytune + energy_sixteen)
#   FLUX_GROUP           (optional: 0–3 flux-group index for single-HPO-per-job mode, configs 2/3/5)
#   FLUX_MODE_NAME       (total, thermal, epithermal, fast — flux only)
#   FLUX_CHOICE          (1, 4, 5, or 6 — flux only)
#   LOG_DIR              (path for .out/.err)
#   N_TRIALS             (Optuna trials, default 5000)
#
# When TRAIN_FILE is set, N_GEOMETRIES is ignored (the file already
# contains exactly the right training data for that step).
# ──────────────────────────────────────────────────────────────────────

TARGET=${TARGET:-flux}

echo "=========================================================================="
echo "Job:         $SLURM_JOB_NAME"
echo "Job ID:      $SLURM_JOB_ID"
echo "Started:     $(date)"
echo "Target:      $TARGET"
echo "NCI mode:    CUTOFF=$NCI_DISTANCE_CUTOFF"
if [ "$TARGET" = "flux" ]; then
    echo "Flux mode:   $FLUX_MODE_NAME (menu choice $FLUX_CHOICE)"
fi
if [ -n "$NN_CONFIG" ]; then
    echo "NN_CONFIG:   $NN_CONFIG"
fi
if [ -n "$FLUX_GROUP" ]; then
    echo "FLUX_GROUP:  $FLUX_GROUP (single-HPO mode)"
fi
echo "Trials:      ${N_TRIALS:-5000}"
if [ -n "$TRAIN_FILE" ]; then
    echo "TRAIN_FILE:  $TRAIN_FILE (independent training set)"
    unset N_GEOMETRIES
elif [ -n "$N_GEOMETRIES" ]; then
    echo "N_GEOMETRIES: $N_GEOMETRIES (training on first $((N_GEOMETRIES * 33)) RUNs)"
else
    echo "N_GEOMETRIES: ALL (full dataset)"
fi
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

# ── Config (set before Python imports encoding_methods) ─────────────
export NCI_DISTANCE_CUTOFF=${NCI_DISTANCE_CUTOFF}
export FLUX_LOSS=mse
if [ -n "${N_GEOMETRIES:-}" ]; then
    export N_GEOMETRIES=${N_GEOMETRIES}
fi
export TRAIN_FILE=${TRAIN_FILE:-}
export TEST_C_FILE=${TEST_C_FILE:-data/test_c.txt}
export FLUX_MODE=${FLUX_MODE:-}
export NN_CONFIG=${NN_CONFIG:-}
export FLUX_GROUP=${FLUX_GROUP:-}

cd "$HOME/IntegratedReactorModel/ML/" || exit 1

echo "Training $TARGET with $NTHREADS threads..."
echo "=========================================================================="

# ── Feed answers to the interactive menu ────────────────────────────
if [ "$TARGET" = "keff" ]; then
    # Keff menu sequence (22 inputs — no flux mode or position pooling)
    #  1  Train flux?                          → n
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
    # 20  Training data path                   → $TRAIN_FILE or empty (default data/train.txt)
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
${TRAIN_FILE:-}

y
EOF

else
    # Flux menu sequence (24 inputs)
    #  1  Train flux?                          → y
    #  2  Flux mode (1-6)                      → $FLUX_CHOICE
    #  3  Train keff?                          → n
    #  4  XGBoost?                             → y
    #  5  Random Forest?                       → n
    #  6  SVM?                                 → n
    #  7  Neural Net?                          → n
    #  8  One-Hot encoding?                    → n
    #  9  Categorical encoding?                → n
    # 10  Physics encoding?                    → y
    # 11  Spatial encoding?                    → n
    # 12  Graph encoding?                      → n
    # 13  Optuna?                              → y
    # 14  Three-Stage?                         → n
    # 15  Ray Tune?                            → n
    # 16  Three-Stage Neural Net?              → n
    # 17  No optimization?                     → n
    # 18  Position pooling (restructured)?     → y
    # 19  Optuna trials                        → $N_TRIALS
    # 20  Parallel computing?                  → y
    # 21  Number of cores                      → $NTHREADS
    # 22  Training data path                   → $TRAIN_FILE or empty (default data/train.txt)
    # 23  Test data path                       → (empty = default data/test.txt)
    # 24  Proceed?                             → y
    python -u main.py <<EOF
y
${FLUX_CHOICE}
n
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
y
${N_TRIALS:-5000}
y
${NTHREADS}
${TRAIN_FILE:-}

y
EOF

fi

echo "=========================================================================="
echo "Completed: $(date)"
echo "Runtime: $((SECONDS/60)) minutes"
echo "=========================================================================="
