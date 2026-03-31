#!/bin/bash
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=general
#SBATCH --wckey=edu_class
#SBATCH --mem=32G

# ──────────────────────────────────────────────────────────────────────
# Lightweight SLURM script for Set B clean re-evaluation.
# No Optuna search — just retrains with extracted best params (~10s).
#
# Environment variables expected:
#   LOG_DIR        path to the original .out log directory
#   TRAIN_FILE     training data file (e.g. data/train_lc_set_13.txt)
#   MAX_RUNS       (optional) max training runs for selected experiment
#   TARGET         flux or keff
#   FLUX_MODE      total, thermal_only, epithermal_only, fast_only
#   NCI_DISTANCE_CUTOFF  (default: 0)
# ──────────────────────────────────────────────────────────────────────

echo "=========================================================================="
echo "Job:         $SLURM_JOB_NAME"
echo "Job ID:      $SLURM_JOB_ID"
echo "Started:     $(date)"
echo "Target:      $TARGET"
echo "Log dir:     $LOG_DIR"
echo "Train file:  $TRAIN_FILE"
if [ -n "$MAX_RUNS" ]; then
    echo "Max runs:    $MAX_RUNS"
fi
if [ "$TARGET" = "flux" ]; then
    echo "Flux mode:   $FLUX_MODE"
fi
echo "=========================================================================="

# ── Conda / module setup ────────────────────────────────────────────
module purge
export PATH=$HOME/.conda/envs/openmc_fresh/bin:$PATH
export PYTHONPATH=$HOME/.conda/envs/openmc_fresh/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/.conda/envs/openmc_fresh/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

export NCI_DISTANCE_CUTOFF=${NCI_DISTANCE_CUTOFF:-0}
export TEST_C_FILE=${TEST_C_FILE:-data/test_c.txt}

cd "$HOME/IntegratedReactorModel/ML/" || exit 1

# ── Build command ───────────────────────────────────────────────────
CMD="python -u evaluate_setB_clean.py"
CMD="$CMD --log-dir $LOG_DIR"
CMD="$CMD --train-file $TRAIN_FILE"
CMD="$CMD --target $TARGET"

if [ -n "$MAX_RUNS" ]; then
    CMD="$CMD --max-runs $MAX_RUNS"
fi

if [ "$TARGET" = "flux" ]; then
    CMD="$CMD --flux-mode ${FLUX_MODE:-total}"
fi

echo "Running: $CMD"
echo "=========================================================================="

$CMD

echo "=========================================================================="
echo "Completed: $(date)"
echo "Runtime: $((SECONDS/60)) minutes ($SECONDS seconds)"
echo "=========================================================================="
