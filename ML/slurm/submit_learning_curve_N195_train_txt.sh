#!/bin/bash
#
# Submit N=195 learning curve jobs (5 targets) using ORDERED data/train.txt
# (algorithmic / "selected" curve: first N geometries × 33 fills).
#
# This is NOT the random LC: we do NOT use data/train_lc_set_195.txt.
#
#   • N_GEOMETRIES=195  → load first 195×33 RUNs from data/train.txt
#   • TRAIN_FILE empty → main.py defaults to data/train.txt (see cpu_ml.sh heredoc)
#
# If your train.txt is already exactly 195 geometries and you want ALL rows with
# no truncation, remove N_GEOMETRIES from --export below (keep TRAIN_FILE empty).
#
# Folder structure:
#   ML_logs/learning_curve_N195_train_txt/
#   ├── N195_total/
#   ├── N195_thermal/
#   ├── ...
#   └── N195_keff/

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# ── Configuration ───────────────────────────────────────────────────
N_TRIALS=${N_TRIALS:-5000}
NCI_DISTANCE_CUTOFF=0
DELAY_SECONDS=${DELAY_SECONDS:-1}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/cpu_ml.sh"

N_GEO=195

# 5 targets: 4 flux modes + keff
# Format: "TARGET FLUX_MODE_NAME FLUX_CHOICE"
TARGETS=(
    "flux total 1"
    "flux thermal_only 4"
    "flux epithermal_only 5"
    "flux fast_only 6"
    "keff keff 0"
)

# ── Create log directories ──────────────────────────────────────────
BASE_LOG="$HOME/IntegratedReactorModel/ML/ML_logs/learning_curve_N195_train_txt"
mkdir -p "${BASE_LOG}/summary"

for target_spec in "${TARGETS[@]}"; do
    read -r target_type flux_name flux_choice <<< "$target_spec"
    if [ "$target_type" = "keff" ]; then
        mkdir -p "${BASE_LOG}/N$(printf '%03d' $N_GEO)_keff"
    else
        mkdir -p "${BASE_LOG}/N$(printf '%03d' $N_GEO)_${flux_name}"
    fi
done

TOTAL=${#TARGETS[@]}

echo "=========================================================================="
echo " Learning curve N=195 — data/train.txt (selected / ordered subset)"
echo "=========================================================================="
echo ""
echo " N_GEOMETRIES: $N_GEO  →  first $((N_GEO * 33)) RUNs from data/train.txt"
echo " TRAIN_FILE:   (empty — NOT train_lc_set_*.txt)"
echo " Targets (${#TARGETS[@]}): total, thermal, epithermal, fast, keff"
echo " NCI cutoff: $NCI_DISTANCE_CUTOFF"
echo " Optuna trials per job: $N_TRIALS"
echo " Total jobs: $TOTAL"
echo " Delay between submissions: ${DELAY_SECONDS}s"
echo ""
echo "=========================================================================="

SUBMITTED=0

for target_spec in "${TARGETS[@]}"; do
    read -r target_type flux_name flux_choice <<< "$target_spec"

    if [ "$target_type" = "keff" ]; then
        label="keff"
    else
        label="${flux_name}"
    fi

    log_dir="${BASE_LOG}/N$(printf '%03d' $N_GEO)_${label}"
    job_name="lc_N${N_GEO}_${label}"

    ((SUBMITTED++))

    echo ""
    echo -e "${BLUE}[${SUBMITTED}/${TOTAL}] ${job_name}${NC}"
    echo "  N_GEOMETRIES=$N_GEO  TRAIN_FILE=(default data/train.txt, truncated to first $((N_GEO * 33)) RUNs)"
    echo "  Target: $target_type ($label)"
    echo "  Logs: ${log_dir}/"

    # TRAIN_FILE= clears any inherited TRAIN_FILE; N_GEOMETRIES selects subset from train.txt
    sbatch \
        --job-name="${job_name}" \
        --output="${log_dir}/ml_%x_%j.out" \
        --error="${log_dir}/ml_%x_%j.err" \
        --export="ALL,TARGET=${target_type},NCI_DISTANCE_CUTOFF=${NCI_DISTANCE_CUTOFF},FLUX_MODE_NAME=${flux_name},FLUX_CHOICE=${flux_choice},TRAIN_FILE=,N_GEOMETRIES=${N_GEO},N_TRIALS=${N_TRIALS},TEST_C_FILE=${TEST_C_FILE:-data/test_c.txt}" \
        --wckey=edu_class \
        "${SLURM_SCRIPT}"

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Submitted${NC}"
    else
        echo -e "  ${RED}✗ Failed${NC}"
    fi

    if [ $SUBMITTED -lt $TOTAL ]; then
        sleep "${DELAY_SECONDS}"
    fi
done

echo ""
echo "=========================================================================="
echo -e "${GREEN}All ${TOTAL} jobs submitted!${NC}"
echo "=========================================================================="
echo ""
echo "Log base: ${BASE_LOG}/"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${BASE_LOG}/N195_total/ml_*.out"
echo ""
echo "Collect results (point at ML_logs or this subfolder):"
echo "  bash slurm/collect_results.sh \"\$HOME/IntegratedReactorModel/ML/ML_logs\""
