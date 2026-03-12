#!/bin/bash
#
# Submit 15 ML training jobs:
#   4 flux modes × 3 NCI settings = 12 flux jobs
#   1 keff       × 3 NCI settings =  3 keff jobs
#
# Folder structure created:
#   ML_logs/
#   ├── NCI_cutoff/        (NCI_DISTANCE_CUTOFF=1)
#   ├── NCI_no_cutoff/     (NCI_DISTANCE_CUTOFF=0)
#   └── No_NCI/            (NCI_DISTANCE_CUTOFF=2)

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ── Configuration ───────────────────────────────────────────────────
N_TRIALS=${N_TRIALS:-5000}
DELAY_SECONDS=${DELAY_SECONDS:-10}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/cpu_ml.sh"
SLURM_SCRIPT_KEFF="${SCRIPT_DIR}/cpu_ml_keff.sh"

# ── Create log directories ──────────────────────────────────────────
BASE_LOG="$HOME/IntegratedReactorModel/ML/ML_logs"
mkdir -p "${BASE_LOG}/NCI_cutoff"
mkdir -p "${BASE_LOG}/NCI_no_cutoff"
mkdir -p "${BASE_LOG}/No_NCI"

echo "=========================================================================="
echo " Submitting 3 ML Training Jobs (keff only)"
echo "=========================================================================="
echo ""
echo " NCI settings:"
echo "   NCI_cutoff     → NCI_DISTANCE_CUTOFF=1  (d > sqrt(5) → 0)"
echo "   NCI_no_cutoff  → NCI_DISTANCE_CUTOFF=0  (all distances contribute)"
echo "   No_NCI         → NCI_DISTANCE_CUTOFF=2  (NCI features disabled)"
echo ""
echo " Keff:        1 job per NCI setting"
# echo " Flux modes:  total, thermal, epithermal, fast"
echo " Model:       XGBoost"
echo " Encoding:    Physics-based"
echo " Optimizer:   Optuna (${N_TRIALS} trials)"
echo " Loss:        MSE"
echo " Delay:       ${DELAY_SECONDS}s between submissions"
echo ""
echo "=========================================================================="

SUBMITTED=0
TOTAL=3

submit_flux_job() {
    local nci_cutoff=$1
    local nci_label=$2      # folder name: NCI_cutoff, NCI_no_cutoff, No_NCI
    local flux_name=$3      # total, thermal, epithermal, fast
    local flux_choice=$4    # menu index: 1, 4, 5, 6

    local log_dir="${BASE_LOG}/${nci_label}"
    local job_name="xgb_${flux_name}_nci${nci_cutoff}"

    ((SUBMITTED++))

    echo ""
    echo -e "${BLUE}[${SUBMITTED}/${TOTAL}] ${job_name}${NC}"
    echo "  NCI: ${nci_label} (cutoff=${nci_cutoff})"
    echo "  Flux: ${flux_name} (choice=${flux_choice})"
    echo "  Logs: ${log_dir}/"

    sbatch \
        --job-name="${job_name}" \
        --output="${log_dir}/ml_%x_%j.out" \
        --error="${log_dir}/ml_%x_%j.err" \
        --export="ALL,NCI_DISTANCE_CUTOFF=${nci_cutoff},FLUX_MODE_NAME=${flux_name},FLUX_CHOICE=${flux_choice},N_TRIALS=${N_TRIALS}" \
        --wckey=edu_class \
        "${SLURM_SCRIPT}"

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Submitted${NC}"
    else
        echo -e "  ${RED}✗ Failed${NC}"
    fi

    if [ $SUBMITTED -lt $TOTAL ]; then
        echo -e "  ${YELLOW}Waiting ${DELAY_SECONDS}s...${NC}"
        sleep "${DELAY_SECONDS}"
    fi
}

submit_keff_job() {
    local nci_cutoff=$1
    local nci_label=$2      # folder name: NCI_cutoff, NCI_no_cutoff, No_NCI

    local log_dir="${BASE_LOG}/${nci_label}"
    local job_name="xgb_keff_nci${nci_cutoff}"

    ((SUBMITTED++))

    echo ""
    echo -e "${BLUE}[${SUBMITTED}/${TOTAL}] ${job_name}${NC}"
    echo "  NCI: ${nci_label} (cutoff=${nci_cutoff})"
    echo "  Target: keff"
    echo "  Logs: ${log_dir}/"

    sbatch \
        --job-name="${job_name}" \
        --output="${log_dir}/ml_%x_%j.out" \
        --error="${log_dir}/ml_%x_%j.err" \
        --export="ALL,NCI_DISTANCE_CUTOFF=${nci_cutoff},N_TRIALS=${N_TRIALS}" \
        --wckey=edu_class \
        "${SLURM_SCRIPT_KEFF}"

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Submitted${NC}"
    else
        echo -e "  ${RED}✗ Failed${NC}"
    fi

    if [ $SUBMITTED -lt $TOTAL ]; then
        echo -e "  ${YELLOW}Waiting ${DELAY_SECONDS}s...${NC}"
        sleep "${DELAY_SECONDS}"
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  FLUX JOBS — uncomment to re-enable
# ══════════════════════════════════════════════════════════════════════
# submit_flux_job 1 "NCI_cutoff" "total"       1
# submit_flux_job 1 "NCI_cutoff" "thermal"     4
# submit_flux_job 1 "NCI_cutoff" "epithermal"  5
# submit_flux_job 1 "NCI_cutoff" "fast"        6
# submit_flux_job 0 "NCI_no_cutoff" "total"       1
# submit_flux_job 0 "NCI_no_cutoff" "thermal"     4
# submit_flux_job 0 "NCI_no_cutoff" "epithermal"  5
# submit_flux_job 0 "NCI_no_cutoff" "fast"        6
# submit_flux_job 2 "No_NCI" "total"       1
# submit_flux_job 2 "No_NCI" "thermal"     4
# submit_flux_job 2 "No_NCI" "epithermal"  5
# submit_flux_job 2 "No_NCI" "fast"        6

# ══════════════════════════════════════════════════════════════════════
#  KEFF JOBS — one per NCI setting
# ══════════════════════════════════════════════════════════════════════
submit_keff_job 1 "NCI_cutoff"
submit_keff_job 0 "NCI_no_cutoff"
submit_keff_job 2 "No_NCI"

echo ""
echo "=========================================================================="
echo -e "${GREEN}All ${TOTAL} jobs submitted!  (keff only)${NC}"
echo "=========================================================================="
echo ""
echo "Log directories:"
echo "  ${BASE_LOG}/NCI_cutoff/"
echo "  ${BASE_LOG}/NCI_no_cutoff/"
echo "  ${BASE_LOG}/No_NCI/"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${BASE_LOG}/NCI_cutoff/ml_*.out"
