#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Submit a single thesis NN + Ray Tune HPO job.
#
# Usage:
#   ./submit_nn_thesis.sh <NN_CONFIG> [FLUX_GROUP] [NCI_CUTOFF] [N_TRIALS]
#
# Examples:
#   ./submit_nn_thesis.sh 1                # Config 1: single net, 16 outputs
#   ./submit_nn_thesis.sh 4                # Config 4: restructured, single net
#   ./submit_nn_thesis.sh 2 0              # Config 2, flux group 0 (total)
#   ./submit_nn_thesis.sh 2 1              # Config 2, flux group 1 (thermal)
#   ./submit_nn_thesis.sh 3 2              # Config 3, flux group 2 (epithermal)
#   ./submit_nn_thesis.sh 5 3              # Config 5, flux group 3 (fast)
#   ./submit_nn_thesis.sh 2 0 1 200        # Config 2, group 0, NCI cutoff=1, 200 trials
#
# For multi-network configs (2, 3, 5), run this script once per flux group:
#   ./submit_nn_thesis.sh 2 0   # total
#   ./submit_nn_thesis.sh 2 1   # thermal
#   ./submit_nn_thesis.sh 2 2   # epithermal
#   ./submit_nn_thesis.sh 2 3   # fast
#
# After all 4 finish, assemble the composite model manually:
#   cd ~/IntegratedReactorModel/ML/
#   python assemble_composite.py --nn_config 2 --model_dir execution/models
#
# Optional environment variables:
#   TRAIN_FILE     Training data  (default: data/train.txt)
#   N_GPUS         GPUs per job   (default: 3, must match --gres in cpu_nn_thesis.sh)
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

NN_CONFIG=${1:?Usage: $0 <NN_CONFIG 1-5> [FLUX_GROUP 0-3] [NCI_CUTOFF] [N_TRIALS]}
FLUX_GROUP=${2:-}
NCI_CUTOFF=${3:-1}
N_TRIALS=${4:-100}
TRAIN_FILE=${TRAIN_FILE:-data/train.txt}
N_GPUS=${N_GPUS:-3}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/cpu_nn_thesis.sh"

BASE_LOG="$HOME/IntegratedReactorModel/ML/logs"
mkdir -p "${BASE_LOG}"

FLUX_NAMES=("total" "thermal" "epithermal" "fast")

# ── Validation ──────────────────────────────────────────────────────
if [[ ! "$NN_CONFIG" =~ ^[1-5]$ ]]; then
    echo -e "${RED}ERROR: NN_CONFIG must be 1-5, got: ${NN_CONFIG}${NC}"
    exit 1
fi

if [[ "$NN_CONFIG" =~ ^[235]$ ]]; then
    if [ -z "$FLUX_GROUP" ]; then
        echo -e "${RED}ERROR: Config ${NN_CONFIG} requires FLUX_GROUP (0-3)${NC}"
        echo ""
        echo "  0 = total flux"
        echo "  1 = thermal flux"
        echo "  2 = epithermal flux"
        echo "  3 = fast flux"
        echo ""
        echo "Usage: $0 ${NN_CONFIG} <FLUX_GROUP>"
        exit 1
    fi
    if [[ ! "$FLUX_GROUP" =~ ^[0-3]$ ]]; then
        echo -e "${RED}ERROR: FLUX_GROUP must be 0-3, got: ${FLUX_GROUP}${NC}"
        exit 1
    fi
fi

# ── Build job name ──────────────────────────────────────────────────
if [ -n "$FLUX_GROUP" ]; then
    JOB_NAME="nn${NN_CONFIG}_group${FLUX_GROUP}_${FLUX_NAMES[$FLUX_GROUP]}_nci${NCI_CUTOFF}"
else
    JOB_NAME="nn${NN_CONFIG}_full_nci${NCI_CUTOFF}"
fi

# ── Display ─────────────────────────────────────────────────────────
echo "=========================================================================="
echo -e "${BLUE} Submitting: ${JOB_NAME}${NC}"
echo "=========================================================================="
echo "  NN_CONFIG  : ${NN_CONFIG}"
if [ -n "$FLUX_GROUP" ]; then
    echo "  FLUX_GROUP : ${FLUX_GROUP} (${FLUX_NAMES[$FLUX_GROUP]})"
fi
echo "  NCI_CUTOFF : ${NCI_CUTOFF}"
echo "  N_TRIALS   : ${N_TRIALS}"
echo "  N_GPUS     : ${N_GPUS}"
echo "  TRAIN_FILE : ${TRAIN_FILE}"
echo "=========================================================================="

# ── Build export string ─────────────────────────────────────────────
EXPORT_VARS="ALL,NN_CONFIG=${NN_CONFIG},NCI_DISTANCE_CUTOFF=${NCI_CUTOFF}"
EXPORT_VARS+=",N_TRIALS=${N_TRIALS},TRAIN_FILE=${TRAIN_FILE},N_GPUS=${N_GPUS}"
if [ -n "$FLUX_GROUP" ]; then
    EXPORT_VARS+=",FLUX_GROUP=${FLUX_GROUP}"
fi

# ── Submit ──────────────────────────────────────────────────────────
sbatch \
    --job-name="${JOB_NAME}" \
    --output="${BASE_LOG}/ml_${JOB_NAME}_%j.out" \
    --error="${BASE_LOG}/ml_${JOB_NAME}_%j.err" \
    --export="${EXPORT_VARS}" \
    "${TRAIN_SCRIPT}"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Job submitted.${NC}"
else
    echo -e "${RED}Submission failed.${NC}"
    exit 1
fi

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${BASE_LOG}/ml_${JOB_NAME}_*.out"

if [[ "$NN_CONFIG" =~ ^[235]$ ]]; then
    echo ""
    echo "After all 4 flux groups finish, assemble the composite:"
    echo "  cd ~/IntegratedReactorModel/ML/"
    echo "  python assemble_composite.py --nn_config ${NN_CONFIG} --model_dir execution/models"
fi
