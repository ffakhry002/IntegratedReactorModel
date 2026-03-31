#!/bin/bash
#
# Submit Set B clean re-evaluation for all learning curve models.
#
# Iterates over both learning_curve (selected) and learning_curve_random
# experiments, extracts best params from each .out file, retrains, and
# evaluates on Test Set B with leaked geometries removed.
#
# Output goes to ML_logs/learning_curve_setB_clean/ and
# ML_logs/learning_curve_random_setB_clean/.

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

DELAY_SECONDS=${DELAY_SECONDS:-1}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/cpu_setB_eval.sh"

GEOMETRY_STEPS=(13 26 39 52 65 78 91 104 117 130 143 156 169 182 195 208 221 234 247 256)

TARGETS=(
    "flux total 1"
    "flux thermal_only 4"
    "flux epithermal_only 5"
    "flux fast_only 6"
    "keff keff 0"
)

ML_LOGS="$HOME/IntegratedReactorModel/ML/ML_logs"

# ── Which experiments to run ─────────────────────────────────────────
# Format: "source_dir  output_suffix  train_mode"
#   train_mode = "random" (use train_lc_set_N.txt) or "selected" (use train.txt + max_runs)
EXPERIMENTS=(
    "learning_curve          learning_curve_setB_clean          selected"
    "learning_curve_random   learning_curve_random_setB_clean   random"
)

TOTAL=0
SUBMITTED=0

# Count total jobs
for exp_spec in "${EXPERIMENTS[@]}"; do
    read -r src_dir out_suffix train_mode <<< "$exp_spec"
    src_path="${ML_LOGS}/${src_dir}"
    [ -d "$src_path" ] || continue
    for n_geo in "${GEOMETRY_STEPS[@]}"; do
        for target_spec in "${TARGETS[@]}"; do
            read -r target_type flux_name flux_choice <<< "$target_spec"
            if [ "$target_type" = "keff" ]; then label="keff"; else label="${flux_name}"; fi
            subdir="${src_path}/N$(printf '%03d' $n_geo)_${label}"
            [ -d "$subdir" ] && ((TOTAL++))
        done
    done
done

echo "=========================================================================="
echo " Set B Clean Re-Evaluation"
echo "=========================================================================="
echo ""
echo " Experiments: ${#EXPERIMENTS[@]}"
echo " Geometry steps: ${#GEOMETRY_STEPS[@]}"
echo " Targets: 5 (total, thermal, epithermal, fast, keff)"
echo " Total jobs: $TOTAL"
echo " Delay: ${DELAY_SECONDS}s"
echo ""
echo "=========================================================================="

for exp_spec in "${EXPERIMENTS[@]}"; do
    read -r src_dir out_suffix train_mode <<< "$exp_spec"
    src_path="${ML_LOGS}/${src_dir}"

    if [ ! -d "$src_path" ]; then
        echo -e "\n${YELLOW}Skipping ${src_dir} (directory not found)${NC}"
        continue
    fi

    out_base="${ML_LOGS}/${out_suffix}"
    mkdir -p "${out_base}/summary"

    echo ""
    echo -e "${BLUE}━━━ Experiment: ${src_dir} → ${out_suffix} (${train_mode}) ━━━${NC}"

    for n_geo in "${GEOMETRY_STEPS[@]}"; do
        for target_spec in "${TARGETS[@]}"; do
            read -r target_type flux_name flux_choice <<< "$target_spec"

            if [ "$target_type" = "keff" ]; then
                label="keff"
            else
                label="${flux_name}"
            fi

            log_subdir="N$(printf '%03d' $n_geo)_${label}"
            src_log_dir="${src_path}/${log_subdir}"

            if [ ! -d "$src_log_dir" ]; then
                continue
            fi

            # Check that the source directory has a .out file
            out_file=$(ls "$src_log_dir"/ml_*.out 2>/dev/null | head -1)
            if [ -z "$out_file" ]; then
                echo -e "  ${YELLOW}[SKIP] ${log_subdir}: no .out file${NC}"
                continue
            fi

            # Determine training data
            if [ "$train_mode" = "random" ]; then
                train_file="data/train_lc_set_${n_geo}.txt"
                max_runs_arg=""
            else
                train_file="data/train.txt"
                max_runs_arg="$((n_geo * 33))"
            fi

            out_log_dir="${out_base}/${log_subdir}"
            mkdir -p "$out_log_dir"
            job_name="setB_${out_suffix}_N${n_geo}_${label}"

            ((SUBMITTED++))

            echo -e "  ${BLUE}[${SUBMITTED}/${TOTAL}] ${job_name}${NC}"
            echo "    Source: ${src_log_dir}"
            echo "    Train:  ${train_file}${max_runs_arg:+ (max_runs=$max_runs_arg)}"

            export_vars="ALL,TARGET=${target_type},FLUX_MODE=${flux_name},LOG_DIR=${src_log_dir},TRAIN_FILE=${train_file},NCI_DISTANCE_CUTOFF=0,TEST_C_FILE=${TEST_C_FILE:-data/test_c.txt}"
            if [ -n "$max_runs_arg" ]; then
                export_vars="${export_vars},MAX_RUNS=${max_runs_arg}"
            fi

            sbatch \
                --job-name="${job_name}" \
                --output="${out_log_dir}/setB_%x_%j.out" \
                --error="${out_log_dir}/setB_%x_%j.err" \
                --export="${export_vars}" \
                --wckey=edu_class \
                "${SLURM_SCRIPT}"

            if [ $? -eq 0 ]; then
                echo -e "    ${GREEN}✓ Submitted${NC}"
            else
                echo -e "    ${RED}✗ Failed${NC}"
            fi

            if [ $SUBMITTED -lt $TOTAL ]; then
                sleep "${DELAY_SECONDS}"
            fi
        done
    done
done

echo ""
echo "=========================================================================="
echo -e "${GREEN}Submitted ${SUBMITTED}/${TOTAL} jobs${NC}"
echo "=========================================================================="
echo ""
echo "Output directories:"
for exp_spec in "${EXPERIMENTS[@]}"; do
    read -r _ out_suffix _ <<< "$exp_spec"
    echo "  ${ML_LOGS}/${out_suffix}/"
done
echo ""
echo "After completion, collect results with:"
echo "  bash slurm/collect_results.sh"
echo "  (Set B clean results will be in the *_setB_clean directories)"
