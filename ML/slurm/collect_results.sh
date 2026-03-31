#!/bin/bash
#
# Collect ML training results from all log directories.
#
# Terminal output: NCI comparison, then selected & random learning curves.
# CSV output: One combined file with selected + random learning curve results.
#             Set B uses cleaned re-eval logs only when BOTH learning_curve_setB_clean
#             and learning_curve_random_setB_clean exist (paired); otherwise both
#             use original ml_*.out Optuna logs (see FORCE_LC_ORIGINAL in script).
#
# Usage (run from ML_logs/):
#   bash ../slurm/collect_results.sh
#   bash ../slurm/collect_results.sh /path/to/ML_logs

BASE_LOG="${1:-$(pwd)}"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

if [ ! -d "$BASE_LOG" ]; then
    echo -e "${RED}Error: Log directory not found: ${BASE_LOG}${NC}"
    echo "Usage: $0 [path/to/ML_logs]"
    exit 1
fi

DIVIDER="=============================================================================================="
THIN_DIV="----------------------------------------------------------------------------------------------"

echo ""
echo "$DIVIDER"
echo -e "${BOLD}  ML Training Results Summary${NC}"
echo -e "  Log directory: ${CYAN}${BASE_LOG}${NC}"
echo -e "  Collected:     $(date)"
echo "$DIVIDER"

# ══════════════════════════════════════════════════════════════════════
#  Section 1: NCI Comparison (terminal only, no CSV)
# ══════════════════════════════════════════════════════════════════════

printf "\n${BOLD}%-20s %-14s %-10s %-10s %-10s %-10s %-8s${NC}\n" \
    "NCI Setting" "Target" "Test R²" "Test MAPE" "Test MSE" "Test RMSE" "Status"
echo "$THIN_DIV"

FOUND=0
COMPLETED=0
FAILED=0
RUNNING=0

for nci_dir in NCI_cutoff NCI_no_cutoff No_NCI; do
    dir="${BASE_LOG}/${nci_dir}"
    [ -d "$dir" ] || continue

    case "$nci_dir" in
        NCI_cutoff)    nci_label="NCI (cutoff)"    ;;
        NCI_no_cutoff) nci_label="NCI (no cutoff)"  ;;
        No_NCI)        nci_label="No NCI"            ;;
        *)             nci_label="$nci_dir"          ;;
    esac

    for outfile in "$dir"/ml_*.out; do
        [ -f "$outfile" ] || continue
        ((FOUND++))

        if grep -q "^Target:.*keff" "$outfile" 2>/dev/null; then
            flux_mode="keff"
        else
            flux_mode=$(grep -m1 "^Flux mode:" "$outfile" 2>/dev/null | sed 's/Flux mode:[[:space:]]*//' | awk '{print $1}')
            [ -z "$flux_mode" ] && flux_mode="unknown"
        fi

        test_r2=$(grep -m1 "Test R²:" "$outfile" 2>/dev/null | awk '{print $NF}')
        test_mape=$(grep -m1 "Test MAPE:" "$outfile" 2>/dev/null | awk '{print $NF}')
        test_mse=$(grep -m1 "Test MSE:" "$outfile" 2>/dev/null | awk '{print $NF}')
        test_rmse=$(grep -m1 "Test RMSE:" "$outfile" 2>/dev/null | awk '{print $NF}')

        if [ -n "$test_r2" ] && [ -n "$test_mape" ]; then
            ((COMPLETED++))
            status="${GREEN}done${NC}"
        elif grep -q "TRAINING COMPLETE" "$outfile" 2>/dev/null; then
            ((COMPLETED++))
            status="${GREEN}done${NC}"
        elif grep -q "Error during training\|Traceback (most recent\|FAILED\|RuntimeError\|MemoryError\|KeyError\|ValueError" "$outfile" 2>/dev/null; then
            ((FAILED++))
            status="${RED}FAIL${NC}"
        else
            ((RUNNING++))
            status="${YELLOW}run…${NC}"
        fi

        printf "%-20s %-14s %-10s %-10s %-10s %-10s ${status}\n" \
            "$nci_label" "$flux_mode" \
            "${test_r2:---}" "${test_mape:---}" \
            "${test_mse:---}" "${test_rmse:---}"
    done
done

echo "$THIN_DIV"
echo ""

if [ "$FOUND" -gt 0 ]; then
    echo -e "${BOLD}NCI Summary:${NC}  ${FOUND} log files"
    echo -e "  Completed: ${GREEN}${COMPLETED}${NC}   Running/Pending: ${YELLOW}${RUNNING}${NC}   Failed: ${RED}${FAILED}${NC}"

    if [ "$COMPLETED" -gt 0 ]; then
        echo ""
        echo "$DIVIDER"
        echo -e "${BOLD}  NCI Ranking by Target  (sorted by R², best → worst)${NC}"
        echo "$DIVIDER"

        MEDAL_1="${GREEN}1st${NC}"
        MEDAL_2="${CYAN}2nd${NC}"
        MEDAL_3="${YELLOW}3rd${NC}"

        print_ranking() {
            local target=$1
            local label=$2
            local tmpfile
            tmpfile=$(mktemp)

            for nci_dir in NCI_cutoff NCI_no_cutoff No_NCI; do
                dir="${BASE_LOG}/${nci_dir}"
                [ -d "$dir" ] || continue
                case "$nci_dir" in
                    NCI_cutoff)    nci_label="NCI (cutoff)"    ;;
                    NCI_no_cutoff) nci_label="NCI (no cutoff)" ;;
                    No_NCI)        nci_label="No NCI"          ;;
                    *)             nci_label="$nci_dir"        ;;
                esac
                for outfile in "$dir"/ml_*.out; do
                    [ -f "$outfile" ] || continue
                    if [ "$target" = "keff" ]; then
                        grep -q "^Target:.*keff" "$outfile" 2>/dev/null || continue
                    else
                        file_flux=$(grep -m1 "^Flux mode:" "$outfile" 2>/dev/null \
                            | sed 's/Flux mode:[[:space:]]*//' | awk '{print $1}')
                        [ "$file_flux" != "$target" ] && continue
                    fi
                    r2=$(grep -m1 "Test R²:" "$outfile" 2>/dev/null | awk '{print $NF}')
                    mape=$(grep -m1 "Test MAPE:" "$outfile" 2>/dev/null | awk '{print $NF}')
                    [ -z "$r2" ] && continue
                    echo "${r2}|${mape}|${nci_label}" >> "$tmpfile"
                    break
                done
            done

            local count
            count=$(wc -l < "$tmpfile" | tr -d ' ')
            if [ "$count" -eq 0 ]; then
                rm -f "$tmpfile"
                return
            fi
            echo ""
            echo -e "  ${BOLD}${label}${NC}"
            rank=0
            sort -t'|' -k1 -rn "$tmpfile" | while IFS='|' read -r r2 mape nci; do
                ((rank++))
                case $rank in
                    1) medal="$MEDAL_1" ;;
                    2) medal="$MEDAL_2" ;;
                    3) medal="$MEDAL_3" ;;
                esac
                printf "    ${medal}  %-20s  R²: %-10s  MAPE: %s\n" "$nci" "$r2" "$mape"
            done
            rm -f "$tmpfile"
        }

        print_ranking "total"       "TOTAL flux"
        print_ranking "thermal"     "THERMAL flux"
        print_ranking "epithermal"  "EPITHERMAL flux"
        print_ranking "fast"        "FAST flux"
        print_ranking "keff"        "K-EFF"
    fi
fi

echo ""
echo "$DIVIDER"
echo ""

# ══════════════════════════════════════════════════════════════════════
#  Section 2: Learning Curve Results (Selected + Random)
#
#  Scans *_setB_clean directories for cleaned Set B metrics.
#  Falls back to original directories if clean versions don't exist.
#  Outputs ONE combined CSV with an "experiment" column.
# ══════════════════════════════════════════════════════════════════════

CSV_FILE="${BASE_LOG}/learning_curve_combined_results.csv"

echo "experiment,n_geometries,n_runs,target,overall_mse,overall_r2,overall_mape,setA_mse,setA_r2,setA_mape,setB_mse,setB_r2,setB_mape,setA_4G_r2,setA_4G_mse,setA_4G_mape,setA_3G1P_r2,setA_3G1P_mse,setA_3G1P_mape,setA_3G1B_r2,setA_3G1B_mse,setA_3G1B_mape,setA_2G2P_r2,setA_2G2P_mse,setA_2G2P_mape,setA_2G2B_r2,setA_2G2B_mse,setA_2G2B_mape,setA_2G1P1B_r2,setA_2G1P1B_mse,setA_2G1P1B_mape,setB_4G_r2,setB_4G_mse,setB_4G_mape,setB_3G1P_r2,setB_3G1P_mse,setB_3G1P_mape,setB_3G1B_r2,setB_3G1B_mse,setB_3G1B_mape,setB_2G2P_r2,setB_2G2P_mse,setB_2G2P_mape,setB_2G2B_r2,setB_2G2B_mse,setB_2G2B_mape,setB_2G1P1B_r2,setB_2G1P1B_mse,setB_2G1P1B_mape,setC_mse,setC_r2,setC_mape,setC_4G_r2,setC_4G_mse,setC_4G_mape,setC_3G1P_r2,setC_3G1P_mse,setC_3G1P_mape,setC_3G1B_r2,setC_3G1B_mse,setC_3G1B_mape,setC_2G2P_r2,setC_2G2P_mse,setC_2G2P_mape,setC_2G2B_r2,setC_2G2B_mse,setC_2G2B_mape,setC_2G1P1B_r2,setC_2G1P1B_mse,setC_2G1P1B_mape,overall_mse_log,overall_r2_log,overall_mse_raw,overall_r2_raw,setA_mse_log,setA_r2_log,setA_mse_raw,setA_r2_raw,setB_mse_log,setB_r2_log,setB_mse_raw,setB_r2_raw,setC_mse_log,setC_r2_log,setC_mse_raw,setC_r2_raw,setA_4G_r2_raw,setA_4G_mse_raw,setA_3G1P_r2_raw,setA_3G1P_mse_raw,setA_3G1B_r2_raw,setA_3G1B_mse_raw,setA_2G2P_r2_raw,setA_2G2P_mse_raw,setA_2G2B_r2_raw,setA_2G2B_mse_raw,setA_2G1P1B_r2_raw,setA_2G1P1B_mse_raw,setB_4G_r2_raw,setB_4G_mse_raw,setB_3G1P_r2_raw,setB_3G1P_mse_raw,setB_3G1B_r2_raw,setB_3G1B_mse_raw,setB_2G2P_r2_raw,setB_2G2P_mse_raw,setB_2G2B_r2_raw,setB_2G2B_mse_raw,setB_2G1P1B_r2_raw,setB_2G1P1B_mse_raw,setC_4G_r2_raw,setC_4G_mse_raw,setC_3G1P_r2_raw,setC_3G1P_mse_raw,setC_3G1B_r2_raw,setC_3G1B_mse_raw,setC_2G2P_r2_raw,setC_2G2P_mse_raw,setC_2G2B_r2_raw,setC_2G2B_mse_raw,setC_2G1P1B_r2_raw,setC_2G1P1B_mse_raw,status" > "$CSV_FILE"

# ── Helper function: extract metrics from one .out file ──────────────
# Appends one row to $CSV_FILE and prints one terminal line.
# Args: $1=experiment_label $2=n_geo $3=target $4=outfile
collect_one_log() {
    local exp_label="$1" n_geo="$2" lc_target="$3" outfile="$4"
    parse_dual_value() {
        local line="$1" key="$2"
        echo "$line" | sed -n "s/.*${key}=\([0-9.eE+-]*\).*/\1/p"
    }
    parse_fill_raw_value() {
        local block="$1" fill_tag="$2" key="$3"
        local line
        line=$(echo "$block" | grep -m1 -E "^[[:space:]]*${fill_tag}:")
        parse_dual_value "$line" "$key"
    }

    ovr_r2=$(grep -m1 "Test R²:" "$outfile" 2>/dev/null | awk '{print $NF}')
    ovr_mape=$(grep -m1 "Test MAPE:" "$outfile" 2>/dev/null | awk '{print $NF}')
    ovr_mse=$(grep -m1 "Test MSE:" "$outfile" 2>/dev/null | awk '{print $NF}')

    # Set A
    setA_line=$(grep -A1 "Test Set A (random" "$outfile" 2>/dev/null | grep "MSE:" | head -1)
    setA_r2=$(echo "$setA_line" | sed 's/.*R²: \([0-9.-]*\).*/\1/')
    setA_mape=$(echo "$setA_line" | sed 's/.*MAPE: \([0-9.]*\)%.*/\1/')
    setA_mse=$(echo "$setA_line" | sed 's/.*MSE: \([0-9.]*\).*/\1/')
    [ "$setA_r2" = "$setA_line" ] && setA_r2=""
    [ "$setA_mape" = "$setA_line" ] && setA_mape=""
    [ "$setA_mse" = "$setA_line" ] && setA_mse=""

    # Set B (matches both "self-adjusted, CLEANED" and plain "self-adjusted")
    setB_line=$(grep -A1 "Test Set B (self" "$outfile" 2>/dev/null | grep "MSE:" | head -1)
    setB_r2=$(echo "$setB_line" | sed 's/.*R²: \([0-9.-]*\).*/\1/')
    setB_mape=$(echo "$setB_line" | sed 's/.*MAPE: \([0-9.]*\)%.*/\1/')
    setB_mse=$(echo "$setB_line" | sed 's/.*MSE: \([0-9.]*\).*/\1/')
    [ "$setB_r2" = "$setB_line" ] && setB_r2=""
    [ "$setB_mape" = "$setB_line" ] && setB_mape=""
    [ "$setB_mse" = "$setB_line" ] && setB_mse=""

    # Per-fill-type: Set A
    setA_ft_block=$(sed -n '/Per-fill-type breakdown.*Set A/,/Per-fill-type breakdown.*Set B\|^$/p' "$outfile" 2>/dev/null)
    setA_4G_r2=$(echo "$setA_ft_block" | grep '4G:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setA_4G_mse=$(echo "$setA_ft_block" | grep '4G:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setA_4G_mape=$(echo "$setA_ft_block" | grep '4G:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setA_3G1P_r2=$(echo "$setA_ft_block" | grep '3G1P:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setA_3G1P_mse=$(echo "$setA_ft_block" | grep '3G1P:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setA_3G1P_mape=$(echo "$setA_ft_block" | grep '3G1P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setA_3G1B_r2=$(echo "$setA_ft_block" | grep '3G1B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setA_3G1B_mse=$(echo "$setA_ft_block" | grep '3G1B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setA_3G1B_mape=$(echo "$setA_ft_block" | grep '3G1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setA_2G2P_r2=$(echo "$setA_ft_block" | grep '2G2P:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setA_2G2P_mse=$(echo "$setA_ft_block" | grep '2G2P:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setA_2G2P_mape=$(echo "$setA_ft_block" | grep '2G2P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setA_2G2B_r2=$(echo "$setA_ft_block" | grep '2G2B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setA_2G2B_mse=$(echo "$setA_ft_block" | grep '2G2B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setA_2G2B_mape=$(echo "$setA_ft_block" | grep '2G2B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setA_2G1P1B_r2=$(echo "$setA_ft_block" | grep '2G1P1B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setA_2G1P1B_mse=$(echo "$setA_ft_block" | grep '2G1P1B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setA_2G1P1B_mape=$(echo "$setA_ft_block" | grep '2G1P1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')

    # Per-fill-type: Set B
    setB_ft_block=$(sed -n '/Per-fill-type breakdown.*Set B/,/Per-fill-type breakdown.*Set C\|Evaluation complete\|^$/p' "$outfile" 2>/dev/null)
    setB_4G_r2=$(echo "$setB_ft_block" | grep '4G:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setB_4G_mse=$(echo "$setB_ft_block" | grep '4G:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setB_4G_mape=$(echo "$setB_ft_block" | grep '4G:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setB_3G1P_r2=$(echo "$setB_ft_block" | grep '3G1P:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setB_3G1P_mse=$(echo "$setB_ft_block" | grep '3G1P:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setB_3G1P_mape=$(echo "$setB_ft_block" | grep '3G1P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setB_3G1B_r2=$(echo "$setB_ft_block" | grep '3G1B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setB_3G1B_mse=$(echo "$setB_ft_block" | grep '3G1B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setB_3G1B_mape=$(echo "$setB_ft_block" | grep '3G1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setB_2G2P_r2=$(echo "$setB_ft_block" | grep '2G2P:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setB_2G2P_mse=$(echo "$setB_ft_block" | grep '2G2P:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setB_2G2P_mape=$(echo "$setB_ft_block" | grep '2G2P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setB_2G2B_r2=$(echo "$setB_ft_block" | grep '2G2B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setB_2G2B_mse=$(echo "$setB_ft_block" | grep '2G2B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setB_2G2B_mape=$(echo "$setB_ft_block" | grep '2G2B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setB_2G1P1B_r2=$(echo "$setB_ft_block" | grep '2G1P1B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setB_2G1P1B_mse=$(echo "$setB_ft_block" | grep '2G1P1B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setB_2G1P1B_mape=$(echo "$setB_ft_block" | grep '2G1P1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')

    # Set C
    setC_metrics_line=$(grep -A1 "Test Set C (random" "$outfile" 2>/dev/null | grep "MSE:" | head -1)
    setC_r2=$(echo "$setC_metrics_line" | sed 's/.*R²: \([0-9.-]*\).*/\1/')
    setC_mape=$(echo "$setC_metrics_line" | sed 's/.*MAPE: \([0-9.]*\)%.*/\1/')
    setC_mse=$(echo "$setC_metrics_line" | sed 's/.*MSE: \([0-9.]*\).*/\1/')
    [ "$setC_r2" = "$setC_metrics_line" ] && setC_r2=""
    [ "$setC_mape" = "$setC_metrics_line" ] && setC_mape=""
    [ "$setC_mse" = "$setC_metrics_line" ] && setC_mse=""

    # Per-fill-type: Set C
    setC_ft_block=$(sed -n '/Per-fill-type breakdown.*Set C/,/Evaluation complete\|^$/p' "$outfile" 2>/dev/null)
    setC_4G_r2=$(echo "$setC_ft_block" | grep '4G:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setC_4G_mse=$(echo "$setC_ft_block" | grep '4G:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setC_4G_mape=$(echo "$setC_ft_block" | grep '4G:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setC_3G1P_r2=$(echo "$setC_ft_block" | grep '3G1P:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setC_3G1P_mse=$(echo "$setC_ft_block" | grep '3G1P:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setC_3G1P_mape=$(echo "$setC_ft_block" | grep '3G1P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setC_3G1B_r2=$(echo "$setC_ft_block" | grep '3G1B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setC_3G1B_mse=$(echo "$setC_ft_block" | grep '3G1B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setC_3G1B_mape=$(echo "$setC_ft_block" | grep '3G1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setC_2G2P_r2=$(echo "$setC_ft_block" | grep '2G2P:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setC_2G2P_mse=$(echo "$setC_ft_block" | grep '2G2P:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setC_2G2P_mape=$(echo "$setC_ft_block" | grep '2G2P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setC_2G2B_r2=$(echo "$setC_ft_block" | grep '2G2B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setC_2G2B_mse=$(echo "$setC_ft_block" | grep '2G2B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setC_2G2B_mape=$(echo "$setC_ft_block" | grep '2G2B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
    setC_2G1P1B_r2=$(echo "$setC_ft_block" | grep '2G1P1B:' | grep -o 'R²=[0-9.-]*' | sed 's/R²=//')
    setC_2G1P1B_mse=$(echo "$setC_ft_block" | grep '2G1P1B:' | grep -o 'MSE=[0-9.]*' | sed 's/MSE=//')
    setC_2G1P1B_mape=$(echo "$setC_ft_block" | grep '2G1P1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')

    # Dual-space set-level metrics ([dual-space] log_mse=... log_r2=... raw_mse=... raw_r2=...)
    ovr_dual_line=$(grep -m1 "\[dual-space\]" "$outfile" 2>/dev/null)
    ovr_mse_log=$(parse_dual_value "$ovr_dual_line" "log_mse")
    ovr_r2_log=$(parse_dual_value "$ovr_dual_line" "log_r2")
    ovr_mse_raw=$(parse_dual_value "$ovr_dual_line" "raw_mse")
    ovr_r2_raw=$(parse_dual_value "$ovr_dual_line" "raw_r2")

    setA_dual_line=$(sed -n '/--- Test Set A (random lattice/,/--- Test Set B (self/p' "$outfile" 2>/dev/null | grep -m1 "\[dual-space\]")
    setA_mse_log=$(parse_dual_value "$setA_dual_line" "log_mse")
    setA_r2_log=$(parse_dual_value "$setA_dual_line" "log_r2")
    setA_mse_raw=$(parse_dual_value "$setA_dual_line" "raw_mse")
    setA_r2_raw=$(parse_dual_value "$setA_dual_line" "raw_r2")

    setB_dual_line=$(sed -n '/--- Test Set B (self/,/--- Test Set C (random cores\|Evaluation complete/p' "$outfile" 2>/dev/null | grep -m1 "\[dual-space\]")
    setB_mse_log=$(parse_dual_value "$setB_dual_line" "log_mse")
    setB_r2_log=$(parse_dual_value "$setB_dual_line" "log_r2")
    setB_mse_raw=$(parse_dual_value "$setB_dual_line" "raw_mse")
    setB_r2_raw=$(parse_dual_value "$setB_dual_line" "raw_r2")

    setC_dual_line=$(sed -n '/--- Test Set C (random cores/,/Evaluation complete/p' "$outfile" 2>/dev/null | grep -m1 "\[dual-space\]")
    setC_mse_log=$(parse_dual_value "$setC_dual_line" "log_mse")
    setC_r2_log=$(parse_dual_value "$setC_dual_line" "log_r2")
    setC_mse_raw=$(parse_dual_value "$setC_dual_line" "raw_mse")
    setC_r2_raw=$(parse_dual_value "$setC_dual_line" "raw_r2")

    # Per-fill-type raw-space metrics ([raw: raw_mse=... raw_r2=...])
    setA_4G_r2_raw=$(parse_fill_raw_value "$setA_ft_block" "4G" "raw_r2")
    setA_4G_mse_raw=$(parse_fill_raw_value "$setA_ft_block" "4G" "raw_mse")
    setA_3G1P_r2_raw=$(parse_fill_raw_value "$setA_ft_block" "3G1P" "raw_r2")
    setA_3G1P_mse_raw=$(parse_fill_raw_value "$setA_ft_block" "3G1P" "raw_mse")
    setA_3G1B_r2_raw=$(parse_fill_raw_value "$setA_ft_block" "3G1B" "raw_r2")
    setA_3G1B_mse_raw=$(parse_fill_raw_value "$setA_ft_block" "3G1B" "raw_mse")
    setA_2G2P_r2_raw=$(parse_fill_raw_value "$setA_ft_block" "2G2P" "raw_r2")
    setA_2G2P_mse_raw=$(parse_fill_raw_value "$setA_ft_block" "2G2P" "raw_mse")
    setA_2G2B_r2_raw=$(parse_fill_raw_value "$setA_ft_block" "2G2B" "raw_r2")
    setA_2G2B_mse_raw=$(parse_fill_raw_value "$setA_ft_block" "2G2B" "raw_mse")
    setA_2G1P1B_r2_raw=$(parse_fill_raw_value "$setA_ft_block" "2G1P1B" "raw_r2")
    setA_2G1P1B_mse_raw=$(parse_fill_raw_value "$setA_ft_block" "2G1P1B" "raw_mse")

    setB_4G_r2_raw=$(parse_fill_raw_value "$setB_ft_block" "4G" "raw_r2")
    setB_4G_mse_raw=$(parse_fill_raw_value "$setB_ft_block" "4G" "raw_mse")
    setB_3G1P_r2_raw=$(parse_fill_raw_value "$setB_ft_block" "3G1P" "raw_r2")
    setB_3G1P_mse_raw=$(parse_fill_raw_value "$setB_ft_block" "3G1P" "raw_mse")
    setB_3G1B_r2_raw=$(parse_fill_raw_value "$setB_ft_block" "3G1B" "raw_r2")
    setB_3G1B_mse_raw=$(parse_fill_raw_value "$setB_ft_block" "3G1B" "raw_mse")
    setB_2G2P_r2_raw=$(parse_fill_raw_value "$setB_ft_block" "2G2P" "raw_r2")
    setB_2G2P_mse_raw=$(parse_fill_raw_value "$setB_ft_block" "2G2P" "raw_mse")
    setB_2G2B_r2_raw=$(parse_fill_raw_value "$setB_ft_block" "2G2B" "raw_r2")
    setB_2G2B_mse_raw=$(parse_fill_raw_value "$setB_ft_block" "2G2B" "raw_mse")
    setB_2G1P1B_r2_raw=$(parse_fill_raw_value "$setB_ft_block" "2G1P1B" "raw_r2")
    setB_2G1P1B_mse_raw=$(parse_fill_raw_value "$setB_ft_block" "2G1P1B" "raw_mse")

    setC_4G_r2_raw=$(parse_fill_raw_value "$setC_ft_block" "4G" "raw_r2")
    setC_4G_mse_raw=$(parse_fill_raw_value "$setC_ft_block" "4G" "raw_mse")
    setC_3G1P_r2_raw=$(parse_fill_raw_value "$setC_ft_block" "3G1P" "raw_r2")
    setC_3G1P_mse_raw=$(parse_fill_raw_value "$setC_ft_block" "3G1P" "raw_mse")
    setC_3G1B_r2_raw=$(parse_fill_raw_value "$setC_ft_block" "3G1B" "raw_r2")
    setC_3G1B_mse_raw=$(parse_fill_raw_value "$setC_ft_block" "3G1B" "raw_mse")
    setC_2G2P_r2_raw=$(parse_fill_raw_value "$setC_ft_block" "2G2P" "raw_r2")
    setC_2G2P_mse_raw=$(parse_fill_raw_value "$setC_ft_block" "2G2P" "raw_mse")
    setC_2G2B_r2_raw=$(parse_fill_raw_value "$setC_ft_block" "2G2B" "raw_r2")
    setC_2G2B_mse_raw=$(parse_fill_raw_value "$setC_ft_block" "2G2B" "raw_mse")
    setC_2G1P1B_r2_raw=$(parse_fill_raw_value "$setC_ft_block" "2G1P1B" "raw_r2")
    setC_2G1P1B_mse_raw=$(parse_fill_raw_value "$setC_ft_block" "2G1P1B" "raw_mse")

    # Keff: raw/log identical, backfill from base metrics if dual lines absent
    if [ "$lc_target" = "keff" ]; then
        [ -z "$ovr_mse_log" ] && ovr_mse_log="$ovr_mse";   [ -z "$ovr_r2_log" ] && ovr_r2_log="$ovr_r2"
        [ -z "$ovr_mse_raw" ] && ovr_mse_raw="$ovr_mse";   [ -z "$ovr_r2_raw" ] && ovr_r2_raw="$ovr_r2"
        [ -z "$setA_mse_log" ] && setA_mse_log="$setA_mse"; [ -z "$setA_r2_log" ] && setA_r2_log="$setA_r2"
        [ -z "$setA_mse_raw" ] && setA_mse_raw="$setA_mse"; [ -z "$setA_r2_raw" ] && setA_r2_raw="$setA_r2"
        [ -z "$setB_mse_log" ] && setB_mse_log="$setB_mse"; [ -z "$setB_r2_log" ] && setB_r2_log="$setB_r2"
        [ -z "$setB_mse_raw" ] && setB_mse_raw="$setB_mse"; [ -z "$setB_r2_raw" ] && setB_r2_raw="$setB_r2"
        [ -z "$setC_mse_log" ] && setC_mse_log="$setC_mse"; [ -z "$setC_r2_log" ] && setC_r2_log="$setC_r2"
        [ -z "$setC_mse_raw" ] && setC_mse_raw="$setC_mse"; [ -z "$setC_r2_raw" ] && setC_r2_raw="$setC_r2"
        for set_prefix in setA setB setC; do
            for fill_tag in 4G 3G1P 3G1B 2G2P 2G2B 2G1P1B; do
                eval "[ -z \"\${${set_prefix}_${fill_tag}_mse_raw}\" ] && ${set_prefix}_${fill_tag}_mse_raw=\"\${${set_prefix}_${fill_tag}_mse}\""
                eval "[ -z \"\${${set_prefix}_${fill_tag}_r2_raw}\" ] && ${set_prefix}_${fill_tag}_r2_raw=\"\${${set_prefix}_${fill_tag}_r2}\""
            done
        done
    fi

    # Status
    if [ -n "$ovr_r2" ]; then
        lc_status="done"
        status_color="${GREEN}done${NC}"
    elif grep -q "Error\|Traceback\|FAILED\|RuntimeError\|MemoryError\|KeyError\|ValueError" "$outfile" 2>/dev/null; then
        lc_status="FAIL"
        status_color="${RED}FAIL${NC}"
    else
        lc_status="run"
        status_color="${YELLOW}run…${NC}"
    fi

    # Terminal line
    printf "%-10s %-8s %-12s %-12s %-14s %-14s | %-10s %-12s | %-10s %-12s | %-10s %-12s ${status_color}\n" \
        "$exp_label" "$n_geo" "$lc_target" \
        "${ovr_r2:---}" "${ovr_mape:---}" "${ovr_mse:---}" \
        "${setA_r2:---}" "${setA_mape:---}" \
        "${setB_r2:---}" "${setB_mape:---}" \
        "${setC_r2:---}" "${setC_mape:---}"

    # CSV row
    csv_row="${exp_label},${n_geo},$((n_geo * 33)),${lc_target},${ovr_mse:---},${ovr_r2:---},${ovr_mape:---},${setA_mse:---},${setA_r2:---},${setA_mape:---},${setB_mse:---},${setB_r2:---},${setB_mape:---},${setA_4G_r2:---},${setA_4G_mse:---},${setA_4G_mape:---},${setA_3G1P_r2:---},${setA_3G1P_mse:---},${setA_3G1P_mape:---},${setA_3G1B_r2:---},${setA_3G1B_mse:---},${setA_3G1B_mape:---},${setA_2G2P_r2:---},${setA_2G2P_mse:---},${setA_2G2P_mape:---},${setA_2G2B_r2:---},${setA_2G2B_mse:---},${setA_2G2B_mape:---},${setA_2G1P1B_r2:---},${setA_2G1P1B_mse:---},${setA_2G1P1B_mape:---},${setB_4G_r2:---},${setB_4G_mse:---},${setB_4G_mape:---},${setB_3G1P_r2:---},${setB_3G1P_mse:---},${setB_3G1P_mape:---},${setB_3G1B_r2:---},${setB_3G1B_mse:---},${setB_3G1B_mape:---},${setB_2G2P_r2:---},${setB_2G2P_mse:---},${setB_2G2P_mape:---},${setB_2G2B_r2:---},${setB_2G2B_mse:---},${setB_2G2B_mape:---},${setB_2G1P1B_r2:---},${setB_2G1P1B_mse:---},${setB_2G1P1B_mape:---},${setC_mse:---},${setC_r2:---},${setC_mape:---},${setC_4G_r2:---},${setC_4G_mse:---},${setC_4G_mape:---},${setC_3G1P_r2:---},${setC_3G1P_mse:---},${setC_3G1P_mape:---},${setC_3G1B_r2:---},${setC_3G1B_mse:---},${setC_3G1B_mape:---},${setC_2G2P_r2:---},${setC_2G2P_mse:---},${setC_2G2P_mape:---},${setC_2G2B_r2:---},${setC_2G2B_mse:---},${setC_2G2B_mape:---},${setC_2G1P1B_r2:---},${setC_2G1P1B_mse:---},${setC_2G1P1B_mape:---}"
    csv_row="${csv_row},${ovr_mse_log:---},${ovr_r2_log:---},${ovr_mse_raw:---},${ovr_r2_raw:---},${setA_mse_log:---},${setA_r2_log:---},${setA_mse_raw:---},${setA_r2_raw:---},${setB_mse_log:---},${setB_r2_log:---},${setB_mse_raw:---},${setB_r2_raw:---},${setC_mse_log:---},${setC_r2_log:---},${setC_mse_raw:---},${setC_r2_raw:---}"
    csv_row="${csv_row},${setA_4G_r2_raw:---},${setA_4G_mse_raw:---},${setA_3G1P_r2_raw:---},${setA_3G1P_mse_raw:---},${setA_3G1B_r2_raw:---},${setA_3G1B_mse_raw:---},${setA_2G2P_r2_raw:---},${setA_2G2P_mse_raw:---},${setA_2G2B_r2_raw:---},${setA_2G2B_mse_raw:---},${setA_2G1P1B_r2_raw:---},${setA_2G1P1B_mse_raw:---},${setB_4G_r2_raw:---},${setB_4G_mse_raw:---},${setB_3G1P_r2_raw:---},${setB_3G1P_mse_raw:---},${setB_3G1B_r2_raw:---},${setB_3G1B_mse_raw:---},${setB_2G2P_r2_raw:---},${setB_2G2P_mse_raw:---},${setB_2G2B_r2_raw:---},${setB_2G2B_mse_raw:---},${setB_2G1P1B_r2_raw:---},${setB_2G1P1B_mse_raw:---},${setC_4G_r2_raw:---},${setC_4G_mse_raw:---},${setC_3G1P_r2_raw:---},${setC_3G1P_mse_raw:---},${setC_3G1B_r2_raw:---},${setC_3G1B_mse_raw:---},${setC_2G2P_r2_raw:---},${setC_2G2P_mse_raw:---},${setC_2G2B_r2_raw:---},${setC_2G2B_mse_raw:---},${setC_2G1P1B_r2_raw:---},${setC_2G1P1B_mse_raw:---}"
    csv_row="${csv_row},${lc_status}"
    echo "$csv_row" >> "$CSV_FILE"
}

# ── Scan each experiment ─────────────────────────────────────────────
# Prefer *_setB_clean directories (cleaned B); fall back to originals.
# Format: "label  clean_dir  fallback_dir  clean_glob  fallback_glob"
EXPERIMENTS=(
    "selected    learning_curve_setB_clean          learning_curve          setB_*.out  ml_*.out"
    "random      learning_curve_random_setB_clean   learning_curve_random   setB_*.out  ml_*.out"
)

# BUGFIX (paired sources): setB_*.out logs come from evaluate_setB_clean.py (retrain from
# extracted params), while ml_*.out logs are the original Optuna training run.
# If we use setB_clean for "selected" but fall back to ml_*.out for "random" (because
# learning_curve_random_setB_clean is missing), the CSV compares INCOMENSURATE metrics
# for Set A, B, and C — not just Set B. Only use clean dirs when BOTH exist, unless
# FORCE_LC_ORIGINAL=1 forces original ml_*.out for both.
CLEAN_SELECTED="${BASE_LOG}/learning_curve_setB_clean"
CLEAN_RANDOM="${BASE_LOG}/learning_curve_random_setB_clean"
USE_CLEAN_PAIRED=0
if [ "${FORCE_LC_ORIGINAL:-0}" != "1" ] \
    && [ -d "$CLEAN_SELECTED" ] && [ -d "$CLEAN_RANDOM" ]; then
    USE_CLEAN_PAIRED=1
fi

if [ "$USE_CLEAN_PAIRED" -eq 1 ]; then
    echo ""
    echo -e "${CYAN}Learning curve CSV:${NC} using paired ${BOLD}setB_*.out${NC} re-eval logs (both clean dirs present)."
elif [ "${FORCE_LC_ORIGINAL:-0}" = "1" ]; then
    echo ""
    echo -e "${CYAN}Learning curve CSV:${NC} ${BOLD}FORCE_LC_ORIGINAL=1${NC} — using ${BOLD}ml_*.out${NC} Optuna logs for both experiments."
elif [ -d "$CLEAN_SELECTED" ] || [ -d "$CLEAN_RANDOM" ]; then
    echo ""
    echo -e "${YELLOW}Learning curve CSV:${NC} only one of learning_curve_*_setB_clean exists — using ${BOLD}ml_*.out${NC} for ${BOLD}both${NC} selected & random (avoids mixing re-eval vs Optuna). Create both clean dirs or set FORCE_LC_ORIGINAL=1."
fi

TOTAL_LC_FOUND=0
TOTAL_LC_DONE=0

for exp_spec in "${EXPERIMENTS[@]}"; do
    read -r exp_label clean_dir fallback_dir clean_glob fallback_glob <<< "$exp_spec"

    # Decide which directory to use (must be paired for apples-to-apples CSV)
    if [ "$USE_CLEAN_PAIRED" -eq 1 ] && [ -d "${BASE_LOG}/${clean_dir}" ]; then
        LC_DIR="${BASE_LOG}/${clean_dir}"
        OUT_GLOB="$clean_glob"
        b_label="Set B (cleaned, paired)"
    elif [ -d "${BASE_LOG}/${fallback_dir}" ]; then
        LC_DIR="${BASE_LOG}/${fallback_dir}"
        OUT_GLOB="$fallback_glob"
        if [ -d "$CLEAN_SELECTED" ] || [ -d "$CLEAN_RANDOM" ]; then
            b_label="Set B (original Optuna, paired)"
        else
            b_label="Set B (original)"
        fi
    else
        continue
    fi

    echo ""
    echo "$DIVIDER"
    echo -e "${BOLD}  Learning Curve: ${exp_label} — ${b_label}${NC}"
    echo -e "  Directory: ${CYAN}${LC_DIR}${NC}"
    echo "$DIVIDER"

    printf "\n${BOLD}%-10s %-8s %-12s %-12s %-14s %-14s | %-10s %-12s | %-10s %-12s | %-10s %-12s %-8s${NC}\n" \
        "Exp" "N_GEO" "Target" "Ovr R²" "Ovr MAPE" "Ovr MSE" "SetA R²" "SetA MAPE" "SetB R²" "SetB MAPE" "SetC R²" "SetC MAPE" "Status"
    echo "$THIN_DIV"

    EXP_FOUND=0
    EXP_DONE=0

    for lc_subdir in "$LC_DIR"/N*; do
        [ -d "$lc_subdir" ] || continue
        dirname=$(basename "$lc_subdir")

        n_geo=$(echo "$dirname" | sed 's/^N0*//' | cut -d'_' -f1)
        lc_target=$(echo "$dirname" | cut -d'_' -f2-)

        for outfile in "$lc_subdir"/${OUT_GLOB}; do
            [ -f "$outfile" ] || continue
            ((EXP_FOUND++))
            ((TOTAL_LC_FOUND++))

            collect_one_log "$exp_label" "$n_geo" "$lc_target" "$outfile"

            # Count completed
            if [ -n "$(grep -m1 'Test R²:' "$outfile" 2>/dev/null)" ]; then
                ((EXP_DONE++))
                ((TOTAL_LC_DONE++))
            fi
        done
    done

    echo "$THIN_DIV"
    echo ""
    echo -e "${BOLD}${exp_label} summary:${NC}  ${EXP_FOUND} log files, ${EXP_DONE} completed"
done

echo ""
echo "$DIVIDER"
echo -e "${BOLD}Learning Curve Total:${NC}  ${TOTAL_LC_FOUND} log files, ${TOTAL_LC_DONE} completed"
echo -e "Combined CSV: ${CYAN}${CSV_FILE}${NC}"
echo "$DIVIDER"
echo ""
