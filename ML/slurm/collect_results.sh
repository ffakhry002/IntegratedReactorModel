#!/bin/bash
#
# Collect MAPE and R² results from all ML training log files.
# Run from anywhere — no arguments needed (uses default log path),
# or pass a custom log directory as the first argument.
#
# Usage:
#   bash collect_results.sh
#   bash collect_results.sh /path/to/ML_logs

BASE_LOG="${1:-$HOME/IntegratedReactorModel/ML/ML_logs}"

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
        elif grep -q "Error\|Traceback\|FAILED\|error" "$outfile" 2>/dev/null; then
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

if [ "$FOUND" -eq 0 ]; then
    echo -e "${YELLOW}No .out files found in ${BASE_LOG}/*/${NC}"
    echo "Jobs may not have started yet.  Check with:  squeue -u \$USER"
    exit 0
fi

echo -e "${BOLD}Summary:${NC}  ${FOUND} log files found"
echo -e "  Completed: ${GREEN}${COMPLETED}${NC}   Running/Pending: ${YELLOW}${RUNNING}${NC}   Failed: ${RED}${FAILED}${NC}"

if [ "$COMPLETED" -gt 0 ]; then
    echo ""
    echo "$DIVIDER"
    echo -e "${BOLD}  NCI Ranking by Target  (sorted by R², best → worst)${NC}"
    echo "$DIVIDER"

    MEDAL_1="${GREEN}1st${NC}"
    MEDAL_2="${CYAN}2nd${NC}"
    MEDAL_3="${YELLOW}3rd${NC}"

    # Helper: given a target name, check if log matches it
    # For flux targets, match via "Flux mode:" header line
    # For keff, match via "Target:     keff" header line
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

echo ""
echo "$DIVIDER"
echo ""

# ══════════════════════════════════════════════════════════════════════
#  Learning Curve Results Collection
# ══════════════════════════════════════════════════════════════════════
LC_DIR="${BASE_LOG}/learning_curve"

if [ -d "$LC_DIR" ]; then
    echo ""
    echo "$DIVIDER"
    echo -e "${BOLD}  Learning Curve Results${NC}"
    echo -e "  Directory: ${CYAN}${LC_DIR}${NC}"
    echo "$DIVIDER"

    # CSV output file
    CSV_FILE="${LC_DIR}/summary/learning_curve_results.csv"
    mkdir -p "${LC_DIR}/summary"

    # Write CSV header
    echo "n_geometries,n_runs,target,overall_mse,overall_r2,overall_mape,setA_mse,setA_r2,setA_mape,setB_mse,setB_r2,setB_mape,setA_4G_mape,setA_3G1P_mape,setA_3G1B_mape,setA_2G2P_mape,setA_2G2B_mape,setA_2G1P1B_mape,setB_4G_mape,setB_3G1P_mape,setB_3G1B_mape,setB_2G2P_mape,setB_2G2B_mape,setB_2G1P1B_mape,setC_mse,setC_r2,setC_mape,setC_4G_mape,setC_3G1P_mape,setC_3G1B_mape,setC_2G2P_mape,setC_2G2B_mape,setC_2G1P1B_mape,status" > "$CSV_FILE"

    printf "\n${BOLD}%-8s %-12s %-10s %-10s %-10s | %-10s %-10s | %-10s %-10s | %-10s %-10s %-8s${NC}\n" \
        "N_GEO" "Target" "Ovr R²" "Ovr MAPE" "Ovr MSE" "SetA R²" "SetA MAPE" "SetB R²" "SetB MAPE" "SetC R²" "SetC MAPE" "Status"
    echo "$THIN_DIV"

    LC_FOUND=0
    LC_DONE=0

    for lc_subdir in "$LC_DIR"/N*; do
        [ -d "$lc_subdir" ] || continue
        dirname=$(basename "$lc_subdir")

        # Parse N_GEOMETRIES and target from directory name (e.g., N013_total)
        n_geo=$(echo "$dirname" | sed 's/^N0*//' | cut -d'_' -f1)
        lc_target=$(echo "$dirname" | cut -d'_' -f2-)

        for outfile in "$lc_subdir"/ml_*.out; do
            [ -f "$outfile" ] || continue
            ((LC_FOUND++))

            # Extract overall metrics
            ovr_r2=$(grep -m1 "Test R²:" "$outfile" 2>/dev/null | awk '{print $NF}')
            ovr_mape=$(grep -m1 "Test MAPE:" "$outfile" 2>/dev/null | awk '{print $NF}')
            ovr_mse=$(grep -m1 "Test MSE:" "$outfile" 2>/dev/null | awk '{print $NF}')

            # Extract Set A metrics
            setA_block=$(grep -A3 "Test Set A" "$outfile" 2>/dev/null | head -4)
            setA_r2=$(echo "$setA_block" | grep -o 'R²: [0-9.-]*' | awk '{print $2}')
            setA_mape=$(echo "$setA_block" | grep -o 'MAPE: [0-9.]*%' | sed 's/MAPE: //;s/%//')
            setA_mse=$(echo "$setA_block" | grep -o 'MSE: [0-9.]*' | head -1 | awk '{print $2}')

            # Extract Set B metrics
            setB_block=$(grep -A3 "Test Set B" "$outfile" 2>/dev/null | head -4)
            setB_r2=$(echo "$setB_block" | grep -o 'R²: [0-9.-]*' | awk '{print $2}')
            setB_mape=$(echo "$setB_block" | grep -o 'MAPE: [0-9.]*%' | sed 's/MAPE: //;s/%//')
            setB_mse=$(echo "$setB_block" | grep -o 'MSE: [0-9.]*' | head -1 | awk '{print $2}')

            # Extract per-fill-type MAPE for Set A
            setA_ft_block=$(sed -n '/Per-fill-type breakdown for Set A/,/Per-fill-type breakdown for Set B\|^$/p' "$outfile" 2>/dev/null)
            setA_4G=$(echo "$setA_ft_block" | grep '4G:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setA_3G1P=$(echo "$setA_ft_block" | grep '3G1P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setA_3G1B=$(echo "$setA_ft_block" | grep '3G1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setA_2G2P=$(echo "$setA_ft_block" | grep '2G2P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setA_2G2B=$(echo "$setA_ft_block" | grep '2G2B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setA_2G1P1B=$(echo "$setA_ft_block" | grep '2G1P1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')

            # Extract per-fill-type MAPE for Set B
            setB_ft_block=$(sed -n '/Per-fill-type breakdown for Set B/,/^$/p' "$outfile" 2>/dev/null)
            setB_4G=$(echo "$setB_ft_block" | grep '4G:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setB_3G1P=$(echo "$setB_ft_block" | grep '3G1P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setB_3G1B=$(echo "$setB_ft_block" | grep '3G1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setB_2G2P=$(echo "$setB_ft_block" | grep '2G2P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setB_2G2B=$(echo "$setB_ft_block" | grep '2G2B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setB_2G1P1B=$(echo "$setB_ft_block" | grep '2G1P1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')

            # Extract Set C metrics (random cores)
            setC_block=$(grep -A3 "Test Set C" "$outfile" 2>/dev/null | head -4)
            setC_r2=$(echo "$setC_block" | grep -o 'R²: [0-9.-]*' | awk '{print $2}')
            setC_mape=$(echo "$setC_block" | grep -o 'MAPE: [0-9.]*%' | sed 's/MAPE: //;s/%//')
            setC_mse=$(echo "$setC_block" | grep -o 'MSE: [0-9.]*' | head -1 | awk '{print $2}')

            # Extract per-fill-type MAPE for Set C
            setC_ft_block=$(sed -n '/Per-fill-type breakdown for Set C/,/^$/p' "$outfile" 2>/dev/null)
            setC_4G=$(echo "$setC_ft_block" | grep '4G:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setC_3G1P=$(echo "$setC_ft_block" | grep '3G1P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setC_3G1B=$(echo "$setC_ft_block" | grep '3G1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setC_2G2P=$(echo "$setC_ft_block" | grep '2G2P:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setC_2G2B=$(echo "$setC_ft_block" | grep '2G2B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')
            setC_2G1P1B=$(echo "$setC_ft_block" | grep '2G1P1B:' | grep -o 'MAPE=[0-9.]*%' | sed 's/MAPE=//;s/%//')

            # Determine status
            if [ -n "$ovr_r2" ]; then
                ((LC_DONE++))
                lc_status="done"
                status_color="${GREEN}done${NC}"
            elif grep -q "Error\|Traceback\|FAILED" "$outfile" 2>/dev/null; then
                lc_status="FAIL"
                status_color="${RED}FAIL${NC}"
            else
                lc_status="run"
                status_color="${YELLOW}run…${NC}"
            fi

            printf "%-8s %-12s %-10s %-10s %-10s | %-10s %-10s | %-10s %-10s | %-10s %-10s ${status_color}\n" \
                "$n_geo" "$lc_target" \
                "${ovr_r2:---}" "${ovr_mape:---}" "${ovr_mse:---}" \
                "${setA_r2:---}" "${setA_mape:---}" \
                "${setB_r2:---}" "${setB_mape:---}" \
                "${setC_r2:---}" "${setC_mape:---}"

            # Append to CSV
            echo "${n_geo},$((n_geo * 33)),${lc_target},${ovr_mse:---},${ovr_r2:---},${ovr_mape:---},${setA_mse:---},${setA_r2:---},${setA_mape:---},${setB_mse:---},${setB_r2:---},${setB_mape:---},${setA_4G:---},${setA_3G1P:---},${setA_3G1B:---},${setA_2G2P:---},${setA_2G2B:---},${setA_2G1P1B:---},${setB_4G:---},${setB_3G1P:---},${setB_3G1B:---},${setB_2G2P:---},${setB_2G2B:---},${setB_2G1P1B:---},${setC_mse:---},${setC_r2:---},${setC_mape:---},${setC_4G:---},${setC_3G1P:---},${setC_3G1B:---},${setC_2G2P:---},${setC_2G2B:---},${setC_2G1P1B:---},${lc_status}" >> "$CSV_FILE"
        done
    done

    echo "$THIN_DIV"
    echo ""
    echo -e "${BOLD}Learning Curve Summary:${NC}  ${LC_FOUND} log files, ${LC_DONE} completed"
    echo -e "CSV saved to: ${CYAN}${CSV_FILE}${NC}"
fi
