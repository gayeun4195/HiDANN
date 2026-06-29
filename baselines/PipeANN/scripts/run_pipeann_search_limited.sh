#!/bin/bash

# ==========================================================================================
# run_pipeann_search_limited.sh
# 
# Purpose:
#   Run PipeANN search experiments under a strict 10% memory budget.
#   It dynamically sizes the Memory Index (entry point) to fit within the budget
#   after accounting for Resident PQ Tables and Thread Overhead.
# ==========================================================================================

# ==========================================================================================
# Configuration
# ==========================================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
OUTPUT_BASE="${OUTPUT_BASE:-runs/baseline_search/PipeANN}"
LOG_DIR="${OUTPUT_BASE}/logs/QPS"
DATA_ROOT="${DATA_ROOT:-data}"
VIBE_ROOT="${VIBE_ROOT:-${DATA_ROOT}/vibe}"
MSTURING_ROOT="${MSTURING_ROOT:-${DATA_ROOT}/msturing}"
GIST_ROOT="${GIST_ROOT:-${DATA_ROOT}/gist}"
GENERATED_ROOT="${GENERATED_ROOT:-${DATA_ROOT}/generated}"
mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

# PipeANN / DiskANN Locations
PIPEANN_ROOT="${PIPEANN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DISKANN_ROOT="${DISKANN_ROOT:-$(cd "${SCRIPT_DIR}/../../DiskANN" && pwd)}"
SIGMA_ROOT="${SIGMA_ROOT:-$DISKANN_ROOT}"

# Tools
SEARCH_EXE="${PIPEANN_ROOT}/build/tests/search_disk_index"
BUILD_MEM_EXE="${PIPEANN_ROOT}/build/tests/build_memory_index"
GEN_SLICE_EXE="${PIPEANN_ROOT}/build/tests/utils/gen_random_slice"

# Construction Base (Where DiskANN indices are)
CONSTRUCTION_BASE="${CONSTRUCTION_BASE:-runs/baseline_construction/DiskANN}"

# Import Budget Calculator from SIGMA
source "${SIGMA_ROOT}/scripts/2_search/cache/calculate_cache_budget_optimized.sh"

# Parameters
THREADS=64
BEAM_WIDTH=8 # Search Beam Width (PipeANN parameter)

K=10
SEARCH_RAM_BUDGET_RATIO=0.1
R=32
BUILD_L=50   # For Memory Index Construction
SEARCH_L_LIST="10 20 30 40 50 75 100 150 200 250 300 350 400 450 500"
RUNS_PER_CONFIG=10

# Define Datasets (Subset as requested)
declare -A DATA_PATHS=(
  [simplewiki]="${VIBE_ROOT}/simplewiki/simplewiki.fbin"
  [agnews]="${VIBE_ROOT}/agnews/agnews.fbin"
  [gooaq]="${VIBE_ROOT}/gooaq/gooaq.fbin"
  [yahoo]="${VIBE_ROOT}/yahoo/yahoo.fbin"
  [gist]="${GIST_ROOT}/dataset/gist1M.fbin"
  [msturing100M]="${MSTURING_ROOT}/dataset/msturing100M.fbin"
  [text10M]="${GENERATED_ROOT}/fineweb/fineweb_1024d_10m.fbin"
  [image10M]="${GENERATED_ROOT}/imagenet/imagenet_1024d_10m.fbin"
)

declare -A DATA_QUERIES=(
  [simplewiki]="${VIBE_ROOT}/simplewiki/simplewiki_query.fbin"
  [agnews]="${VIBE_ROOT}/agnews/agnews_query.fbin"
  [gooaq]="${VIBE_ROOT}/gooaq/gooaq_query.fbin"
  [yahoo]="${VIBE_ROOT}/yahoo/yahoo_query.fbin"
  [gist]="${GIST_ROOT}/query/gist_query.fbin"
  [msturing100M]="${MSTURING_ROOT}/query/query100K.fbin"
  [text10M]="${GENERATED_ROOT}/fineweb/fineweb_1024d_query_10k.fbin"
  [image10M]="${GENERATED_ROOT}/imagenet/imagenet_1024d_query_10k.fbin"
)

declare -A DATA_GTS=(
  [simplewiki]="${VIBE_ROOT}/simplewiki/simplewiki_gt100"
  [agnews]="${VIBE_ROOT}/agnews/agnews_gt100"
  [gooaq]="${VIBE_ROOT}/gooaq/gooaq_gt100"
  [yahoo]="${VIBE_ROOT}/yahoo/yahoo_gt100"
  [gist]="${GIST_ROOT}/gt/gist1M_gt100"
  [msturing100M]="${MSTURING_ROOT}/gt/msturing100M_100k_gt100"
  [text10M]="${GENERATED_ROOT}/fineweb/fineweb_1024d_10m_gt100.bin"
  [image10M]="${GENERATED_ROOT}/imagenet/imagenet_10m_gt100.bin"
)

declare -A DATA_DIMS=(
  [simplewiki]=3072
  [agnews]=1024
  [gooaq]=768
  [yahoo]=384
  [gist]=960
  [msturing100M]=100
  [text10M]=1024
  [image10M]=1024
)

# PQ Bytes (QD)
declare -A DATA_QDS=(
  [simplewiki]=512
  [agnews]=256
  [gooaq]=192
  [yahoo]=96
  [gist]=240
  [msturing100M]=25
  [text10M]=256
  [image10M]=256
)

run_pipeann_experiment() {
    local NAME=$1
    local DATA_PATH=${DATA_PATHS[$NAME]}
    local QUERY_FILE=${DATA_QUERIES[$NAME]}
    local GT_FILE=${DATA_GTS[$NAME]}
    local DIM=${DATA_DIMS[$NAME]}
    local QD=${DATA_QDS[$NAME]}
    local LOG_FILE="${LOG_DIR}/${NAME}_pipeann_limited.log"

    if [ -z "$DATA_PATH" ]; then return; fi

    echo "========================================================================"
    echo "Starting PipeANN Search Experiment (Limited Budget) for $NAME"
    echo "========================================================================"

    # 1. Budget Calculation
    local FILE_SIZE=$(stat -c%s "$DATA_PATH")
    local N=$(( (FILE_SIZE - 8) / (DIM * 4) ))
    
    # Use SIGMA's calculator to get strict byte budget and cgroup limit
    # Args: N DIM R QD THREADS BEAM_WIDTH RATIO K P BATCH
    calculate_cache_and_physical_limit $N $DIM $R $QD $THREADS $BEAM_WIDTH $SEARCH_RAM_BUDGET_RATIO $K 0 50

    echo "  Total N: $N"
    echo "  Strict Budget (Bytes): $STRICT_BUDGET_BYTES"
    echo "  Relaxed Cgroup Limit: $RELAXED_CGROUP_LIMIT"

    # 2. Calculate Available Memory for Memory Index
    # PipeANN Memory Components:
    # A. PQ Compressed Vectors: N * QD
    # B. PQ Pivots: 256 * Dim * 4 (Usually negligible but let's deduct)
    # C. Memory Index Graph + Data
    
    local PQ_VEC_BYTES=$(echo "$N * $QD" | bc)
    local PQ_PIVOT_BYTES=$(echo "256 * $DIM * 4" | bc)
    local PQ_TOTAL_BYTES=$(echo "$PQ_VEC_BYTES + $PQ_PIVOT_BYTES" | bc)
    
    local AVAILABLE_FOR_INDEX=$(echo "$STRICT_BUDGET_BYTES - $PQ_TOTAL_BYTES" | bc)
    
    if [ "$AVAILABLE_FOR_INDEX" -le 0 ]; then
        echo "  [ERROR] Budget too small! PQ Vectors take up almost everything."
        echo "          PQ: $PQ_TOTAL_BYTES vs Budget: $STRICT_BUDGET_BYTES"
        return
    fi
    
    echo "  PQ Resident Bytes: $PQ_TOTAL_BYTES"
    echo "  Available for Mem Index: $AVAILABLE_FOR_INDEX"

    # 3. Calculate Sampling Rate
    # Formula: Size = 1.1 * (N_samp * R * 4 * 1.3 + N_samp * Dim * 4)
    #               = N_samp * 1.1 * (5.2 * R + 4 * Dim)
    # N_samp = Available / (1.1 * (5.2 * R + 4 * Dim))
    
    local PER_NODE_COST=$(echo "1.1 * (5.2 * $R + 4 * $DIM)" | bc)
    local MAX_N_SAMPLE=$(echo "$AVAILABLE_FOR_INDEX / $PER_NODE_COST" | bc)
    
    # Calculate Rate
    local SAMPLING_RATE=$(echo "scale=6; $MAX_N_SAMPLE / $N" | bc)
    
    # Cap Rate at 1.0
    if (( $(echo "$SAMPLING_RATE > 1.0" | bc -l) )); then
        SAMPLING_RATE=1.0
    fi
    # Min Rate check (e.g. 0.001)
    if (( $(echo "$SAMPLING_RATE < 0.00001" | bc -l) )); then
         echo "  [WARNING] Sampling Rate too low ($SAMPLING_RATE). Setting to minimal sampling."
         # Do we continue? PipeANN might fail.
    fi

    echo "  Max Memory Index Nodes: $MAX_N_SAMPLE"
    echo "  Calculated Sampling Rate: $SAMPLING_RATE"

    # 4. Prepare Environment (Symlink Farm)
    local WORK_DIR="${OUTPUT_BASE}/index/${NAME}"
    mkdir -p "$WORK_DIR"
    rm -f "${WORK_DIR}"/* # Clean previous run
    
    local SRC_PREFIX="${CONSTRUCTION_BASE}/${NAME}"
    local TARGET_PREFIX="${WORK_DIR}/${NAME}_idx"
    
    # Link Disk Index files
    ln -sf "${SRC_PREFIX}_disk.index" "${TARGET_PREFIX}_disk.index"
    ln -sf "${SRC_PREFIX}_disk.index.tags" "${TARGET_PREFIX}_disk.index.tags"
    ln -sf "${SRC_PREFIX}_pq_pivots.bin" "${TARGET_PREFIX}_pq_pivots.bin"
    ln -sf "${SRC_PREFIX}_pq_compressed.bin" "${TARGET_PREFIX}_pq_compressed.bin"

    # 5. Build Memory Index
    # Gen Random Slice
    # usage: gen_random_slice <type> <data_bin> <output_prefix> <sampling_rate>
    echo "  Generating Random Slice ($SAMPLING_RATE)..."
    DATA_TYPE="float" # Assuming float for now (based on SIGMA/DiskANN scripts using float fbin)
    # Wait, DiskANN script uses 'float' for build_disk_index data_type.
    # gen_random_slice needs specific type? code was template.
    # check if dataset is float or uint8.
    # The file extension is .fbin, so float.
    
    # If sampling rate is 1.0, gen_random_slice might behave or we might just link.
    # But usually creating a separate file is safer.
    
    "$GEN_SLICE_EXE" float "$DATA_PATH" "${TARGET_PREFIX}_SAMPLE" "$SAMPLING_RATE" >> "$LOG_FILE" 2>&1
    
    local SAMPLE_DATA="${TARGET_PREFIX}_SAMPLE_data.bin"
    local SAMPLE_IDS="${TARGET_PREFIX}_SAMPLE_ids.bin"
    
    echo "  Building Memory Index..."
    # Usage: build_memory_index <type> <data> <ids> <output_index> <R> <L> <alpha> <threads> <metric>
    "$BUILD_MEM_EXE" float "$SAMPLE_DATA" "$SAMPLE_IDS" "${TARGET_PREFIX}_mem.index" \
        "$R" "$BUILD_L" 1.2 "$THREADS" l2 >> "$LOG_FILE" 2>&1

    # 6. Run Search (Repeatedly)
    # Search Mode 2 (PipeANN)
    local MEM_L=1
    
    for i in $(seq 1 $RUNS_PER_CONFIG); do
        local LOG_FILE_RUN="${LOG_DIR}/${NAME}_pipeann_limited_run${i}.log"
        echo "  Running Search Run $i/$RUNS_PER_CONFIG..."
        
        # Copy previous build logs to the run log to keep context
        cat "$LOG_FILE" > "$LOG_FILE_RUN"
        
        local UNIT_NAME="pipeann_${NAME}_run${i}_$$"
        
        # Usage: search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <K> <metric> <nbr_type> <mode> <mem_L> <Ls...>
        systemd-run --user --scope --unit="$UNIT_NAME" \
            -p MemoryMax=$RELAXED_CGROUP_LIMIT \
            "$SEARCH_EXE" float "${TARGET_PREFIX}" "$THREADS" "$BEAM_WIDTH" \
            "$QUERY_FILE" "$GT_FILE" "$K" l2 pq 2 "$MEM_L" $SEARCH_L_LIST >> "$LOG_FILE_RUN" 2>&1
            
        systemctl --user stop "$UNIT_NAME" 2>/dev/null
    done

    echo "  Completed $NAME."
    echo ""
}

# Run Batch
if [ -n "$1" ]; then
    run_pipeann_experiment "$1"
else
    DATASETS=("simplewiki" "agnews" "gooaq" "yahoo" "gist" "msturing100M" "text10M" "image10M")
    for dataset in "${DATASETS[@]}"; do
        run_pipeann_experiment "$dataset"
    done
fi
