#!/bin/bash

# ==========================================================================================
# run_diskann_search_on_sigma_index_optimized.sh
# 
# Purpose:
#   Run DiskANN search experiments on SIGMA indices using OPTIMIZED settings.
#   1. Converts SIGMA 'mem layout' to DiskANN 'disk layout'.
#   2. Calculates Exact Medoid and patches it.
#   3. Runs search with Optimized Budget and systemd-run.
# ==========================================================================================

# ==========================================================================================
# Configuration
# ==========================================================================================
# Output for search results
OUTPUT_BASE="${OUTPUT_BASE:-runs/baseline_search/DiskANN}"
LOG_DIR="${OUTPUT_BASE}/logs/SIGMA_index/QPS"

# Path where SIGMA construction script output the indices
CONSTRUCTION_BASE="${CONSTRUCTION_BASE:-runs/baseline_construction/SIGMA}"
DATA_ROOT="${DATA_ROOT:-data}"
VIBE_ROOT="${VIBE_ROOT:-${DATA_ROOT}/vibe}"
DEEP_ROOT="${DEEP_ROOT:-${DATA_ROOT}/deep}"
MSTURING_ROOT="${MSTURING_ROOT:-${DATA_ROOT}/msturing}"
GIST_ROOT="${GIST_ROOT:-${DATA_ROOT}/gist}"
GENERATED_ROOT="${GENERATED_ROOT:-${DATA_ROOT}/generated}"

mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

# Parameters
THREADS=64
BEAM_WIDTH=8
K=10
SEARCH_RAM_BUDGET_RATIO=0.1
L_VALUES=(10 20 30 40 50 75 100 150 200 250 300 350 400 450 500)
RUNS_PER_CONFIG=10
R=32

# Import OPTIMIZED Budget Calculator
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/cache/calculate_cache_budget_optimized.sh"

# DiskANN Tools
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DISKANN_REPO="$REPO_ROOT"
CREATE_LAYOUT_EXE="${DISKANN_REPO}/build/apps/utils/create_disk_layout"
SEARCH_DISK_INDEX="${DISKANN_REPO}/build/apps/search_disk_index"

if [ ! -f "$CREATE_LAYOUT_EXE" ]; then
    echo "Error: create_disk_layout not found at $CREATE_LAYOUT_EXE"
    exit 1
fi

if [ ! -f "$SEARCH_DISK_INDEX" ]; then
    echo "Error: search_disk_index not found at $SEARCH_DISK_INDEX"
    exit 1
fi

# Define Datasets
declare -A DATA_PATHS=(
  [agnews]="${VIBE_ROOT}/agnews/agnews.fbin"
  [arxiv]="${VIBE_ROOT}/arxiv/arxiv.fbin"
  [ccnews]="${VIBE_ROOT}/ccnews/ccnews.fbin"
  [gooaq]="${VIBE_ROOT}/gooaq/gooaq.fbin"
  [imagenet-512]="${VIBE_ROOT}/imagenet-512/imagenet.fbin"
  [landmark]="${VIBE_ROOT}/landmark/landmark.fbin"
  [simplewiki]="${VIBE_ROOT}/simplewiki/simplewiki.fbin"
  [yahoo]="${VIBE_ROOT}/yahoo/yahoo.fbin"
  [deep10M]="${DEEP_ROOT}/dataset/deep10M.fbin"
  [msturing10M]="${MSTURING_ROOT}/dataset/msturing10M.fbin"
  [deep100M]="${DEEP_ROOT}/dataset/deep100M.fbin"
  [msturing100M]="${MSTURING_ROOT}/dataset/msturing100M.fbin"
  [gist]="${GIST_ROOT}/dataset/gist1M.fbin"
  [text10M]="${GENERATED_ROOT}/fineweb/fineweb_1024d_10m.fbin"
  [image10M]="${GENERATED_ROOT}/imagenet/imagenet_1024d_10m.fbin"
)

# Define Query Paths
declare -A DATA_QUERIES=(
  [agnews]="${VIBE_ROOT}/agnews/agnews_query.fbin"
  [arxiv]="${VIBE_ROOT}/arxiv/arxiv_query.fbin"
  [ccnews]="${VIBE_ROOT}/ccnews/ccnews_query.fbin"
  [gooaq]="${VIBE_ROOT}/gooaq/gooaq_query.fbin"
  [imagenet-512]="${VIBE_ROOT}/imagenet-512/imagenet_query.fbin"
  [landmark]="${VIBE_ROOT}/landmark/landmark_query.fbin"
  [simplewiki]="${VIBE_ROOT}/simplewiki/simplewiki_query.fbin"
  [yahoo]="${VIBE_ROOT}/yahoo/yahoo_query.fbin"
  [deep10M]="${DEEP_ROOT}/query/deep1B_queries.fbin"
  [msturing10M]="${MSTURING_ROOT}/query/query100K.fbin"
  [deep100M]="${DEEP_ROOT}/query/deep1B_queries.fbin"
  [msturing100M]="${MSTURING_ROOT}/query/query100K.fbin"
  [gist]="${GIST_ROOT}/query/gist_query.fbin"
  [text10M]="${GENERATED_ROOT}/fineweb/fineweb_1024d_query_10k.fbin"
  [image10M]="${GENERATED_ROOT}/imagenet/imagenet_1024d_query_10k.fbin"
)

# Define Groundtruth Paths
declare -A DATA_GTS=(
  [agnews]="${VIBE_ROOT}/agnews/agnews_gt100"
  [arxiv]="${VIBE_ROOT}/arxiv/arxiv_gt100"
  [ccnews]="${VIBE_ROOT}/ccnews/ccnews_gt100"
  [gooaq]="${VIBE_ROOT}/gooaq/gooaq_gt100"
  [imagenet-512]="${VIBE_ROOT}/imagenet-512/imagenet_gt100"
  [landmark]="${VIBE_ROOT}/landmark/landmark_gt100"
  [simplewiki]="${VIBE_ROOT}/simplewiki/simplewiki_gt100"
  [yahoo]="${VIBE_ROOT}/yahoo/yahoo_gt100"
  [deep10M]="${DEEP_ROOT}/gt/deep10M_gt100"
  [msturing10M]="${MSTURING_ROOT}/gt/msturing10M_100k_gt100"
  [deep100M]="${DEEP_ROOT}/gt/deep100M_gt100"
  [msturing100M]="${MSTURING_ROOT}/gt/msturing100M_100k_gt100"
  [gist]="${GIST_ROOT}/gt/gist1M_gt100"
  [text10M]="${GENERATED_ROOT}/fineweb/fineweb_1024d_10m_gt100.bin"
  [image10M]="${GENERATED_ROOT}/imagenet/imagenet_10m_gt100.bin"
)

# Define QDs
declare -A DATA_QDS=(
  [agnews]=256
  [arxiv]=192
  [ccnews]=128
  [gooaq]=192
  [imagenet-512]=128
  [landmark]=192
  [simplewiki]=512
  [yahoo]=96
  [deep10M]=32
  [msturing10M]=25
  [deep100M]=32
  [msturing100M]=25
  [gist]=240
  [text10M]=256
  [image10M]=256
)

# Define Dims
declare -A DATA_DIMS=(
  [agnews]=1024
  [arxiv]=768
  [ccnews]=768
  [gooaq]=768
  [imagenet-512]=512
  [landmark]=768
  [simplewiki]=3072
  [yahoo]=384
  [deep10M]=96
  [msturing10M]=100
  [deep100M]=96
  [msturing100M]=100
  [gist]=960
  [text10M]=1024
  [image10M]=1024
)


run_sigma_search_experiment() {
    local NAME=$1
    local DATA_PATH=${DATA_PATHS[$NAME]}
    local QD=${DATA_QDS[$NAME]}
    local DIM=${DATA_DIMS[$NAME]}
    local QUERY_FILE=${DATA_QUERIES[$NAME]}
    local GT_FILE=${DATA_GTS[$NAME]}
    
    if [ -z "$DATA_PATH" ]; then echo "Error: Missing config for $NAME"; return; fi
    if [ -z "$QD" ]; then echo "Error: Missing QD for $NAME"; return; fi

    echo "========================================================================"
    echo "Starting Search Experiment on SIGMA Index (Optimized) for $NAME"
    echo "========================================================================"

    local SRC_GRAPH="${CONSTRUCTION_BASE}/${NAME}_P22_QD${QD}_final.mem.index"
    local SRC_PQ_CODES="${CONSTRUCTION_BASE}/${NAME}_QD${QD}.pq_compressed.bin"
    local SRC_PQ_PIVOTS="${CONSTRUCTION_BASE}/${NAME}_QD${QD}.pq_pivots.bin"

    if [ ! -f "$SRC_GRAPH" ]; then echo "Error: Graph not found: $SRC_GRAPH"; return; fi

    # Compatibility: Check for renamed files from previous runs and rollback if possible to keep clean state
    local RENAMED_PQ_CODES="${CONSTRUCTION_BASE}/${NAME}_QD${QD}_pq_compressed.bin"
    local RENAMED_PQ_PIVOTS="${CONSTRUCTION_BASE}/${NAME}_QD${QD}_pq_pivots.bin"

    if [ -f "$RENAMED_PQ_CODES" ] && [ ! -f "$SRC_PQ_CODES" ]; then
        echo "Rolling back previously renamed PQ files..."
        mv "$RENAMED_PQ_CODES" "$SRC_PQ_CODES"
        mv "$RENAMED_PQ_PIVOTS" "$SRC_PQ_PIVOTS"
    fi

    if [ ! -f "$SRC_PQ_CODES" ]; then
         echo "Error: Source PQ Codes not found at $SRC_PQ_CODES"
         return
    fi

    # --- Setup Dedicated Layout Directory ---
    local DISK_LAYOUT_DIR="${CONSTRUCTION_BASE}/disk_layout"
    mkdir -p "$DISK_LAYOUT_DIR"
    
    local TARGET_PQ_CODES="${DISK_LAYOUT_DIR}/${NAME}_QD${QD}_pq_compressed.bin"
    local TARGET_PQ_PIVOTS="${DISK_LAYOUT_DIR}/${NAME}_QD${QD}_pq_pivots.bin"
    local TARGET_DISK_INDEX="${DISK_LAYOUT_DIR}/${NAME}_QD${QD}_disk.index"

    # Copy files
    if [ ! -f "$TARGET_PQ_CODES" ]; then
        echo "Copying PQ files..."
        cp "$SRC_PQ_CODES" "$TARGET_PQ_CODES"
        cp "$SRC_PQ_PIVOTS" "$TARGET_PQ_PIVOTS"
    fi

    # Copy Medoids
    local SRC_MEDOIDS="${SRC_GRAPH}_medoids.bin"
    local TARGET_MEDOIDS="${TARGET_DISK_INDEX}_medoids.bin"
    
    if [ -f "$SRC_MEDOIDS" ] && [ ! -f "$TARGET_MEDOIDS" ]; then
         echo "Copying Medoids..."
         cp "$SRC_MEDOIDS" "$TARGET_MEDOIDS"
    fi

    # Create Layout (Skipped as per user instruction: assume exists)
    # However, we still define variables for paths
    
    # --- Patch Medoid (SKIPPED) ---
    # User confirmed disk_layout exists, so we assume medoids are correct or handled.
    # We will just rely on reading the file for P.

    # 4. Calculate Memory Budget
    local FILE_SIZE=$(stat -c%s "$DATA_PATH")
    local N=$(( (FILE_SIZE - 8) / (DIM * 4) ))
    local INDEX_PREFIX="${DISK_LAYOUT_DIR}/${NAME}_QD${QD}"
    
    # Medoid Detection
    local MEDOID_FILE="${INDEX_PREFIX}_medoids.bin"
    if [ ! -f "$MEDOID_FILE" ]; then MEDOID_FILE="${INDEX_PREFIX}_disk.index_medoids.bin"; fi
    local P=1
    if [ -f "$MEDOID_FILE" ]; then
        P=$(od -An -tu4 -N4 "$MEDOID_FILE" | tr -d ' ' | xargs)
        if [ -z "$P" ]; then P=1; fi
    fi
    
    calculate_cache_and_physical_limit $N $DIM $R $QD $THREADS $BEAM_WIDTH $SEARCH_RAM_BUDGET_RATIO $K $P
    
    local MSG_CALC="  Budget Calculation:
    N: $N, Dim: $DIM, QD: $QD
    Budget Ratio: $SEARCH_RAM_BUDGET_RATIO
    Detected Medoids (P): $P
    Calculated Cache Nodes: $CALCULATED_CACHE_NODES
    Calculated Cache Ratio: $CALCULATED_CACHE_RATIO %
    Relaxed Cgroup Limit: $(($RELAXED_CGROUP_LIMIT / 1024 / 1024)) MB"
    echo "$MSG_CALC"
    
    # 5. Run Search (Optimized with systemd-run)
    local L_LIST="${L_VALUES[*]}"
    
    for i in $(seq 1 $RUNS_PER_CONFIG); do
        local LOG_FILE="${LOG_DIR}/${NAME}_search_cache_optimized_run${i}.log"
        echo "  Run $i/$RUNS_PER_CONFIG..."
        echo "$MSG_CALC" >> "$LOG_FILE"
        
        local UNIT_NAME="sigma_idx_search_${NAME}_run${i}_$$"
        
        systemd-run --user --scope --unit="$UNIT_NAME" \
            -p MemoryMax=$RELAXED_CGROUP_LIMIT \
            "$SEARCH_DISK_INDEX" \
            --data_type float \
            --dist_fn l2 \
            --index_path_prefix "$INDEX_PREFIX" \
            --query_file "$QUERY_FILE" \
            --gt_file "$GT_FILE" \
            --num_nodes_to_cache "$CALCULATED_CACHE_NODES" \
            -K "$K" \
            -L $L_LIST \
            -W "$BEAM_WIDTH" \
            -T "$THREADS" \
            --result_path "${OUTPUT_BASE}/${NAME}_res_run${i}" \
            >> "$LOG_FILE" 2>&1
            
        if [ $? -ne 0 ]; then
            echo "    FAILED Run $i"
        else
            echo "    SUCCESS Run $i"
        fi
        systemctl --user stop "$UNIT_NAME" 2>/dev/null
    done
}

# Run for requested dataset (Dry run with arxiv)
DATASETS=("simplewiki" "agnews" "gooaq" "yahoo" "gist" "msturing100M" "text10M" "image10M")

for NAME in "${DATASETS[@]}"; do
    run_sigma_search_experiment "$NAME"
done

echo "Optimized Search on SIGMA Index completed."
