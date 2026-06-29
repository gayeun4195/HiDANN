#!/bin/bash

# Configuration
OUTPUT_BASE="${OUTPUT_BASE:-runs/baseline_construction/vamana}"
LOG_DIR="${OUTPUT_BASE}/logs"
DATA_ROOT="${DATA_ROOT:-data}"
VIBE_ROOT="${VIBE_ROOT:-${DATA_ROOT}/vibe}"
DEEP_ROOT="${DEEP_ROOT:-${DATA_ROOT}/deep}"
MSTURING_ROOT="${MSTURING_ROOT:-${DATA_ROOT}/msturing}"
GIST_ROOT="${GIST_ROOT:-${DATA_ROOT}/gist}"
GENERATED_ROOT="${GENERATED_ROOT:-${DATA_ROOT}/generated}"
mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

# Parameters
L=50
R=32
ALPHA=1.0
THREADS=64

# Define Datasets and Paths
# Format: [name]="path"
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

# Determine repo root relative to this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_MEMORY_INDEX="$REPO_ROOT/build/apps/build_memory_index"

if [ ! -f "$BUILD_MEMORY_INDEX" ]; then
  echo "Error: build_memory_index executable not found at $BUILD_MEMORY_INDEX"
  exit 1
fi

run_experiment() {
    local NAME=$1
    local DATA_PATH=${DATA_PATHS[$NAME]}
    local LOG_FILE="${LOG_DIR}/${NAME}_vamana.log"
    
    if [ -z "$DATA_PATH" ]; then
        echo "Error: Data path for '$NAME' not defined."
        return
    fi
    
    local OUTPUT_PREFIX="${OUTPUT_BASE}/${NAME}_R${R}_L${L}_A${ALPHA}"

    echo "----------------------------------------------------------------"
    echo "Running Vamana experiment for: $NAME"
    echo "  Data: $DATA_PATH"
    echo "  Output Prefix: $OUTPUT_PREFIX"
    echo "  Log: $LOG_FILE"
    echo "----------------------------------------------------------------"
    
    # Run build_memory_index
    # Parameters from reference:
    # --data_type float
    # --dist_fn l2
    # --index_path_prefix ...
    # --data_path ...
    # -R ...
    # -L ...
    # --alpha ...
    "$BUILD_MEMORY_INDEX" \
        --data_type float \
        --dist_fn l2 \
        --index_path_prefix "$OUTPUT_PREFIX" \
        --data_path "$DATA_PATH" \
        -R "$R" \
        -L "$L" \
        --alpha "$ALPHA" \
        -T "$THREADS" \
        2>&1 | tee "$LOG_FILE"
        
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "FAILED: $NAME"
    else
        echo "SUCCESS: $NAME"
        
        # Parse and print summary
        SUMMARY=$(grep -E "Indexing time|Time taken for save|Index built with degree" "$LOG_FILE")
        {
            echo ""
            echo "Summary for $NAME:"
            echo "----------------------------------------"
            echo "$SUMMARY"
            echo "----------------------------------------"
        } | tee -a "$LOG_FILE"
    fi
}

echo "Starting Vamana Batch Experiments..."
echo "Output Directory: $OUTPUT_BASE"

# Iterate over defined datasets
# for NAME in "${!DATA_PATHS[@]}"; do
#     run_experiment "$NAME"
# done

run_experiment "text10M"
run_experiment "image10M"

echo "Batch completed."
