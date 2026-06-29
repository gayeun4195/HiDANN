#!/bin/bash

# Configuration
OUTPUT_BASE="${OUTPUT_BASE:-runs/baseline_construction/SOGAIC}"
LOG_DIR="${OUTPUT_BASE}/logs"
GROUP_NAME="sigma_exp"
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
THREADS=64
BUDGET_RATIO=0.1

# SOGAIC Specifics
MAX_OVERLAP=4.0
EPSILON=1.8
GAMMA_SLACK=1.0

# Pipeline Script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_sogaic_pipeline_limited.sh"

if [ ! -f "$PIPELINE_SCRIPT" ]; then
  echo "Error: Pipeline script not found at $PIPELINE_SCRIPT"
  exit 1
fi

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

# Define Data Dimensions (D) based on provided image
# Format: [name]=Dim
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
  [text10M]=256
  [image10M]=256
)


# Ensure BC is installed
if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' calculator not found. Please install: sudo apt-get install bc"
    exit 1
fi

run_sogaic_limited() {
    local NAME=$1
    local DATA_PATH=${DATA_PATHS[$NAME]}
    local DIM=${DATA_DIMS[$NAME]}
    local LOG_FILE="${LOG_DIR}/${NAME}_sogaic_limited.log"
    
    if [ -z "$DATA_PATH" ]; then
        echo "Error: Data path for '$NAME' not defined."
        return
    fi
    if [ -z "$DIM" ]; then
        echo "Error: Dimension for '$NAME' not defined."
        return
    fi
    
    # --- Dynamic Memory Limit Calculation (Same logic as DiskANN) ---
    FILE_SIZE=$(stat -c%s "$DATA_PATH")
    N=$(( (FILE_SIZE - 8) / (DIM * 4) ))
    
    # Calculate Dynamic PQ Sampling Rate (Same as DiskANN/SIGMA)
    # Rate1 = 0.05
    # Rate2 = 256000 / N
    PQ_SAMPLING_RATE=$(echo "scale=6; rate=256000/$N; if(rate < 0.05) rate else 0.05" | bc)
    
    GRAPH_SIZE=$(( N * R * 4 ))
    TOTAL_INDEX_SIZE=$(( FILE_SIZE + GRAPH_SIZE ))
    
    # Intended Limit: Total * Ratio
    # This LIMIT is based on the ORIGINAL dataset size, ensuring fair hardware comparison.
    # SOGAIC will have to fit its expanded index construction into this same physical RAM.
    INTENDED_LIMIT=$(echo "$TOTAL_INDEX_SIZE * $BUDGET_RATIO" | bc | cut -d. -f1)
    
    # Calculate Safe Chunk Budget for C++ logic (50% of intended memory limit)
    # This sets the processing block size dynamic limit
    CHUNK_BUDGET_BYTES=$(echo "$INTENDED_LIMIT * 0.5 / 1" | bc)
    export DISKANN_CHUNK_BYTES=$CHUNK_BUDGET_BYTES

    # Add Buffer for OS overhead (e.g., 512MB)
    CGROUP_LIMIT_BYTES=$(( INTENDED_LIMIT + 536870912 ))
    
    echo "----------------------------------------------------------------"
    echo "Running LIMITED Memory SOGAIC experiment for: $NAME"
    echo "  Data: $DATA_PATH"
    echo "  Dim: $DIM, N: $N"
    echo "  Intended Memory Limit (Ratio $BUDGET_RATIO): $(($INTENDED_LIMIT / 1024 / 1024)) MB"
    echo "  Cgroup Limit (with buffer): $(($CGROUP_LIMIT_BYTES / 1024 / 1024)) MB"
    echo "  Dynamic PQ Sampling Rate: $PQ_SAMPLING_RATE"
    echo "  Internal Chunk Budget: $(($CHUNK_BUDGET_BYTES / 1024 / 1024)) MB"
    echo "----------------------------------------------------------------"
    
    # Run Pipeline Script using systemd-run (No Sudo Required)
    # This creates a transient scope strictly for this process with the defined memory limits.
    UNIT_NAME="sogaic_${NAME}_$$"

    # Clean up function
    cleanup() {
        echo "Caught signal, cleaning up..."
        systemctl --user stop "$UNIT_NAME.scope" 2>/dev/null
        exit 1
    }
    trap cleanup SIGINT SIGTERM

    # Run the workload in the FOREGROUND
    # Note: We do NOT pass --cgroup to the inner script. The systemd scope handles it for the whole tree.
    systemd-run --user --scope --unit="$UNIT_NAME" \
        -p MemoryMax=${CGROUP_LIMIT_BYTES} \
        "$PIPELINE_SCRIPT" "$DATA_PATH" "$OUTPUT_BASE/$NAME" \
        --datatype float \
        --budget-ratio "$BUDGET_RATIO" \
        --overlap "$MAX_OVERLAP" \
        --epsilon "$EPSILON" \
        --gamma "$GAMMA_SLACK" \
        --threads "$THREADS" \
        --R "$R" \
        --L "$L" \
        --chunk-budget "$CHUNK_BUDGET_BYTES" \
        --memory-budget-bytes "$INTENDED_LIMIT" \
        --sampling "$PQ_SAMPLING_RATE" \
        2>&1 | tee "$LOG_FILE"

        
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "FAILED: $NAME"
    else
        echo "SUCCESS: $NAME"
    fi
}

echo "Starting SOGAIC LIMITED Memory Batch Experiments..."
echo "Output Directory: $OUTPUT_BASE"

# Iterate over all datasets
# for NAME in "${!DATA_PATHS[@]}"; do
#     run_sogaic_limited "$NAME"
# done

run_sogaic_limited "text10M"
run_sogaic_limited "image10M"

echo "Batch completed."
