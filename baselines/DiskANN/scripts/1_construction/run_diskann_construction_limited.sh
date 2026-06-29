#!/bin/bash

# Configuration
OUTPUT_BASE="${OUTPUT_BASE:-runs/baseline_construction/DiskANN}"
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
SEARCH_RAM_BUDGET_RATIO=0.1
# PQ_SAMPLING_RATE will be calculated dynamically
# BUILD_DRAM_BUDGET will be calculated dynamically

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

# Define QD (Quantization Dimensions) based on table
# Format: [name]=QD
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
  [text10M]=1024
  [image10M]=1024
)

# Determine repo root relative to this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DISK_INDEX="$REPO_ROOT/build/apps/build_disk_index"

if [ ! -f "$BUILD_DISK_INDEX" ]; then
  echo "Error: build_disk_index executable not found at $BUILD_DISK_INDEX"
  exit 1
fi


# Ensure BC is installed
if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' calculator not found. Please install: sudo apt-get install bc"
    exit 1
fi


run_experiment() {
    local NAME=$1
    local DATA_PATH=${DATA_PATHS[$NAME]}
    local QD=${DATA_QDS[$NAME]}
    local DIM=${DATA_DIMS[$NAME]}
    local LOG_FILE="${LOG_DIR}/${NAME}_diskann_limited.log"
    
    if [ -z "$DATA_PATH" ]; then
        echo "Error: Data path for '$NAME' not defined."
        return
    fi
    
    if [ -z "$QD" ]; then
        echo "Error: QD for '$NAME' not defined."
        return
    fi

    if [ -z "$DIM" ]; then
        echo "Error: Dimension (DIM) for '$NAME' not defined."
        return
    fi

    # --- Dynamic Memory Limit Calculation ---
    FILE_SIZE=$(stat -c%s "$DATA_PATH")
    
    # Calculate N (Number of points) from file size: (FileSize - 8 bytes header) / (Dim * 4 bytes)
    N=$(( (FILE_SIZE - 8) / (DIM * 4) ))
    
    # Calculate Dynamic PQ Sampling Rate
    # Rate1 = 0.05
    # Rate2 = 256000 / N
    PQ_SAMPLING_RATE=$(echo "scale=6; rate=256000/$N; if(rate < 0.05) rate else 0.05" | bc)
    
    # Calculate Graph Size: N * R * 4 bytes
    GRAPH_SIZE=$(( N * R * 4 ))
    
    # Estimated Total Index Size
    TOTAL_INDEX_SIZE=$(( FILE_SIZE + GRAPH_SIZE ))
    
    # Intended Limit: Total * Ratio
    INTENDED_LIMIT=$(echo "$TOTAL_INDEX_SIZE * $SEARCH_RAM_BUDGET_RATIO" | bc | cut -d. -f1)
    
    # Calculate Safe Chunk Budget for C++ logic (50% of intended memory limit)
    # This sets the processing block size dynamic limit
    CHUNK_BUDGET_BYTES=$(echo "$INTENDED_LIMIT * 0.5 / 1" | bc)
    export DISKANN_CHUNK_BYTES=$CHUNK_BUDGET_BYTES
    
    # Add Buffer for OS overhead (e.g., 512MB)
    CGROUP_LIMIT_BYTES=$(( INTENDED_LIMIT + 536870912 ))
    
    # Calculate BUILD_DRAM_BUDGET passed to DiskANN (in GB)
    # It must be smaller than the cgroup limit. Let's use the INTENDED_LIMIT converted to GB.
    # We maintain the Ratio logic exactly.
    # Note: DiskANN checks 'full_index_ram * ratio' internally. 
    # But we also need to pass -M (build budget) which acts as a hard cap on buffer sizes.
    # We set -M to match the intended limit in GB.
    BUILD_DRAM_BUDGET_GB=$(echo "scale=2; $INTENDED_LIMIT / 1024 / 1024 / 1024" | bc)
    
    # Safety Check: If budget is < 1GB, check if it's too small for app. 
    # But for small datasets, it should be fine.
    
    echo "----------------------------------------------------------------"
    echo "Running LIMITED Memory DiskANN experiment for: $NAME"
    echo "  Data: $DATA_PATH"
    echo "  Dim: $DIM, N: $N"
    echo "  Calculated Index Size: $(($TOTAL_INDEX_SIZE / 1024 / 1024)) MB"
    echo "  Intended Memory Limit (Ratio $SEARCH_RAM_BUDGET_RATIO): $(($INTENDED_LIMIT / 1024 / 1024)) MB"
    echo "  Cgroup Limit (with buffer): $(($CGROUP_LIMIT_BYTES / 1024 / 1024)) MB"
    echo "  Internal Chunk Budget: $(($CHUNK_BUDGET_BYTES / 1024 / 1024)) MB"
    echo "  Dynamic PQ Sampling Rate: $PQ_SAMPLING_RATE"
    echo "  Command -M Argument: $BUILD_DRAM_BUDGET_GB GB"
    echo "----------------------------------------------------------------"
    
    # Run build_disk_index using systemd-run (No Sudo Required)
    # This creates a transient scope strictly for this process with the defined memory limits.
    UNIT_NAME="diskann_${NAME}_$$"

    # Clean up function to kill background processes on interrupt
    cleanup() {
        echo "Caught signal, cleaning up..."
        systemctl --user stop "$UNIT_NAME.scope" 2>/dev/null
        exit 1
    }
    trap cleanup SIGINT SIGTERM

    # Run the workload in the FOREGROUND
    systemd-run --user --scope --unit="$UNIT_NAME" \
        -p MemoryMax=${CGROUP_LIMIT_BYTES} \
        "$BUILD_DISK_INDEX" \
        --data_type float \
        --dist_fn l2 \
        --index_path_prefix "$OUTPUT_BASE/$NAME" \
        --data_path "$DATA_PATH" \
        -B 32 \
        -M "$BUILD_DRAM_BUDGET_GB" \
        -R "$R" \
        -L "$L" \
        -T "$THREADS" \
        --QD "$QD" \
        --search_ram_budget_ratio "$SEARCH_RAM_BUDGET_RATIO" \
        --chunk_bytes "$CHUNK_BUDGET_BYTES" \
        --pq_sampling_rate "$PQ_SAMPLING_RATE" \
        --detailed_stats 2>&1 | tee "$LOG_FILE"
        
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "FAILED: $NAME"
    else
        echo "SUCCESS: $NAME"
        
        # Parse and print summary
        SUMMARY=$(grep "Time for" "$LOG_FILE")
        {
            echo ""
            echo "Summary for $NAME:"
            echo "----------------------------------------"
            echo "$SUMMARY"
            echo "----------------------------------------"
        } | tee -a "$LOG_FILE"
    fi
}

echo "Starting DiskANN LIMITED Memory Batch Experiments..."
echo "Output Directory: $OUTPUT_BASE"


# Iterate over defined datasets
# for NAME in "${!DATA_QDS[@]}"; do
#     run_experiment "$NAME"
# done

run_experiment "text10M"
run_experiment "image10M"

echo "Batch completed."
