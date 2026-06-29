#!/bin/bash

# SOGAIC Pipeline Script (Limited Memory Version)
# Implements: Partitioning (Normal) -> Distributed Build (Limited) -> Merging (Limited)

set -e

# Configuration
BUILD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../build"
APPS_DIR="$BUILD_DIR/apps"
UTILS_DIR="$BUILD_DIR/apps/utils"

SOGAIC_PARTITION="$UTILS_DIR/sogaic_partition_data"
BUILD_MEMORY_INDEX="$APPS_DIR/build_memory_index"
MERGE_SHARDS="$UTILS_DIR/merge_shards"

if [ ! -f "$SOGAIC_PARTITION" ]; then
    echo "Error: Tools not found. Please build the project first."
    exit 1
fi

# Default Parameters
DATA_TYPE="float"
R=32
L=50
THREADS=$(nproc)
SAMPLING_RATE=0 
MAX_OVERLAP=4.0
EPSILON=1.8
GAMMA_SLACK=1.0
PARTITION_BUDGET_FRACTION="${SOGAIC_PARTITION_BUDGET_FRACTION:-1.0}"
SOGAIC_CLEAN_TEMP="${SOGAIC_CLEAN_TEMP:-1}"
BUDGET_RATIO=0.1
CGROUP_NAME=""
MEMORY_BUDGET_BYTES="${SOGAIC_MEMORY_BUDGET_BYTES:-0}"

usage() {
    echo "Usage: $0 <data_path> <output_prefix> [options]"
    echo "Options:"
    echo "  --datatype <type>      Data type (float/int8/uint8) [default: float]"
    echo "  --budget-ratio <val>   Memory Budget Ratio (e.g. 0.1 for 10%) [default: 0.1]"
    echo "  --overlap <val>        Max overlap (Omega) [default: 4.0]"
    echo "  --epsilon <val>        Relaxation factor (Epsilon) [default: 1.8]"
    echo "  --gamma <val>          Gamma compatibility value [default: 1.0]"
    echo "  --partition-budget-fraction <val>  Fraction of memory budget used for Gamma sizing [default: ${PARTITION_BUDGET_FRACTION}]"
    echo "  --memory-budget-bytes <bytes>  Derived per-shard build memory budget"
    echo "  --sampling <val>       Sampling rate [default: 0]"
    echo "  --threads <val>        Number of threads [default: nproc]"
    echo "  --R <val>              Graph degree R [default: 32]"
    echo "  --L <val>              Build complexity L [default: 50]"
    echo "  --clean-temp <0|1>     Remove per-shard temporary files after a successful merge [default: ${SOGAIC_CLEAN_TEMP}]"
    echo "  --cgroup <name>        Cgroup name for limiting memory during Build/Merge"
    exit 1
}

if [ "$#" -lt 2 ]; then
    usage
fi

DATA_PATH=$1
OUTPUT_PREFIX_BASE=$2
shift 2

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --datatype) DATA_TYPE="$2"; shift ;;
        --budget-ratio) BUDGET_RATIO="$2"; shift ;;
        --overlap) MAX_OVERLAP="$2"; shift ;;
        --epsilon) EPSILON="$2"; shift ;;
        --gamma) GAMMA_SLACK="$2"; shift ;;
        --partition-budget-fraction) PARTITION_BUDGET_FRACTION="$2"; shift ;;
        --memory-budget-bytes) MEMORY_BUDGET_BYTES="$2"; shift ;;
        --sampling) SAMPLING_RATE="$2"; shift ;;
        --threads) THREADS="$2"; shift ;;
        --R) R="$2"; shift ;;
        --L) L="$2"; shift ;;
        --clean-temp) SOGAIC_CLEAN_TEMP="$2"; shift ;;
        --cgroup) CGROUP_NAME="$2"; shift ;;
        --chunk-budget|--chunk-bytes) CHUNK_BUDGET="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export OMP_NUM_THREADS=$THREADS

OUTPUT_DIR=$(dirname "$OUTPUT_PREFIX_BASE")
mkdir -p "$OUTPUT_DIR"
PREFIX_NAME=$(basename "$OUTPUT_PREFIX_BASE")
LOG_FILE="${OUTPUT_PREFIX_BASE}_sogaic.log"

SUB_R=$(awk "BEGIN {print int($R * 2 / 3)}")
if [ "$SUB_R" -lt 16 ]; then SUB_R=16; fi
if [ "$SUB_R" -gt "$R" ]; then SUB_R=$R; fi

echo "==========================================================" | tee "$LOG_FILE"
echo "SOGAIC Construction Pipeline (LIMITED MEMORY MODE)" | tee -a "$LOG_FILE"
echo "Data: $DATA_PATH" | tee -a "$LOG_FILE"
echo "Target Budget Ratio: ${BUDGET_RATIO}" | tee -a "$LOG_FILE"
echo "Memory Budget Bytes: ${MEMORY_BUDGET_BYTES}" | tee -a "$LOG_FILE"
echo "Partition Budget Fraction: ${PARTITION_BUDGET_FRACTION}" | tee -a "$LOG_FILE"
echo "Clean Temp Files: ${SOGAIC_CLEAN_TEMP}" | tee -a "$LOG_FILE"
echo "Cgroup Limit: ${CGROUP_NAME:-None}" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# 1. Partitioning (Run WITHOUT Limit to avoid OOM in high-block-size partitioner)
echo "[1/3] Running SOGAIC Partitioning (Limited Memory)..." | tee -a "$LOG_FILE"
START_TIME=$(date +%s)

CMD=( "$SOGAIC_PARTITION" "$DATA_TYPE" "$DATA_PATH" "$OUTPUT_PREFIX_BASE" \
    "$SAMPLING_RATE" "$BUDGET_RATIO" "$SUB_R" "$MAX_OVERLAP" "$EPSILON" "$GAMMA_SLACK" "${CHUNK_BUDGET:-0}" "$MEMORY_BUDGET_BYTES" "$PARTITION_BUDGET_FRACTION" )

if [ -n "$CGROUP_NAME" ]; then
    cgexec -g memory:$CGROUP_NAME "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
else
    "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

END_PARTITION=$(date +%s)
echo "Partitioning took $((END_PARTITION - START_TIME)) seconds." | tee -a "$LOG_FILE"

NUM_PARTITIONS_FILE="${OUTPUT_PREFIX_BASE}_num_partitions.txt"
if [ ! -f "$NUM_PARTITIONS_FILE" ]; then
    echo "Error: Partition count file not found. Partitioning likely failed." | tee -a "$LOG_FILE"
    exit 1
fi
NUM_PARTITIONS=$(cat "$NUM_PARTITIONS_FILE")
echo "  -> Calculated Number of Partitions: $NUM_PARTITIONS" | tee -a "$LOG_FILE"

# 2. Build Memory Index (Run WITH Limit)
echo "[2/3] Building Indices for Shards (Limited Memory)..." | tee -a "$LOG_FILE"

for ((i=0; i<NUM_PARTITIONS; i++)); do
    SHARD_DATA="${OUTPUT_PREFIX_BASE}_subshard-${i}.bin"
    SHARD_INDEX_PREFIX="${OUTPUT_PREFIX_BASE}_subshard-${i}"
    
    echo "  -> Building shard $i with R=$SUB_R..." | tee -a "$LOG_FILE"
    
    CMD=( "$BUILD_MEMORY_INDEX" \
        --data_type "$DATA_TYPE" \
        --dist_fn l2 \
        --data_path "$SHARD_DATA" \
        --index_path_prefix "$SHARD_INDEX_PREFIX" \
        -R "$SUB_R" \
        -L "$L" \
        -T "$THREADS" )

    if [ -n "$CGROUP_NAME" ]; then
        cgexec -g memory:$CGROUP_NAME "${CMD[@]}" >> "$LOG_FILE" 2>&1
    else
        "${CMD[@]}" >> "$LOG_FILE" 2>&1
    fi
done

END_BUILD=$(date +%s)
echo "Index Building took $((END_BUILD - END_PARTITION)) seconds." | tee -a "$LOG_FILE"

# 3. Merge Shards (Run WITH Limit)
echo "[3/3] Merging Shards (Limited Memory)..." | tee -a "$LOG_FILE"

MERGED_INDEX="${OUTPUT_PREFIX_BASE}_merged.index"
MERGED_MEDOIDS="${OUTPUT_PREFIX_BASE}_medoids.bin"

CMD=( "$MERGE_SHARDS" \
    "${OUTPUT_PREFIX_BASE}_subshard-" \
    "" \
    "${OUTPUT_PREFIX_BASE}_subshard-" \
    "_ids_uint32.bin" \
    "$NUM_PARTITIONS" \
    "$R" \
    "$MERGED_INDEX" \
    "$MERGED_MEDOIDS" )

if [ -n "$CGROUP_NAME" ]; then
     cgexec -g memory:$CGROUP_NAME "${CMD[@]}" >> "$LOG_FILE" 2>&1
else
     "${CMD[@]}" >> "$LOG_FILE" 2>&1
fi

END_MERGE=$(date +%s)
echo "Merging took $((END_MERGE - END_BUILD)) seconds." | tee -a "$LOG_FILE"

if [ "${SOGAIC_CLEAN_TEMP}" = "1" ]; then
    if [ ! -s "$MERGED_INDEX" ] || [ ! -s "$MERGED_MEDOIDS" ]; then
        echo "Error: refusing to clean SOGAIC temporary files because final merge outputs are missing or empty." | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "Cleaning SOGAIC per-shard temporary files..." | tee -a "$LOG_FILE"
    for ((i=0; i<NUM_PARTITIONS; i++)); do
        SHARD_PREFIX="${OUTPUT_PREFIX_BASE}_subshard-${i}"
        rm -f -- \
            "${SHARD_PREFIX}.bin" \
            "${SHARD_PREFIX}_ids_uint32.bin" \
            "${SHARD_PREFIX}" \
            "${SHARD_PREFIX}.data" \
            "${SHARD_PREFIX}.tags" \
            "${SHARD_PREFIX}_labels_to_medoids.txt"
    done
    echo "Finished cleaning SOGAIC per-shard temporary files." | tee -a "$LOG_FILE"
fi

TOTAL_TIME=$((END_MERGE - START_TIME))

# Extract Granular Times from Log
KMEANS_TIME=$(grep "K-Means Time:" "$LOG_FILE" | tail -n 1 | awk '{print $3}' | sed 's/s//')
ASSIGN_TIME=$(grep "Assignment Time:" "$LOG_FILE" | tail -n 1 | awk '{print $3}' | sed 's/s//')

# Shell measured times
PARTITION_TOTAL_TIME=$((END_PARTITION - START_TIME))
BUILD_TIME=$((END_BUILD - END_PARTITION))
MERGE_TIME=$((END_MERGE - END_BUILD))

echo "==========================================================" | tee -a "$LOG_FILE"
echo "               SOGAIC PIPELINE SUMMARY                    " | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"
echo "Dataset: $DATA_PATH" | tee -a "$LOG_FILE"
echo "Partitions: $NUM_PARTITIONS" | tee -a "$LOG_FILE"
echo "----------------------------------------------------------" | tee -a "$LOG_FILE"
echo "1. Partitioning (Total):    ${PARTITION_TOTAL_TIME} s" | tee -a "$LOG_FILE"
echo "   - K-Means Training:      ${KMEANS_TIME} s" | tee -a "$LOG_FILE"
echo "   - Assignment (Sharding): ${ASSIGN_TIME} s" | tee -a "$LOG_FILE"
echo "2. Index Construction:      ${BUILD_TIME} s" | tee -a "$LOG_FILE"
echo "3. Index Merging:           ${MERGE_TIME} s" | tee -a "$LOG_FILE"
echo "----------------------------------------------------------" | tee -a "$LOG_FILE"
echo "TOTAL PIPELINE TIME:        ${TOTAL_TIME} s" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"
echo "Final Index: $MERGED_INDEX" | tee -a "$LOG_FILE"
