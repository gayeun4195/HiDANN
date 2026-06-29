#!/bin/bash

# ==========================================================================================
# calculate_cache_budget.sh
# 
# Purpose:
#   Calculates the number of nodes to cache (NUM_NODES_TO_CACHE) for DiskANN search
#   based on a theoretical "Optimized Memory Model" (tight allocation).
#   Also calculates the required "Relaxed Physical Limit" to prevent OOM kills
#   due to the structural inefficiency of the current DiskANN binary (fixed max buffers).
#
# Usage:
#   source calculate_cache_budget.sh
#   calculate_cache_and_physical_limit <N> <DIM> <R> <QD> <THREADS> <BEAM_WIDTH> <BUDGET_RATIO> <K>
#
# Outputs (Exported Variables):
#   CALCULATED_CACHE_NODES   : The number of nodes to pass to --num_nodes_to_cache
#   RELAXED_CGROUP_LIMIT     : The bytes to set for cgset memory.limit_in_bytes
#   STRICT_BUDGET_BYTES      : The theoretical 10% budget (for logging)
#   WASTE_BYTES              : The extra memory allowance (for logging)
# ==========================================================================================

calculate_cache_and_physical_limit() {
    local N=$1
    local DIM=$2
    local R=$3
    local QD=$4
    local THREADS=$5
    local BEAM_WIDTH=$6
    local RATIO=$7
    local K=$8
    
    # ----------------------------------------------------------------------
    # 1. Constants & Basic Sizes
    # ----------------------------------------------------------------------
    local SECTOR_LEN=4096
    local FLOAT_SIZE=4
    local UINT_SIZE=4
    local FULL_PRECISION_REORDER_MULTIPLIER=3
    
    # Aligned Dimension (Round up to multiple of 8)
    local ALIGNED_DIM=$(( ((DIM + 7) / 8) * 8 ))
    
    # Raw Data Size (Vector + Graph) estimation
    # Graph size = N * (R + 1) * 4 bytes (Adjacency + Count)
    local RAW_VEC_BYTES=$(echo "$N * $DIM * $FLOAT_SIZE" | bc)
    local GRAPH_BYTES=$(echo "$N * ($R + 1) * $UINT_SIZE" | bc)
    local TOTAL_DATA_BYTES=$(echo "$RAW_VEC_BYTES + $GRAPH_BYTES" | bc)
    
    # STRICT BUDGET (Target Logic)
    local MEMORY_BUDGET_BYTES=$(echo "$TOTAL_DATA_BYTES * $RATIO" | bc | cut -d. -f1)
    
    # Resident PQ Vectors (Always required)
    local RESIDENT_PQ_BYTES=$(echo "$N * $QD" | bc)
    
    # ----------------------------------------------------------------------
    # 2. Optimized Model Calculation (Theoretical Usage)
    #    Based on strict Python script logic
    # ----------------------------------------------------------------------
    
    # A. Per-Thread Fixed Scratch (Tighter than binary)
    #    - Query Buffers: (12 bytes per dim typically for T/float/rotated)
    #    - PQ Table Scratch: 256 * QD * 4 bytes
    #    - PQ Coord Scratch: QD * R * 1 byte
    #    - Dist Scratch: R * 4 bytes
    
    local OPT_QUERY_BUF=$(echo "$ALIGNED_DIM * 12" | bc)
    local OPT_PQ_TABLE=$(echo "256 * $QD * 4" | bc)
    local OPT_PQ_COORD=$(echo "$QD * $R" | bc)
    local OPT_DIST=$(echo "$R * 4" | bc)
    
    local PER_THREAD_FIXED_OPT=$(echo "$OPT_QUERY_BUF + $OPT_PQ_TABLE + $OPT_PQ_COORD + $OPT_DIST" | bc)
    
    # B. Per-Thread IO Scratch (Dynamic based on Beam/K/NodeSize)
    #    1. Calculate Sectors per Node (S)
    #       NodeSize = (AlignedDim * 4) + ((R + 1) * 4)
    local NODE_SIZE_BYTES=$(echo "($ALIGNED_DIM * 4) + (($R + 1) * 4)" | bc)
    local S=$(echo "($NODE_SIZE_BYTES + 4095) / 4096" | bc) # Ceil division
    
    #    2. Required Nodes needed in buffer
    #       Python logic: max(Beam, 3 * K) nodes
    local REQ_NODES=$(echo "$FULL_PRECISION_REORDER_MULTIPLIER * $K" | bc)
    if [ "$BEAM_WIDTH" -gt "$REQ_NODES" ]; then
        REQ_NODES=$BEAM_WIDTH
    fi
    
    #    3. Required IO Sectors
    local REQ_IO_SECTORS=$(echo "$REQ_NODES * $S" | bc)
    local PER_THREAD_IO_OPT=$(echo "$REQ_IO_SECTORS * $SECTOR_LEN" | bc)
    
    # C. Per-Thread Misc Overhead (Optimized)
    #    Includes Stack, small vectors, robin_set overheads.
    #    Safety margin: 256KB (Covers L-related small allocations and visited sets)
    local MISC_OVERHEAD=262144 
    
    local PER_THREAD_TOTAL_OPT=$(echo "$PER_THREAD_FIXED_OPT + $PER_THREAD_IO_OPT + $MISC_OVERHEAD" | bc)
    local TOTAL_THREAD_OVERHEAD_OPT=$(echo "$THREADS * $PER_THREAD_TOTAL_OPT" | bc)
    
    # ----------------------------------------------------------------------
    # 3. Calculate Cache Nodes (Using Optimized Model)
    # ----------------------------------------------------------------------
    local REMAINING_FOR_CACHE=$(echo "$MEMORY_BUDGET_BYTES - $RESIDENT_PQ_BYTES - $TOTAL_THREAD_OVERHEAD_OPT" | bc)
    
    # Cost Per Node for Caching (Resident Memory)
    # SIMULATION MODE: Assuming C++ code is patched to allocate only (R+1) integers instead of 512.
    local COST_PER_NODE=$(echo "$ALIGNED_DIM * 4 + ($R + 1) * 4 + 64" | bc)
    
    if [ "$REMAINING_FOR_CACHE" -le 0 ]; then
        CALCULATED_CACHE_NODES=0
        # Warn if negative
        echo "WARNING: Even with optimized model, budget is insufficient!" >&2
        echo "  Budget: $MEMORY_BUDGET_BYTES, Required: $((RESIDENT_PQ_BYTES + TOTAL_THREAD_OVERHEAD_OPT))" >&2
    else
        CALCULATED_CACHE_NODES=$(echo "$REMAINING_FOR_CACHE / $COST_PER_NODE" | bc)
        # Cap at N
        if [ "$CALCULATED_CACHE_NODES" -gt "$N" ]; then
            CALCULATED_CACHE_NODES=$N
        fi
    fi
    
    # ----------------------------------------------------------------------
    # 4. Calculate Physical Relaxation (The "Waste")
    #    The actual binary allocates MAX constants.
    # ----------------------------------------------------------------------
    local MAX_GRAPH_DEGREE=512
    local MAX_PQ_CHUNKS=512
    local MAX_N_SECTOR_READS=128
    
    # A. Actual Fixed Scratch (From Source Code)
    #    - Sector Scratch: 128 * 4KB = 524288
    #    - PQ Coords: 512 * 512 = 262144
    #    - PQ Table: 256 * 512 * 4 = 524288
    #    - Coord Scratch: AlignedDim * 4 (Variable but part of fixed alloc)
    #    - Query Buffers: AlignedDim * 12
    
    local ACTUAL_SECTOR_BUF=524288
    local ACTUAL_PQ_COORD=262144
    local ACTUAL_PQ_TABLE=524288
    # Note: Coord scratch logic in source: ROUND_UP(sizeof(T)*dim, 256)
    local ACTUAL_COORD_SCRATCH=$(echo "$ALIGNED_DIM * 4" | bc) 
    
    # Actual uses the same Misc Overhead assumption + Fixed Bloat
    local PER_THREAD_FIXED_ACTUAL=$(echo "$ACTUAL_SECTOR_BUF + $ACTUAL_PQ_COORD + $ACTUAL_PQ_TABLE + $ACTUAL_COORD_SCRATCH + $OPT_QUERY_BUF + $MISC_OVERHEAD" | bc)
    local TOTAL_THREAD_OVERHEAD_ACTUAL=$(echo "$THREADS * $PER_THREAD_FIXED_ACTUAL" | bc)
    
    # Waste = Actual - Optimized
    # Calculate Thread Waste
    local THREAD_WASTE=$(echo "$TOTAL_THREAD_OVERHEAD_ACTUAL - $TOTAL_THREAD_OVERHEAD_OPT" | bc)
    
    # Calculate Resident Cache Waste (Crucial for unpatched binary)
    # The binary allocates 512 neighbors per node, but we calculated budget using R.
    # We must provide physical memory for the difference (512 - R) so it doesn't crash.
    local MAX_GRAPH_DEGREE=512
    local NODE_WASTE_BYTES=0
    if [ "$MAX_GRAPH_DEGREE" -gt "$R" ]; then
        NODE_WASTE_BYTES=$(echo "$CALCULATED_CACHE_NODES * ($MAX_GRAPH_DEGREE - $R) * 4" | bc)
    fi
    
    local WASTE_BYTES=$(echo "$THREAD_WASTE + $NODE_WASTE_BYTES" | bc)
    
    # Safety check
    if [ "$(echo "$WASTE_BYTES < 0" | bc)" -eq 1 ]; then
        WASTE_BYTES=0
    fi
    
    # ----------------------------------------------------------------------
    # 5. Final Outputs
    # ----------------------------------------------------------------------
    RELAXED_CGROUP_LIMIT=$(echo "$MEMORY_BUDGET_BYTES + $WASTE_BYTES" | bc)
    STRICT_BUDGET_BYTES=$MEMORY_BUDGET_BYTES
    
    # Calculate Caching Percentage
    CALCULATED_CACHE_RATIO=$(echo "scale=4; ($CALCULATED_CACHE_NODES / $N) * 100" | bc)

    # Export for caller
    export CALCULATED_CACHE_NODES
    export CALCULATED_CACHE_RATIO
    export RELAXED_CGROUP_LIMIT
    export STRICT_BUDGET_BYTES
    export WASTE_BYTES
}
