#!/bin/bash

# ==========================================================================================
# calculate_cache_budget_optimized.sh (DiskANN Optimized Version)
# 
# Purpose:
#   Calculates the number of nodes to cache (NUM_NODES_TO_CACHE) for DiskANN search
#   based on a SIGMA-aligned "Optimized Memory Model".
#   
#   Key Logic Changes vs Standard:
#   1. Logical Budget (for Caching): 
#      - IGNORES thread overheads (Fixed Scratch & IO Buffers) to maximize cache.
#      - This aligns with SIGMA optimized logic where Graph IO is ignored.
#      - SUBTRACTS Medoids memory (Resident).
#
#   2. Physical Limit (Safety):
#      - Adds BACK all actual thread overheads (Fixed + IO) + Medoids + Safety Margin.
#      - Ensures no OOM kill occurs despite the "over-committed" cache.
#
# Usage:
#   source calculate_cache_budget_optimized.sh
#   calculate_cache_and_physical_limit <N> <DIM> <R> <QD> <THREADS> <BEAM_WIDTH> <BUDGET_RATIO> <K> <P>
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
    local P=${9:-1} # Default to 1 medoid if not provided
    
    # ----------------------------------------------------------------------
    # 1. Constants & Basic Sizes
    # ----------------------------------------------------------------------
    local SECTOR_LEN=4096
    local FLOAT_SIZE=4
    local UINT_SIZE=4
    local FULL_PRECISION_REORDER_MULTIPLIER=3
    
    # Aligned Dimension (Round up to multiple of 8)
    local ALIGNED_DIM=$(( ((DIM + 7) / 8) * 8 ))
    
    # Cost Per Node for Caching (DiskANN Resident Memory Model)
    # Assuming DiskANN patched or standard allocation:
    # 4 bytes per dim (float) + (R+1)*4 bytes (neighbors) + Overhead (say 64 bytes for vector/overhead)
    local COST_PER_NODE=$(echo "$ALIGNED_DIM * 4 + ($R + 1) * 4 + 64" | bc)
    
    # Raw Data Size (Vector + Graph) estimation
    local RAW_VEC_BYTES=$(echo "$N * $DIM * $FLOAT_SIZE" | bc)
    local GRAPH_BYTES=$(echo "$N * ($R + 1) * $UINT_SIZE" | bc)
    local TOTAL_DATA_BYTES=$(echo "$RAW_VEC_BYTES + $GRAPH_BYTES" | bc)
    
    # TARGET BUDGET
    local MEMORY_BUDGET_BYTES=$(echo "$TOTAL_DATA_BYTES * $RATIO" | bc | cut -d. -f1)
    
    # ----------------------------------------------------------------------
    # 2. Resident Overhead Calculation
    # ----------------------------------------------------------------------
    
    # A. Resident PQ Vectors (Always required)
    local RESIDENT_PQ_BYTES=$(echo "$N * $QD" | bc)
    
    # B. Resident Medoids (Always required)
    #    Cost = P * COST_PER_NODE
    local RESIDENT_MEDOID_BYTES=$(echo "$P * $COST_PER_NODE" | bc)
    
    # ----------------------------------------------------------------------
    # 3. Logical Cache Calculation (OPTIMIZED - Ignore Thread Overhead)
    #    We subtract ONLY Resident items. Thread buffers (IO/Fixed) are ignored.
    # ----------------------------------------------------------------------
    
    local REMAINING_FOR_CACHE=$(echo "$MEMORY_BUDGET_BYTES - $RESIDENT_PQ_BYTES - $RESIDENT_MEDOID_BYTES" | bc)
    
    if [ "$REMAINING_FOR_CACHE" -le 0 ]; then
        CALCULATED_CACHE_NODES=0
        echo "WARNING: Budget insufficient even for resident structures!" >&2
    else
        CALCULATED_CACHE_NODES=$(echo "$REMAINING_FOR_CACHE / $COST_PER_NODE" | bc)
        # Cap at N
        if [ "$CALCULATED_CACHE_NODES" -gt "$N" ]; then
            CALCULATED_CACHE_NODES=$N
        fi
    fi
    
    # ----------------------------------------------------------------------
    # 4. Physical Relaxation (The "Waste" / Safety Limit)
    #    We must provide physical memory for EVERYTHING we ignored + Actual Overheads.
    # ----------------------------------------------------------------------
    
    # A. Actual Fixed Scratch (From Source Code / Previous Analysis)
    #    - Sector Scratch: 128 * 4KB = 524288
    #    - PQ Coords: 512 * 512 = 262144 (approx, depends on QD but max 512)
    #    - PQ Table: 256 * 512 * 4 = 524288
    #    - Coord Scratch: AlignedDim * 4
    #    - Query Buffers: AlignedDim * 12
    #    - Misc Overhead: 256KB
    
    local ACTUAL_SECTOR_BUF=524288
    local ACTUAL_PQ_COORD=262144
    local ACTUAL_PQ_TABLE=524288
    local ACTUAL_COORD_SCRATCH=$(echo "$ALIGNED_DIM * 4" | bc)
    local OPT_QUERY_BUF=$(echo "$ALIGNED_DIM * 12" | bc)
    local MISC_OVERHEAD=262144
    
    local PER_THREAD_FIXED_ACTUAL=$(echo "$ACTUAL_SECTOR_BUF + $ACTUAL_PQ_COORD + $ACTUAL_PQ_TABLE + $ACTUAL_COORD_SCRATCH + $OPT_QUERY_BUF + $MISC_OVERHEAD" | bc)
    
    # B. Actual IO Scratch (Dynamic based on Beam/K)
    #    1. Calculate Sectors per Node (S)
    local NODE_SIZE_BYTES=$(echo "($ALIGNED_DIM * 4) + (($R + 1) * 4)" | bc)
    local S=$(echo "($NODE_SIZE_BYTES + 4095) / 4096" | bc) # Ceil division
    
    #    2. Required Nodes needed in buffer
    local REQ_NODES=$(echo "$FULL_PRECISION_REORDER_MULTIPLIER * $K" | bc)
    if [ "$BEAM_WIDTH" -gt "$REQ_NODES" ]; then
        REQ_NODES=$BEAM_WIDTH
    fi
    
    #    3. Required IO Sectors (This IS the DiskANN IO Buffer)
    local REQ_IO_SECTORS=$(echo "$REQ_NODES * $S" | bc)
    local PER_THREAD_IO_ACTUAL=$(echo "$REQ_IO_SECTORS * $SECTOR_LEN" | bc)
    
    # Total Thread Overhead
    local TOTAL_THREAD_OVERHEAD_ACTUAL=$(echo "$THREADS * ($PER_THREAD_FIXED_ACTUAL + $PER_THREAD_IO_ACTUAL)" | bc)
    
    # C. Node Waste (If unpatched binary uses 512 degree but R is smaller)
    #    Assuming we want to be safe against unpatched binaries.
    local MAX_GRAPH_DEGREE=512
    local NODE_WASTE_BYTES=0
    if [ "$MAX_GRAPH_DEGREE" -gt "$R" ]; then
        NODE_WASTE_BYTES=$(echo "$CALCULATED_CACHE_NODES * ($MAX_GRAPH_DEGREE - $R) * 4" | bc)
    fi
    
    # ----------------------------------------------------------------------
    # 5. Final Outputs
    # ----------------------------------------------------------------------
    
    # Safety Margin (Matched to SIGMA Optimized: 100MB)
    local SAFETY_MARGIN_BYTES=104857600
    
    # Relaxed Limit = Budget (filled with Cache + PQ + Medoids) + Thread Overhead + Node Waste + Safety
    # Note: MEMORY_BUDGET_BYTES includes the cache part.
    # We add the "IGORED" parts back.
    # Total Physical = ResPQ + ResMedoids + Cache + ThreadOverhead + NodeWaste + Safety
    # But Cache was calced as (Budget - ResPQ - ResMedoids).
    # So Cache + ResPQ + ResMedoids = Budget (approx, max).
    # Thus: Limit = Budget + ThreadOverhead + NodeWaste + Safety.
    
    RELAXED_CGROUP_LIMIT=$(echo "$MEMORY_BUDGET_BYTES + $TOTAL_THREAD_OVERHEAD_ACTUAL + $NODE_WASTE_BYTES + $SAFETY_MARGIN_BYTES" | bc)
    STRICT_BUDGET_BYTES=$MEMORY_BUDGET_BYTES
    WASTE_BYTES=$(echo "$TOTAL_THREAD_OVERHEAD_ACTUAL + $NODE_WASTE_BYTES + $SAFETY_MARGIN_BYTES" | bc)
    
    # Calculate Caching Percentage
    CALCULATED_CACHE_RATIO=$(echo "scale=4; ($CALCULATED_CACHE_NODES / $N) * 100" | bc)

    # Export for caller
    export CALCULATED_CACHE_NODES
    export CALCULATED_CACHE_RATIO
    export RELAXED_CGROUP_LIMIT
    export STRICT_BUDGET_BYTES
    export WASTE_BYTES
}
