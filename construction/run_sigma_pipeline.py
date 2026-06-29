#!/usr/bin/env python3

import argparse
import subprocess
import time
import re
import os
import sys
import fcntl
import struct
from pathlib import Path

GRAPH_SLACK_FACTOR = 1.3
OVERHEAD_FACTOR = 1.1
LINUX_STD_MUTEX_BYTES = 40
PTRDIFF_BYTES = 8

def env_flag_enabled(name, default=True):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value not in {"0", "false", "False", "no", "off"}

def read_fbin_metadata(path):
    with open(path, "rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"failed to read fbin metadata: {path}")
    npts, dim = struct.unpack("<II", raw)
    return npts, dim

def estimate_ram_usage(num_points, dim, datasize, degree):
    rounded_dim = ((dim + 7) // 8) * 8
    size_of_data = num_points * rounded_dim * datasize
    size_of_graph = num_points * degree * 4 * GRAPH_SLACK_FACTOR
    size_of_locks = num_points * LINUX_STD_MUTEX_BYTES
    size_of_outer_vector = num_points * PTRDIFF_BYTES
    return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks + size_of_outer_vector)

def required_uniform_partitions(npts, dim, degree, memory_budget_bytes, start_p):
    def pair_usage(p):
        max_part = (npts + p - 1) // p
        return 2.0 * estimate_ram_usage(max_part, dim, 4, degree), max_part

    hi = max(1, start_p)
    usage, _ = pair_usage(hi)
    while usage > memory_budget_bytes and hi < npts:
        hi = min(npts, hi * 2)
        usage, _ = pair_usage(hi)

    if usage > memory_budget_bytes:
        raise RuntimeError(
            "memory budget is too small even with one point per partition: "
            f"required_pair_ram={usage:.0f} bytes budget={memory_budget_bytes} bytes"
        )

    lo = 1
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        usage, _ = pair_usage(mid)
        if usage <= memory_budget_bytes:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    usage, max_part = pair_usage(best)
    return best, max_part, usage

def clear_file_cache(filepath):
    """
    Hints the OS to clear the page cache for the specified file using posix_fadvise.
    This helps simulate cold cache / limited memory scenarios.
    """
    if not env_flag_enabled("HIDANN_CONSTRUCTION_STAGE_CACHE_DROP", True):
        return
    if not env_flag_enabled("HIDANN_CONSTRUCTION_FADVISE", True):
        return
    if not filepath or not os.path.exists(filepath):
        return

    try:
        # Open file in read-only mode
        fd = os.open(filepath, os.O_RDONLY)
        try:
            # POSIX_FADV_DONTNEED: The specified data will not be accessed in the near future.
            # 0, 0 means the entire file.
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        print(f"[pipeline] Cleared page cache for: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"[pipeline] Warning: Failed to clear cache for {filepath}: {e}")

def run_step(cmd, step_name):
    print(f"\n[pipeline] Starting {step_name}...")
    print(f"[run] {' '.join(cmd)}")
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    full_output = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end='') # Stream to console
            full_output.append(line)

    rc = process.poll()
    end_time = time.time()
    duration = end_time - start_time

    if rc != 0:
        print(f"\n[pipeline] Error: {step_name} failed with exit code {rc}")
        sys.exit(rc)

    print(f"[pipeline] {step_name} finished in {duration:.2f} s")
    return "".join(full_output), duration

def parse_time_output(output, step):
    info = {}
    if step == "our_construction":
        # Look for "Compact (s): ..." and other timing
        # "Cross-partition total (s): 45.33"
        # "Sub-index build total (s): 15.67"
        patterns = {
            "Sub-index Build": r"Sub-index build total \(s\): ([\d\.]+)",
            "Self-subgraph Dump": r"Self-subgraph dump total \(s\): ([\d\.]+)",
            "Cross-partition Search": r"Cross-partition total \(s\): ([\d\.]+)",
            "Compaction": r"Compact \(s\): ([\d\.]+)"
        }
        for k, p in patterns.items():
            matches = re.findall(p, output)
            if matches:
                info[k] = sum(float(v) for v in matches)

    elif step == "our_pruning" or step == "our_pruning_final":
        # "time_total=123.45 s"
        # "pq_gen=1.23 s"
        m_total = re.search(r"time_total=([\d\.]+) s", output)
        if m_total:
            info["Total Pruning"] = float(m_total.group(1))

        m_pq = re.search(r"pq_gen=([\d\.]+) s", output)
        if m_pq:
            info["PQ Generation"] = float(m_pq.group(1))

    elif step == "reverse_edge_logic":
        # "total_runtime=26.00 s"
        m = re.search(r"total_runtime=([\d\.]+) s", output)
        if m:
            info["Total Reverse"] = float(m.group(1))

    return info

def remove_stale_outputs(prefix, detected_p, qd):
    if not env_flag_enabled("HIDANN_CLEAN_CONSTRUCTION_OUTPUTS", True):
        print("[pipeline] HIDANN_CLEAN_CONSTRUCTION_OUTPUTS=0; preserving existing construction outputs")
        return

    prefix_path = Path(prefix)
    parent = prefix_path.parent if str(prefix_path.parent) else Path(".")
    stem = prefix_path.name
    construction_prefix = f"{stem}_P{detected_p}."
    qd_suffix = f"_QD{qd}" if qd > 0 else ""
    pruning_prefix = f"{stem}_P{detected_p}{qd_suffix}"
    reversed_prefix = f"{pruning_prefix}_reversed."
    pq_prefix = f"{stem}{qd_suffix}"

    patterns = []
    for part in range(detected_p):
        patterns += [
            f"{stem}.subshard_u{part}.fbin",
            f"{stem}.subgraph_u{part}.mem.index",
            f"{stem}.subgraph_u{part}.mem.index.data",
            f"{stem}.subgraph_u{part}.mem.index.tags",
        ]
    patterns += [
        f"{construction_prefix}cand.idx",
        f"{construction_prefix}cand.ids",
        f"{construction_prefix}bucket_*.log",
        f"{pruning_prefix}.mem.index",
        f"{pruning_prefix}.mem.index.data",
        f"{pruning_prefix}.mem.index.tags",
        f"{reversed_prefix}cand.idx",
        f"{reversed_prefix}cand.ids",
        f"{pruning_prefix}_final.mem.index",
        f"{pruning_prefix}_final.mem.index.data",
        f"{pruning_prefix}_final.mem.index.tags",
        f"{pq_prefix}.pq_pivots.bin",
        f"{pq_prefix}.pq_compressed.bin",
    ]

    removed = 0
    for pattern in patterns:
        for path in parent.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
    if removed:
        print(f"[pipeline] Removed {removed} stale construction output file(s) for prefix {prefix}")

def main():
    parser = argparse.ArgumentParser(description="Run SIGMA Construction Pipeline")
    parser.add_argument("--data", required=True, help="Input data file (.fbin)")
    parser.add_argument("--prefix", required=True, help="Output prefix")
    parser.add_argument("--buckets", type=int, help="Number of buckets/partitions (Optional, auto-detected if omitted)")
    parser.add_argument("--R", type=int, default=32, help="Degree R")
    parser.add_argument("--L", type=int, required=True, help="Search L")
    parser.add_argument("--min-P", type=int, default=1, help="Minimum partitions for budget-based automatic P")
    parser.add_argument("--P", type=int, help="Legacy alias for --min-P; fixed P only when no memory budget is given")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads")
    parser.add_argument("--QD", type=int, default=0, help="PQ Quantization Dimension (required for new PQ)")
    parser.add_argument("--chunk-budget", type=int, default=0, help="Dynamic chunk budget in bytes")
    parser.add_argument("--memory-budget-bytes", type=int, default=0, help="Logical build memory budget in bytes")
    parser.add_argument("--cross-pairs-file", default="", help="Optional unordered i j pair allowlist for sparse cross-partition probing")
    parser.add_argument("--build-dir", default=os.environ.get("BUILD_DIR", ""),
                        help="Repository build directory. Uses BUILD_DIR when omitted.")
    parser.add_argument("--partition-budget-fraction", type=float,
                        default=float(os.environ.get("HIDANN_PARTITION_BUDGET_FRACTION", "1.0")),
                        help="Fraction of logical memory budget used for automatic P sizing")
    parser.add_argument("--pq-sampling-rate", type=float, default=0.05, help="PQ Sampling Rate (default 0.05)")

    args = parser.parse_args()

    print(
        "[pipeline] HIDANN_CONSTRUCTION_STAGE_CACHE_DROP="
        f"{1 if env_flag_enabled('HIDANN_CONSTRUCTION_STAGE_CACHE_DROP', True) else 0} "
        "HIDANN_CONSTRUCTION_FADVISE="
        f"{1 if env_flag_enabled('HIDANN_CONSTRUCTION_FADVISE', True) else 0} "
        "HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP="
        f"{1 if env_flag_enabled('HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP', True) else 0}"
    )

    # Pass chunk budget if set
    chunk_arg = []
    if args.chunk_budget > 0:
        chunk_arg = [f"--chunk-bytes={args.chunk_budget}"]
    cross_pairs_arg = []
    if args.cross_pairs_file:
        cross_pairs_arg = [f"--cross-pairs-file={args.cross_pairs_file}"]

    print(f"[pipeline] DEBUG: SIGMA_CHUNK_BYTES = {os.environ.get('SIGMA_CHUNK_BYTES', 'Not Set')}")
    print(f"[pipeline] DEBUG: DISKANN_CHUNK_BYTES = {os.environ.get('DISKANN_CHUNK_BYTES', 'Not Set')}")

    # Set OpenMP environment variables for performance if not already set
    if "OMP_PROC_BIND" not in os.environ:
        os.environ["OMP_PROC_BIND"] = "close"
        print("[pipeline] Auto-setting OMP_PROC_BIND=close")

    if "OMP_PLACES" not in os.environ:
        os.environ["OMP_PLACES"] = "cores"
        print("[pipeline] Auto-setting OMP_PLACES=cores")

    budget_messages = []
    if args.partition_budget_fraction <= 0.0 or args.partition_budget_fraction > 1.0:
        print("[pipeline] Error: --partition-budget-fraction must be in (0, 1].")
        sys.exit(1)

    min_P = max(1, args.min_P)
    if args.P is not None:
        min_P = max(min_P, args.P)
        if args.memory_budget_bytes > 0:
            budget_messages.append(
                f"[pipeline] Treating legacy --P={args.P} as a minimum partition count because "
                "--memory-budget-bytes was provided"
            )

    detected_P = None
    if args.memory_budget_bytes > 0:
        npts, dim = read_fbin_metadata(args.data)
        partition_budget_bytes = int(args.memory_budget_bytes * args.partition_budget_fraction)
        if partition_budget_bytes <= 0:
            print("[pipeline] Error: effective partition budget is zero.")
            sys.exit(1)
        required_P, _, _ = required_uniform_partitions(
            npts, dim, args.R, partition_budget_bytes, min_P
        )
        detected_P = max(min_P, required_P)
        if detected_P > min_P:
            budget_messages.append(
                f"[pipeline] Increasing P from min-P {min_P} to {detected_P} for memory budget "
                f"{partition_budget_bytes} bytes (partition fraction {args.partition_budget_fraction})"
            )
        else:
            budget_messages.append(
                f"[pipeline] Using min-P {min_P}; it already satisfies the partition budget"
            )

        max_part = (npts + detected_P - 1) // detected_P
        estimated_pair_ram = 2.0 * estimate_ram_usage(max_part, dim, 4, args.R)
        budget_messages.append(
            f"[pipeline] HiDANN budget check: N={npts} D={dim} P={detected_P} "
            f"max_partition={max_part} estimated_two_index_ram={estimated_pair_ram:.0f} bytes "
            f"logical_budget={args.memory_budget_bytes} bytes "
            f"partition_budget={partition_budget_bytes} bytes "
            f"partition_fraction={args.partition_budget_fraction}"
        )
    else:
        if args.P is not None:
            detected_P = args.P
        elif args.buckets is not None:
            detected_P = args.buckets
        elif args.min_P > 1:
            detected_P = args.min_P
        else:
            print("[pipeline] Error: pass --memory-budget-bytes for automatic P, or pass --P/--min-P for manual P.")
            sys.exit(1)

    if detected_P is None:
        print("[pipeline] Error: failed to determine P.")
        sys.exit(1)

    memory_arg = []
    if args.memory_budget_bytes > 0:
        memory_arg = [f"--memory-budget-bytes={args.memory_budget_bytes}"]

    # Executables are in the same directory as this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Prefer the requested build directory, then the conventional repository build.
    requested_build = args.build_dir.strip()
    if requested_build:
        build_dir = os.path.join(os.path.abspath(requested_build), "construction")
    else:
        build_dir = os.path.join(os.path.dirname(base_dir), "build", "construction")
    if os.path.exists(build_dir):
        const_dir = build_dir
        print(f"[pipeline] Using executables from build directory: {const_dir}")
    else:
        const_dir = base_dir
        print(f"[pipeline] Using executables from source directory: {const_dir}")

    exe_construction = os.path.join(const_dir, "our_construction")
    exe_pruning = os.path.join(const_dir, "our_pruning")
    exe_reverse = os.path.join(const_dir, "reverse_edge_logic")
    exe_exact = os.path.join(const_dir, "ablation_exact_pruning")

    # Check executables
    for exe in [exe_construction, exe_pruning, exe_reverse]:
        if not os.path.exists(exe):
            print(f"Error: Executable not found: {exe}")
            sys.exit(1)

    # Setup Logging
    log_file = f"{args.prefix}_P{detected_P}_QD{args.QD}_pipeline.log"
    print(f"[pipeline] Logging execution to: {log_file}")

    class Tee(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Tee(log_file)
    sys.stderr = sys.stdout # Redirect stderr to same logger

    for msg in budget_messages:
        print(msg)

    timing_summary = {}

    # Naming logic:
    # 1. Construction output: {prefix}.cand.idx (Generic, no QD)
    # 2. Pruning output: {prefix}_QD{QD}.mem.index
    # 3. Reverse output: {prefix}_QD{QD}_reversed.cand.idx
    # 4. Final output: {prefix}_QD{QD}_final.mem.index

    # Step 1: Construction. Split subgraph build and pair processing into
    # separate processes so allocator/thread-local state from subgraph builds
    # cannot carry into the memory-critical cross-partition phase.
    cmd1_base = [
        exe_construction,
        args.data,
        args.prefix,
    ]

    # Now construction cmd:
    # argv[3] is P.
    cmd1_base.append(str(detected_P))
    cmd1_base.append(str(args.R))
    cmd1_base.append(str(args.L))
    cmd1_base.append(str(args.threads))

    # Flag --buckets handling
    # If args.buckets WAS provided explicitly, we pass it.
    # If NOT provided, we omit it, letting C++ calc B.
    if args.buckets is not None:
        cmd1_base.append(f"--buckets={args.buckets}")

    construction_prefix = f"{args.prefix}_P{detected_P}."
    remove_stale_outputs(args.prefix, detected_P, args.QD)

    cmd1_base.append(f"--prefix={construction_prefix}")
    cmd1_base += chunk_arg + memory_arg

    cmd1_build = cmd1_base + ["--stage=build-subgraphs"]
    out1_build, dur1_build = run_step(cmd1_build, "our_construction build-subgraphs")

    # Clean cache for build inputs/outputs before starting the pair-processing
    # process in the same cgroup.
    clear_file_cache(args.data)
    for part in range(detected_P):
        sub_prefix = f"{args.prefix}.subgraph_u{part}.mem.index"
        clear_file_cache(sub_prefix)
        clear_file_cache(sub_prefix + ".data")

    pair_outputs = []
    dur1_pair = 0.0

    cmd1_self = cmd1_base + ["--stage=self-dump"]
    out_self, dur_self = run_step(cmd1_self, "our_construction pair-processing")
    pair_outputs.append(out_self)
    dur1_pair += dur_self

    for anchor in range(max(detected_P - 1, 0)):
        cmd1_cross = cmd1_base + ["--stage=cross-pairs", f"--anchor={anchor}"] + cross_pairs_arg
        out_cross, dur_cross = run_step(cmd1_cross, "our_construction pair-processing")
        pair_outputs.append(out_cross)
        dur1_pair += dur_cross

    cmd1_compact = cmd1_base + ["--stage=compact"]
    out_compact, dur_compact = run_step(cmd1_compact, "our_construction pair-processing")
    pair_outputs.append(out_compact)
    dur1_pair += dur_compact
    out1_pair = "".join(pair_outputs)

    # Clean cache for inputs/outputs of step 1
    clear_file_cache(args.data)
    # Construction output is basically args.prefix + ".cand.idx", but let's be safe
    # Based on naming logic, construction_prefix was used.
    # construction/our_construction.cpp writes: {prefix}.cand.idx and {prefix}.cand.ids
    clear_file_cache(construction_prefix + "cand.idx")
    clear_file_cache(construction_prefix + "cand.ids")

    # Now we need to parse what B actually used was, because maybe P != B?
    # B is internal to construction buffering.

    # Naming for subsequent steps:
    qd_suffix = f"_QD{args.QD}" if args.QD > 0 else ""
    # Per user request: "pruning 이후 파일들에 필요해"
    # Pruning output: {prefix}_QD{N}.mem.index

    pruning_prefix = f"{args.prefix}_P{detected_P}{qd_suffix}"
    intermediate_graph = f"{pruning_prefix}.mem.index"

    # PQ files: {prefix}_QD{QD}.pq_pivots.bin (No P, per user request)
    pq_prefix = f"{args.prefix}{qd_suffix}"
    pq_pivots = f"{pq_prefix}.pq_pivots.bin"
    pq_codes = f"{pq_prefix}.pq_compressed.bin"

    timing_summary["Construction Build"] = (dur1_build, parse_time_output(out1_build, "our_construction"))
    timing_summary["Construction Pair"] = (dur1_pair, parse_time_output(out1_pair, "our_construction"))

    # Step 2: Pruning (1st pass)
    cmd2 = [
        exe_pruning,
        f"--data={args.data}",
        f"--cand-prefix={construction_prefix}", # Reads generated cand.idx (no QD)
        f"--out={intermediate_graph}",         # Writes with QD
        f"--R={args.R}",
        f"--P={detected_P}",
        f"--threads={args.threads}",
        f"--QD={args.QD}",
        f"--pivots={pq_pivots}", # Verify: write capability
        f"--codes={pq_codes}",
        f"--pval={args.pq_sampling_rate}"
    ] + chunk_arg + memory_arg
    out2, dur2 = run_step(cmd2, "our_pruning (1st)")
    timing_summary["Pruning (1st)"] = (dur2, parse_time_output(out2, "our_pruning"))

    # Clean cache for step 2
    # Inputs: cand.idx, ids (already cleared above?), pq pivots
    clear_file_cache(construction_prefix + "cand.idx")
    clear_file_cache(construction_prefix + "cand.ids")
    clear_file_cache(pq_pivots)
    clear_file_cache(pq_codes)
    # Output: intermediate_graph
    clear_file_cache(intermediate_graph)

    reversed_prefix = f"{pruning_prefix}_reversed."

    # Step 3: Reverse Edge Logic
    cmd3 = [
        exe_reverse,
        f"--graph={intermediate_graph}",       # Reads QD graph
        f"--data={args.data}",
        f"--out-prefix={reversed_prefix}",     # Writes QD_reversed.cand.idx
        f"--P={detected_P}",
        f"--R={args.R}",
    ] + memory_arg
    out3, dur3 = run_step(cmd3, "reverse_edge_logic")
    timing_summary["Reverse Logic"] = (dur3, parse_time_output(out3, "reverse_edge_logic"))

    # Clean cache for step 3
    # Inputs: intermediate_graph
    clear_file_cache(intermediate_graph)
    # Output: reversed candidates
    clear_file_cache(reversed_prefix + "cand.idx")
    clear_file_cache(reversed_prefix + "cand.ids")

    final_graph = f"{pruning_prefix}_final.mem.index"

    # Step 4: Pruning (2nd pass)
    cmd4 = [
        exe_pruning,
        f"--data={args.data}",
        f"--cand-prefix={reversed_prefix}",     # Reads reversed cand
        f"--out={final_graph}",                 # Writes final QD graph
        f"--R={args.R}",
        f"--P={detected_P}",
        f"--threads={args.threads}",
        f"--reuse=1",
        f"--pivots={pq_pivots}",      # Reuses pivots from step 2
        f"--codes={pq_codes}",    # same
        f"--QD={args.QD}"
    ] + chunk_arg + memory_arg
    out4, dur4 = run_step(cmd4, "our_pruning (final)")
    timing_summary["Pruning (Final)"] = (dur4, parse_time_output(out4, "our_pruning_final"))

    # Clean cache for step 4
    clear_file_cache(reversed_prefix + "cand.idx")
    clear_file_cache(reversed_prefix + "cand.ids")
    # Final output usually we might want to keep in cache for subsequent usage?
    # But user asked for limited memory simulation, so let's clear it too or leave it?
    # Usually we leave the final result "as is". If they process it further, they read it.
    # Let's clear it to be consistent with "end of pipeline" state.
    clear_file_cache(final_graph)

    print("\n" + "="*60)
    print(f"{'PIPELINE EXECUTION SUMMARY':^60}")
    print("="*60)
    print(f"{'Step':<25} | {'Wall Time (s)':<15} | {'Internal Metrics':<20}")
    print("-" * 60)

    total_pipeline_time = 0
    for step, (dur, metrics) in timing_summary.items():
        total_pipeline_time += dur
        metric_str = ", ".join([f"{k}={v:.2f}s" for k,v in metrics.items()])
        print(f"{step:<25} | {dur:<15.2f} | {metric_str}")

    print("-" * 60)
    print(f"{'Total Pipeline Time':<25} | {total_pipeline_time:<15.2f} |")
    print("="*60)
    print(f"\nFinal Graph: {final_graph}")

if __name__ == "__main__":
    main()
