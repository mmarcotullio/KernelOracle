#!/usr/bin/env python3
"""
collect_traces.py - Collect Linux scheduling traces using ftrace
(the same tracepoints as Google SchedViz) for training TCN/LSTM models.

Captures sched_switch, sched_wakeup, sched_wakeup_new, and sched_migrate_task
events while running configurable workloads (hackbench, sysbench, custom CPU/IO).
Outputs enriched CSV datasets with train/test splits.

Usage:
    # Collect all workloads (requires root):
    sudo python3 collect_traces.py collect --output-dir traces

    # Collect specific workload:
    sudo python3 collect_traces.py collect --output-dir traces -w hackbench_pipe_small

    # List available workloads:
    python3 collect_traces.py collect --list-workloads

    # Parse raw traces and build train/test CSVs (no root needed):
    python3 collect_traces.py build --output-dir traces

    # Do everything end-to-end:
    sudo python3 collect_traces.py all --output-dir traces
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TRACEFS_CANDIDATES = [
    "/sys/kernel/tracing",
    "/sys/kernel/debug/tracing",
]

SCHED_TRACEPOINTS = [
    "sched_switch",
    "sched_wakeup",
    "sched_wakeup_new",
    "sched_migrate_task",
]

# Regex to parse sched_switch events from ftrace output.
# Example line:
#   <idle>-0  [002] d... 12345.678901: sched_switch: prev_comm=swapper/2
#     prev_pid=0 prev_prio=120 prev_state=R ==> next_comm=bash
#     next_pid=1234 next_prio=120
SCHED_SWITCH_RE = re.compile(
    r'\[(?P<cpu>\d+)\]'
    r'.+?'
    r'(?P<timestamp>\d+\.\d+):\s+'
    r'sched_switch:\s+'
    r'prev_comm=(?P<prev_comm>.+?)\s+'
    r'prev_pid=(?P<prev_pid>\d+)\s+'
    r'prev_prio=(?P<prev_prio>\d+)\s+'
    r'prev_state=(?P<prev_state>\S+)\s+'
    r'==>\s+'
    r'next_comm=(?P<next_comm>.+?)\s+'
    r'next_pid=(?P<next_pid>\d+)\s+'
    r'next_prio=(?P<next_prio>\d+)'
)

DEFAULT_BUFFER_SIZE_KB = 16384  # 16 MB per CPU
DEFAULT_RUNS = 3

CSV_COLUMNS = [
    "timestamp", "time_diff", "task_name", "pid", "cpu",
    "prev_state", "prev_comm", "workload_type", "trace_id",
]


# ──────────────────────────────────────────────────────────────────────────────
# Workload Definitions
#
# Each workload generates a different scheduling pattern. Categories are used
# for the train/test split: one entire category can be held out for
# "unseen workload" generalization testing.
# ──────────────────────────────────────────────────────────────────────────────

WORKLOADS = {
    # --- hackbench: scheduler stress test (many context switches) ---
    "hackbench_pipe_small": {
        "cmd": ["hackbench", "-p", "-g", "4", "-l", "3000"],
        "desc": "Hackbench: pipes, 4 groups, 3000 loops",
        "category": "hackbench",
        "tool": "hackbench",
    },
    "hackbench_pipe_large": {
        "cmd": ["hackbench", "-p", "-g", "8", "-l", "5000"],
        "desc": "Hackbench: pipes, 8 groups, 5000 loops",
        "category": "hackbench",
        "tool": "hackbench",
    },
    "hackbench_socket_small": {
        "cmd": ["hackbench", "-g", "4", "-l", "3000"],
        "desc": "Hackbench: sockets, 4 groups, 3000 loops",
        "category": "hackbench",
        "tool": "hackbench",
    },
    "hackbench_socket_large": {
        "cmd": ["hackbench", "-g", "8", "-l", "5000"],
        "desc": "Hackbench: sockets, 8 groups, 5000 loops",
        "category": "hackbench",
        "tool": "hackbench",
    },

    # --- sysbench: CPU and thread benchmarks ---
    "sysbench_cpu_4t": {
        "cmd": ["sysbench", "cpu", "--threads=4", "--time=10", "run"],
        "desc": "Sysbench CPU: 4 threads, 10s",
        "category": "sysbench_cpu",
        "tool": "sysbench",
    },
    "sysbench_cpu_8t": {
        "cmd": ["sysbench", "cpu", "--threads=8", "--time=10", "run"],
        "desc": "Sysbench CPU: 8 threads, 10s",
        "category": "sysbench_cpu",
        "tool": "sysbench",
    },
    "sysbench_threads_8t": {
        "cmd": ["sysbench", "threads", "--threads=8", "--time=10", "run"],
        "desc": "Sysbench threads: 8 threads, 10s",
        "category": "sysbench_threads",
        "tool": "sysbench",
    },
    "sysbench_threads_16t": {
        "cmd": ["sysbench", "threads", "--threads=16", "--time=10", "run"],
        "desc": "Sysbench threads: 16 threads, 10s",
        "category": "sysbench_threads",
        "tool": "sysbench",
    },
    "sysbench_memory_4t": {
        "cmd": ["sysbench", "memory", "--threads=4", "--time=10", "run"],
        "desc": "Sysbench memory: 4 threads, 10s",
        "category": "sysbench_memory",
        "tool": "sysbench",
    },

    # --- Built-in CPU-bound workloads (no external tools needed) ---
    "cpu_bound_4p": {
        "cmd": "builtin_cpu",
        "num_procs": 4,
        "duration": 10,
        "desc": "CPU-bound: 4 processes, 10s",
        "category": "cpu_bound",
        "tool": None,
    },
    "cpu_bound_8p": {
        "cmd": "builtin_cpu",
        "num_procs": 8,
        "duration": 10,
        "desc": "CPU-bound: 8 processes, 10s",
        "category": "cpu_bound",
        "tool": None,
    },

    # --- Mixed I/O + CPU workload ---
    "io_mixed": {
        "cmd": "builtin_io_mixed",
        "duration": 10,
        "desc": "Mixed I/O + CPU, 10s",
        "category": "io_mixed",
        "tool": None,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Signal handling — clean up ftrace on Ctrl+C
# ──────────────────────────────────────────────────────────────────────────────

_tracefs_global = None


def _signal_handler(sig, frame):
    print("\nInterrupted! Cleaning up ftrace...")
    if _tracefs_global:
        try:
            cleanup_ftrace(_tracefs_global)
        except Exception:
            pass
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ──────────────────────────────────────────────────────────────────────────────
# ftrace helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_tracefs():
    """Locate the tracefs mount point."""
    for path in TRACEFS_CANDIDATES:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "trace")):
            return path
    return None


def write_tracefs(tracefs, filename, value):
    """Write a value to a tracefs control file."""
    filepath = os.path.join(tracefs, filename)
    with open(filepath, "w") as f:
        f.write(str(value))


def read_tracefs(tracefs, filename):
    """Read contents of a tracefs file."""
    filepath = os.path.join(tracefs, filename)
    with open(filepath, "r") as f:
        return f.read()


def setup_ftrace(tracefs, buffer_size_kb):
    """Configure ftrace for scheduling trace collection.

    Mirrors the setup in Google SchedViz's trace.sh: enables the four
    sched tracepoints into a per-CPU ring buffer in nooverwrite mode.
    """
    # Stop any active tracing
    write_tracefs(tracefs, "tracing_on", "0")

    # Clear the trace buffer
    write_tracefs(tracefs, "trace", "")

    # Use nop tracer (we only want events, not function tracing)
    write_tracefs(tracefs, "current_tracer", "nop")

    # Set buffer size per CPU
    write_tracefs(tracefs, "buffer_size_kb", str(buffer_size_kb))

    # Nooverwrite mode (like SchedViz) — drop new events if buffer is full
    # rather than silently losing old ones
    write_tracefs(tracefs, "trace_options", "nooverwrite")

    # Disable all events first
    write_tracefs(tracefs, "events/enable", "0")

    # Enable the 4 SchedViz tracepoints
    for tp in SCHED_TRACEPOINTS:
        write_tracefs(tracefs, f"events/sched/{tp}/enable", "1")


def start_tracing(tracefs):
    """Start ftrace recording."""
    write_tracefs(tracefs, "tracing_on", "1")


def stop_tracing(tracefs):
    """Stop ftrace recording."""
    write_tracefs(tracefs, "tracing_on", "0")


def read_trace_buffer(tracefs):
    """Read the entire trace buffer."""
    return read_tracefs(tracefs, "trace")


def cleanup_ftrace(tracefs):
    """Disable tracepoints and clear buffer."""
    stop_tracing(tracefs)
    for tp in SCHED_TRACEPOINTS:
        try:
            write_tracefs(tracefs, f"events/sched/{tp}/enable", "0")
        except Exception:
            pass
    try:
        write_tracefs(tracefs, "trace", "")
    except Exception:
        pass


# ──────────────────────
# Built-in workloads 
# ──────────────────────

def _cpu_burn(duration):
    """Tight CPU loop for the given number of seconds."""
    end_time = time.time() + duration
    x = 1.0
    while time.time() < end_time:
        for _ in range(1000):
            x = math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1)


def run_cpu_bound(num_procs, duration):
    """Spawn multiple processes doing tight CPU computation."""
    procs = []
    for _ in range(num_procs):
        p = multiprocessing.Process(target=_cpu_burn, args=(duration,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def run_io_mixed(duration):
    """Run mixed I/O + CPU workload using dd and CPU burn concurrently."""
    # Start dd processes for I/O pressure
    dd_procs = []
    for i in range(2):
        tmpfile = tempfile.mktemp(prefix=f"trace_io_{i}_")
        dd = subprocess.Popen(
            ["dd", "if=/dev/urandom", f"of={tmpfile}", "bs=4K",
             f"count={duration * 250}", "conv=fdatasync"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        dd_procs.append((dd, tmpfile))

    # Start CPU burn concurrently
    cpu_procs = []
    for _ in range(2):
        p = multiprocessing.Process(target=_cpu_burn, args=(duration,))
        p.start()
        cpu_procs.append(p)

    # Wait for everything
    for p in cpu_procs:
        p.join()
    for dd, tmpfile in dd_procs:
        dd.wait()
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Workload runner
# ──────────────────────────────────────────────────────────────────────────────

def tool_available(name):
    """Check if an external tool is on PATH."""
    return shutil.which(name) is not None


def run_workload(name, config):
    """Execute a workload. Returns True on success, False if skipped."""
    tool = config.get("tool")
    if tool and not tool_available(tool):
        print(f"  SKIP: '{tool}' not installed "
              f"(try: sudo apt install {'rt-tests' if tool == 'hackbench' else tool})")
        return False

    cmd = config["cmd"]

    if cmd == "builtin_cpu":
        n, d = config["num_procs"], config["duration"]
        print(f"  Running built-in CPU workload: {n} processes, {d}s")
        run_cpu_bound(n, d)
    elif cmd == "builtin_io_mixed":
        d = config["duration"]
        print(f"  Running built-in mixed I/O workload: {d}s")
        run_io_mixed(d)
    else:
        print(f"  Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                print(f"  WARNING: exited with code {result.returncode}")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            print("  WARNING: workload timed out after 300s")
            return False
        except FileNotFoundError:
            print(f"  SKIP: command not found: {cmd[0]}")
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Trace collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_single_trace(tracefs, workload_name, workload_config,
                         raw_dir, buffer_size_kb, run_idx):
    """Collect one trace: configure ftrace -> run workload -> save raw output.

    Returns the trace_id on success, None on skip/failure.
    """
    trace_id = f"{workload_name}_run{run_idx}"
    raw_path = raw_dir / f"{trace_id}.txt"
    meta_path = raw_dir / f"{trace_id}.json"

    print(f"\n[{trace_id}] {workload_config['desc']}")

    # Setup ftrace
    setup_ftrace(tracefs, buffer_size_kb)
    time.sleep(0.3)  # let the buffer settle

    start_ts = time.time()
    start_tracing(tracefs)

    # Run the workload
    success = run_workload(workload_name, workload_config)

    stop_tracing(tracefs)
    end_ts = time.time()

    if not success:
        cleanup_ftrace(tracefs)
        return None

    # Read and save raw trace
    trace_text = read_trace_buffer(tracefs)
    cleanup_ftrace(tracefs)

    event_count = trace_text.count("sched_switch:")
    duration = end_ts - start_ts
    print(f"  Captured {event_count} sched_switch events in {duration:.1f}s")

    if event_count == 0:
        print("  WARNING: No events captured — buffer may be too small "
              "or tracing failed.")
        return None

    # Save raw trace text
    with open(raw_path, "w") as f:
        f.write(trace_text)

    # Save per-trace metadata
    meta = {
        "trace_id": trace_id,
        "workload_name": workload_name,
        "category": workload_config["category"],
        "description": workload_config["desc"],
        "start_time": start_ts,
        "end_time": end_ts,
        "duration_seconds": round(duration, 2),
        "sched_switch_events": event_count,
        "buffer_size_kb": buffer_size_kb,
        "collected_at": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "num_cpus": os.cpu_count(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {raw_path} ({os.path.getsize(raw_path) / 1024:.0f} KB)")
    return trace_id


def collect_all(args):
    """Collect traces for the requested workloads."""
    global _tracefs_global

    tracefs = find_tracefs()
    if not tracefs:
        print("ERROR: Cannot find tracefs. Is ftrace available?")
        print("  Try: sudo mount -t tracefs tracefs /sys/kernel/tracing")
        sys.exit(1)

    if os.geteuid() != 0:
        print("ERROR: Root privileges required for ftrace. Run with sudo.")
        sys.exit(1)

    _tracefs_global = tracefs

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Determine which workloads to run
    if args.workload:
        names = [w.strip() for w in args.workload.split(",")]
        for n in names:
            if n not in WORKLOADS:
                print(f"ERROR: Unknown workload '{n}'")
                print(f"  Available: {', '.join(sorted(WORKLOADS.keys()))}")
                sys.exit(1)
        workloads = {n: WORKLOADS[n] for n in names}
    else:
        workloads = WORKLOADS

    num_cpus = os.cpu_count() or 1
    total_buf_mb = (args.buffer_size * num_cpus) / 1024

    print("=" * 60)
    print("KernelOracle Scheduling Trace Collector")
    print("=" * 60)
    print(f"Workloads:   {len(workloads)}")
    print(f"Runs each:   {args.runs}")
    print(f"CPUs:        {num_cpus}")
    print(f"Buffer:      {args.buffer_size} KB/cpu ({total_buf_mb:.0f} MB total)")
    print(f"tracefs:     {tracefs}")
    print(f"Output:      {output_dir}")
    print(f"Kernel:      {os.uname().release}")
    print("=" * 60)

    collected = []
    skipped = []

    for wname, wconfig in workloads.items():
        for run_idx in range(args.runs):
            trace_id = collect_single_trace(
                tracefs, wname, wconfig, raw_dir,
                args.buffer_size, run_idx,
            )
            if trace_id:
                collected.append(trace_id)
            else:
                skipped.append(f"{wname}_run{run_idx}")

            # Brief pause between runs to let the system settle
            time.sleep(1)

    _tracefs_global = None

    print("\n" + "=" * 60)
    print(f"Collection complete: {len(collected)} traces captured, "
          f"{len(skipped)} skipped")
    print(f"Raw traces saved to: {raw_dir}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print("=" * 60)

    # Fix file ownership if running via sudo
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid and sudo_gid:
        uid, gid = int(sudo_uid), int(sudo_gid)
        if uid > 0:
            for root, dirs, files in os.walk(output_dir):
                os.chown(root, uid, gid)
                for fname in files:
                    os.chown(os.path.join(root, fname), uid, gid)
            sudo_user = os.environ.get("SUDO_USER", str(uid))
            print(f"Ownership set to {sudo_user} ({uid}:{gid})")

    return collected


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_raw_trace(raw_text, workload_type, trace_id):
    """Parse raw ftrace text and extract sched_switch events.

    Returns a list of dicts, one per sched_switch event, with fields
    matching CSV_COLUMNS.
    """
    events = []
    prev_ts = None

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = SCHED_SWITCH_RE.search(line)
        if not m:
            continue

        ts = float(m.group("timestamp"))
        td = (ts - prev_ts) if prev_ts is not None else 0.0
        prev_ts = ts

        events.append({
            "timestamp": f"{ts:.6f}",
            "time_diff": f"{td:.9f}",
            "task_name": m.group("next_comm"),
            "pid": m.group("next_pid"),
            "cpu": m.group("cpu"),
            "prev_state": m.group("prev_state"),
            "prev_comm": m.group("prev_comm"),
            "workload_type": workload_type,
            "trace_id": trace_id,
        })

    return events


# ──────────────────────────────────────────────────────────────────────────────
# Dataset building (train / test split)
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(args):
    """Parse all raw traces and build train/test CSV files."""
    raw_dir = (Path(args.raw_dir)
               if getattr(args, "raw_dir", None)
               else Path(args.output_dir) / "raw")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.is_dir():
        print(f"ERROR: Raw trace directory not found: {raw_dir}")
        sys.exit(1)

    raw_files = sorted(raw_dir.glob("*.txt"))
    if not raw_files:
        print(f"ERROR: No .txt trace files found in {raw_dir}")
        sys.exit(1)

    print(f"\nParsing {len(raw_files)} raw trace file(s) from {raw_dir}")

    # Parse all traces, grouped by category
    category_traces = defaultdict(list)   # category -> [(trace_id, events)]
    total_events = 0

    for raw_file in raw_files:
        trace_id = raw_file.stem
        meta_file = raw_file.with_suffix(".json")

        # Read metadata if available, otherwise infer from filename
        if meta_file.exists():
            with open(meta_file) as mf:
                meta = json.load(mf)
            category = meta.get("category", "unknown")
            workload_type = meta.get("workload_name", trace_id)
        else:
            # Best-effort: strip _runN suffix to get workload name
            parts = trace_id.rsplit("_run", 1)
            workload_type = parts[0]
            category = workload_type.rsplit("_", 1)[0] if "_" in workload_type else workload_type

        with open(raw_file) as tf:
            raw_text = tf.read()

        events = parse_raw_trace(raw_text, workload_type, trace_id)
        total_events += len(events)
        category_traces[category].append((trace_id, events))
        print(f"  {trace_id}: {len(events):>7} sched_switch events "
              f"(category: {category})")

    categories = sorted(category_traces.keys())
    print(f"\nTotal: {total_events} events, {len(raw_files)} traces, "
          f"{len(categories)} categories: {categories}")

    # ── Split strategy ──
    #
    # 1. If >= 3 categories, hold out one entire category for "unseen" test
    #    (measures generalization to new workload types).
    # 2. Remaining categories are split by trace file at train_ratio
    #    (e.g. 80% train, 20% test_seen).
    # 3. Fallback: if too few traces per category to split, take last 20%
    #    of events from training data.

    random.seed(args.seed)

    train_events = []
    test_seen_events = []
    test_unseen_events = []

    unseen_cat = None
    no_unseen = getattr(args, "no_unseen_split", False)

    if len(categories) >= 3 and not no_unseen:
        unseen_cat = random.choice(categories)
        print(f"\nHeld-out unseen category: '{unseen_cat}'")
        for _trace_id, events in category_traces[unseen_cat]:
            test_unseen_events.extend(events)

    remaining_cats = [c for c in categories if c != unseen_cat]

    for cat in remaining_cats:
        traces = list(category_traces[cat])
        random.shuffle(traces)
        n_train = max(1, int(len(traces) * args.train_ratio))

        for _trace_id, events in traces[:n_train]:
            train_events.extend(events)
        for _trace_id, events in traces[n_train:]:
            test_seen_events.extend(events)

    # Fallback if we ended up with no test_seen data
    if not test_seen_events and train_events:
        split_idx = int(len(train_events) * args.train_ratio)
        test_seen_events = train_events[split_idx:]
        train_events = train_events[:split_idx]
        print("  (Used event-level fallback split for test_seen)")

    # ── Write CSVs ──

    def write_csv_file(filepath, events):
        if not events:
            return 0
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(events)
        return len(events)

    n_train = write_csv_file(output_dir / "train.csv", train_events)
    n_test_seen = write_csv_file(output_dir / "test_seen.csv", test_seen_events)
    n_test_unseen = write_csv_file(output_dir / "test_unseen.csv", test_unseen_events)

    # Combined dataset (all events, useful for exploratory analysis)
    all_events = train_events + test_seen_events + test_unseen_events
    n_all = write_csv_file(output_dir / "all.csv", all_events)

    # ── Save split metadata ──

    split_meta = {
        "total_events": total_events,
        "train_events": n_train,
        "test_seen_events": n_test_seen,
        "test_unseen_events": n_test_unseen,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "unseen_category": unseen_cat,
        "categories": categories,
        "num_traces": len(raw_files),
        "csv_columns": CSV_COLUMNS,
        "created_at": datetime.now().isoformat(),
    }
    with open(output_dir / "dataset_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    print(f"\nDataset built:")
    print(f"  train.csv:        {n_train:>8} events")
    print(f"  test_seen.csv:    {n_test_seen:>8} events")
    print(f"  test_unseen.csv:  {n_test_unseen:>8} events")
    print(f"  all.csv:          {n_all:>8} events")
    print(f"  dataset_meta.json")
    print(f"\nSaved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def list_workloads():
    """Print available workloads with install status."""
    print("Available workloads:\n")
    print(f"  {'Name':<30s} {'Description':<45s} {'Status'}")
    print(f"  {'─' * 30} {'─' * 45} {'─' * 15}")
    for name, config in sorted(WORKLOADS.items()):
        tool = config.get("tool")
        if tool:
            status = "installed" if tool_available(tool) else "NOT FOUND"
        else:
            status = "built-in"
        print(f"  {name:<30s} {config['desc']:<45s} {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Linux scheduling traces using ftrace "
                    "(Google SchedViz tracepoints) for training TCN/LSTM models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── collect ──
    p_collect = subparsers.add_parser(
        "collect",
        help="Collect raw ftrace traces while running workloads (requires root)",
    )
    p_collect.add_argument(
        "--output-dir", "-o", default="traces",
        help="Output directory (default: traces)",
    )
    p_collect.add_argument(
        "--workload", "-w", default=None,
        help="Run specific workload(s), comma-separated (default: all)",
    )
    p_collect.add_argument(
        "--runs", "-r", type=int, default=DEFAULT_RUNS,
        help=f"Number of runs per workload (default: {DEFAULT_RUNS})",
    )
    p_collect.add_argument(
        "--buffer-size", "-b", type=int, default=DEFAULT_BUFFER_SIZE_KB,
        help=f"ftrace buffer size in KB per CPU (default: {DEFAULT_BUFFER_SIZE_KB})",
    )
    p_collect.add_argument(
        "--list-workloads", action="store_true",
        help="List available workloads and exit",
    )

    # ── build ──
    p_build = subparsers.add_parser(
        "build",
        help="Parse raw traces and build train/test CSV datasets (no root needed)",
    )
    p_build.add_argument(
        "--output-dir", "-o", default="traces",
        help="Output directory (default: traces)",
    )
    p_build.add_argument(
        "--raw-dir", default=None,
        help="Raw trace directory (default: <output-dir>/raw)",
    )
    p_build.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Fraction of traces used for training (default: 0.8)",
    )
    p_build.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    p_build.add_argument(
        "--no-unseen-split", action="store_true",
        help="Don't hold out a category for unseen-workload testing",
    )

    # ── all ──
    p_all = subparsers.add_parser(
        "all",
        help="Collect traces + build dataset end-to-end (requires root)",
    )
    p_all.add_argument("--output-dir", "-o", default="traces")
    p_all.add_argument("--workload", "-w", default=None)
    p_all.add_argument("--runs", "-r", type=int, default=DEFAULT_RUNS)
    p_all.add_argument("--buffer-size", "-b", type=int, default=DEFAULT_BUFFER_SIZE_KB)
    p_all.add_argument("--train-ratio", type=float, default=0.8)
    p_all.add_argument("--seed", type=int, default=42)
    p_all.add_argument("--no-unseen-split", action="store_true")
    p_all.add_argument("--list-workloads", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if getattr(args, "list_workloads", False):
        list_workloads()
        sys.exit(0)

    if args.command == "collect":
        collect_all(args)

    elif args.command == "build":
        build_dataset(args)

    elif args.command == "all":
        collect_all(args)
        # build_dataset expects raw_dir; default to <output-dir>/raw
        args.raw_dir = None
        build_dataset(args)


if __name__ == "__main__":
    main()
