#!/usr/bin/env python3
"""Choose a safe search thread count for Linux AIO based search binaries."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--label", default="search")
    parser.add_argument(
        "--events-per-thread",
        type=int,
        default=int(os.environ.get("HIDANN_AIO_EVENTS_PER_THREAD", "1024")),
        help="libaio queue entries requested by each search worker",
    )
    parser.add_argument(
        "--safety-events",
        type=int,
        default=int(os.environ.get("HIDANN_AIO_SAFETY_EVENTS", "4096")),
        help="queue entries left free for other processes",
    )
    args = parser.parse_args()

    requested_threads = max(1, args.threads)
    max_nr = read_int(Path("/proc/sys/fs/aio-max-nr"))
    current = read_int(Path("/proc/sys/fs/aio-nr"))
    if max_nr is None or current is None:
        print(requested_threads)
        print(
            f"[aio-preflight] {args.label}: /proc/sys/fs/aio-* is unavailable; "
            f"using requested threads={requested_threads}",
            file=sys.stderr,
        )
        return 0

    available = max(0, max_nr - current)
    required = requested_threads * args.events_per_thread
    if required + args.safety_events <= available:
        print(requested_threads)
        return 0

    capped = max(1, (available - args.safety_events) // args.events_per_thread)
    if env_flag("HIDANN_AIO_AUTO_CAP", True) and capped >= 1:
        print(capped)
        print(
            f"[aio-preflight] {args.label}: capping search threads "
            f"{requested_threads}->{capped}; aio-max-nr={max_nr}, aio-nr={current}, "
            f"events_per_thread={args.events_per_thread}. "
            "Set HIDANN_AIO_AUTO_CAP=0 to make this a hard error.",
            file=sys.stderr,
        )
        return 0

    suggested = current + required + args.safety_events
    print(
        f"[aio-preflight] {args.label}: requested {requested_threads} search threads "
        f"need about {required} Linux AIO events, but only {available} are available "
        f"(aio-max-nr={max_nr}, aio-nr={current}). Lower SEARCH_THREADS or run:\n"
        f"  sudo sysctl fs.aio-max-nr={suggested}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
