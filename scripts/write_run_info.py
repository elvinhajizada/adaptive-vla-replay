"""Write run_info.json (provenance) next to a run's results: CLI over adaptive_vla.provenance.

Usage: python scripts/write_run_info.py OUT_DIR POLICY START_EPOCH END_EPOCH "COMMAND" [KEY=VALUE ...]
Extra KEY=VALUE pairs are added as fields; all-digit values become ints, the rest stay strings
(train.sh: tasks="[13]" stays a string, steps=1000 becomes an int).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla.provenance import write_run_info  # noqa: E402


def main():
    out, policy, start, end, cmd, *pairs = sys.argv[1:]
    extra = {}
    for p in pairs:
        k, v = p.split("=", 1)
        extra[k] = int(v) if v.isdigit() else v
    write_run_info(out, policy, int(start), int(end), cmd, **extra)


if __name__ == "__main__":
    main()
