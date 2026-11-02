"""Per-job memory watchdog: keep a Slurm job inside its own memory request.

Our Slurm site does not enforce memory requests (ProctrackType=proctrack/linuxproc,
TaskPlugin=task/affinity, no cgroups, no OverMemoryKill; checked 2026-10-26). `--mem` only books
memory for scheduling, so a job that outgrows it eats the shared node. On 2026-10-26 two of our
eval jobs (four suites in parallel, 50 simulator processes per batch) ran the GPU node out of
memory: the kernel killed processes at random, the node failed (NODE_FAIL) and was drained.

This watchdog sums the proportional set size (PSS: shared pages split between the processes
sharing them) of every process of the job, every POLL seconds:
  - prints a line when the peak grows by STEP_GB, and a warning above WARN (80%) of the request;
  - above KILL (90%) of the request, prints the largest processes and kills the whole job
    (SIGTERM, then SIGKILL), so the job fails alone instead of taking the node down;
  - writes the running peak to $AVLA_RUNS/memguard/<job>.json, to size requests from data.
A job's processes are the descendants of the batch shell, plus any of our processes whose
environment carries SLURM_JOBID=<job> (launchers override SLURM_JOB_ID for eval sub-calls; Slurm's
SLURM_JOBID alias is left alone).

Started by scripts/env.sh in every batch job on a compute node (once per job; AVLA_MEMGUARD_OFF=1
disables it). Usage: python scripts/memguard.py ROOT_PID
"""
import json
import os
import signal
import sys
import time

POLL = 5.0
WARN, KILL = 0.80, 0.90   # KILL leaves room for growth between polls (test: +0.6 GB in 5 s)
STEP_GB = 4.0


def request_mb(env=os.environ):
    """The job's memory request in MB from Slurm's environment, or None if it cannot be told."""
    if env.get("SLURM_MEM_PER_NODE"):
        return int(env["SLURM_MEM_PER_NODE"])
    if env.get("SLURM_MEM_PER_CPU") and env.get("SLURM_CPUS_ON_NODE"):
        return int(env["SLURM_MEM_PER_CPU"]) * int(env["SLURM_CPUS_ON_NODE"])
    return None


def pss_kb(pid):
    """PSS of one process in kB (0 if it is gone or unreadable)."""
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def job_pids(root, job, me=None):
    """Our processes that belong to the job: descendants of root, or SLURM_JOBID=<job> in environ."""
    uid, me = os.getuid(), me or os.getpid()
    parent, tagged = {}, set()
    marks = (f"SLURM_JOBID={job}\0".encode(), f"SLURM_JOB_ID={job}\0".encode(),
             f"SLURM_JOB_ID={job}-".encode())
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != uid:
                continue
            with open(f"/proc/{d}/stat") as f:
                parent[int(d)] = int(f.read().rsplit(")", 1)[1].split()[1])
            with open(f"/proc/{d}/environ", "rb") as f:
                env = f.read()
            if any(m in env for m in marks):
                tagged.add(int(d))
        except (OSError, IndexError, ValueError):
            continue
    pids = set(tagged) | {root}
    grew = True
    while grew:
        new = {p for p, pp in parent.items() if pp in pids and p not in pids}
        pids |= new
        grew = bool(new)
    pids.discard(me)
    return {p for p in pids if p in parent}


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return " ".join(f.read().decode(errors="replace").replace("\0", " ").split())[:120]
    except OSError:
        return "?"


def kill_all(pids):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        time.sleep(10)


def main():
    root = int(sys.argv[1])
    job = os.environ.get("SLURM_JOBID") or os.environ["SLURM_JOB_ID"].split("-")[0]
    req = request_mb()
    out = os.path.join(os.environ.get("AVLA_RUNS", "."), "memguard", f"{job}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tag = f"memguard[{job}]"
    print(f"{tag}: watching the job's PSS against its request "
          f"{req / 1024 if req else float('nan'):.0f} GB (warn {WARN:.0%}, kill {KILL:.0%})", flush=True)
    peak_gb, shown_gb, warned = 0.0, 0.0, False
    while os.path.exists(f"/proc/{root}"):
        pids = job_pids(root, job)
        use = {p: pss_kb(p) for p in pids}
        gb = sum(use.values()) / 2 ** 20
        if gb > peak_gb:
            peak_gb = gb
            with open(out, "w") as f:
                json.dump({"job": job, "peak_pss_gb": round(peak_gb, 2), "n_procs": len(pids),
                           "request_gb": round(req / 1024, 1) if req else None,
                           "node": os.environ.get("SLURMD_NODENAME"), "time": time.strftime("%FT%T")}, f)
        if peak_gb >= shown_gb + STEP_GB:
            shown_gb = peak_gb
            print(f"{tag}: peak {peak_gb:.1f} GB PSS, {len(pids)} processes", flush=True)
        if req and gb > WARN * req / 1024 and not warned:
            warned = True
            print(f"{tag}: WARNING {gb:.1f} GB PSS is above {WARN:.0%} of the {req / 1024:.0f} GB request",
                  flush=True)
        if req and gb > KILL * req / 1024:
            top = sorted(use.items(), key=lambda kv: -kv[1])[:5]
            print(f"{tag}: KILLING THE JOB: {gb:.1f} GB PSS exceeds {KILL:.0%} of its {req / 1024:.0f} GB "
                  f"request (memory is not enforced on this node; the alternative is a node-wide OOM). "
                  f"Largest: " + "; ".join(f"{p} {kb / 2 ** 20:.1f} GB {cmdline(p)}" for p, kb in top),
                  flush=True)
            kill_all(pids)
            sys.exit(1)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
