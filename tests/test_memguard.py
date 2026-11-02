"""scripts/memguard.py: request parsing and process accounting (Linux /proc, no Slurm needed)."""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("memguard", Path(__file__).parents[1] / "scripts" / "memguard.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)


def test_request_mb():
    assert mg.request_mb({"SLURM_MEM_PER_NODE": "98304"}) == 98304
    assert mg.request_mb({"SLURM_MEM_PER_CPU": "2022", "SLURM_CPUS_ON_NODE": "8"}) == 16176
    assert mg.request_mb({}) is None


def test_pss_of_self_and_gone():
    assert mg.pss_kb(os.getpid()) > 1000          # a Python process is well above 1 MB
    assert mg.pss_kb(2 ** 22 + 12345) == 0        # no such pid


def test_job_pids_finds_descendants_and_tagged():
    env = dict(os.environ, SLURM_JOBID="424242")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    tagged = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env=env,
                              start_new_session=True)
    try:
        time.sleep(0.5)
        pids = mg.job_pids(os.getpid(), "424242", me=-1)
        assert child.pid in pids and tagged.pid in pids and os.getpid() in pids
        assert tagged.pid not in mg.job_pids(child.pid, "999999", me=-1)
    finally:
        child.kill()
        tagged.kill()
