"""Replay immutable research archives separately from live-code unit tests.

Only explicitly marked archive/provenance tests use historical source. Their
assertions are unchanged; the child process executes them against verified
source snapshots in a disposable checkout. No report or working file is edited.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "frozen_research(archive): replay a frozen research source snapshot")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    marker = pyfuncitem.get_closest_marker("frozen_research")
    if marker is None or os.environ.get("GCN_FROZEN_REPLAY") == "1":
        return None
    root = Path(__file__).resolve().parents[1]
    archive = root / marker.args[0]
    manifest = json.loads((archive / "manifest.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix="kk2-frozen-replay-") as directory:
        sandbox = Path(directory)
        # Copy rather than symlink: even corruption tests cannot alter originals.
        for name in ("gcn", "tests", "reports"):
            shutil.copytree(root / name, sandbox / name,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        for name, expected in manifest["algorithm_sources"].items():
            relative = Path(name)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert relative.parts[0] == "gcn" and relative.suffix == ".py"
            source = (archive / "source_snapshot" / relative).read_bytes()
            assert hashlib.sha256(source).hexdigest() == expected, f"Corrupt archived source: {name}"
            (sandbox / relative).write_bytes(source)
        relative_test = Path(pyfuncitem.path).relative_to(root)
        target = f"{relative_test}::{pyfuncitem.name}"
        environment = {**os.environ, "GCN_FROZEN_REPLAY": "1", "PYTHONPATH": str(sandbox)}
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", target],
            cwd=sandbox, env=environment, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
    return True
