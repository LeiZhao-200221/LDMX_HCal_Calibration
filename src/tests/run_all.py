#!/usr/bin/env python
"""Run every test module in src/tests (no pytest needed; pytest also works).

    python -X utf8 src/tests/run_all.py            # all fast tests
    python -X utf8 src/tests/run_all.py calib ci   # only modules whose name contains one of the words
    python -X utf8 src/tests/run_all.py --in-process common   # no subprocess per module (debugging)
    python -X utf8 src/tests/run_all.py --timeout 600         # a module running longer than 600 s fails
    $env:LDMX_SLOW_TESTS=1                          # also run the slow tests (real pdf rendering)

A test module is ``test_*.py``; a test is a module-level function ``test_*()`` or a
``unittest.TestCase``.  A test that raises ``unittest.SkipTest`` (e.g. real data not on this
machine) is reported as SKIP, not as a failure.

Each test module runs in its own Python process (memory is returned to the system between modules,
and no global state leaks from one module into the next), with the Agg matplotlib backend and
single-threaded BLAS.  A module whose process dies without reporting (crash, out of memory,
``--timeout``) counts as one failure.

Author: Lei
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
import threading
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # src/
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

TEST_ENV = {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"}
RESULT_TAG = "@@RESULT"


# import a test module from its file
def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"tests.{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# run the tests of one module in this process
def run_module(path: Path) -> tuple[int, int, int]:
    """Run the tests of one module in this process; (passed, failed, skipped)."""
    passed = failed = skipped = 0
    try:
        mod = _load(path)
    except unittest.SkipTest as exc:
        print(f"SKIP  {path.name}: {exc}")
        return 0, 0, 1
    except KeyboardInterrupt:
        raise
    except BaseException:  # also SystemExit: stage code reports user errors that way
        print(f"FAIL  {path.name}: import error\n{traceback.format_exc()}")
        return 0, 1, 0
    funcs = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
             if n.startswith("test_") and f.__module__ == mod.__name__]
    for name, fn in sorted(funcs, key=lambda nf: inspect.getsourcelines(nf[1])[1]):
        t = time.time()
        try:
            fn()
            passed += 1
            print(f"PASS  {path.stem}.{name}  ({time.time() - t:.1f}s)")
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP  {path.stem}.{name}: {exc}")
        except KeyboardInterrupt:
            raise
        except BaseException:  # also SystemExit raised inside a test
            failed += 1
            print(f"FAIL  {path.stem}.{name}\n{traceback.format_exc()}")
    cases = [c for _, c in inspect.getmembers(mod, inspect.isclass)
             if issubclass(c, unittest.TestCase) and c.__module__ == mod.__name__]
    if cases:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in cases)
        res = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
        n_skip = len(res.skipped)
        n_fail = len(res.failures) + len(res.errors)
        failed += n_fail
        skipped += n_skip
        passed += res.testsRun - n_fail - n_skip
    return passed, failed, skipped


# run one module in a fresh Python process
def run_isolated(path: Path, timeout: float | None) -> tuple[int, int, int]:
    """Run one module in a fresh Python process, streaming its output; (passed, failed, skipped)."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    for k, v in TEST_ENV.items():
        env.setdefault(k, v)
    cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--one", str(path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            encoding="utf-8", errors="replace")
    watchdog = threading.Timer(timeout, proc.kill) if timeout else None
    if watchdog is not None:
        watchdog.start()
    result = None
    try:
        for line in proc.stdout:
            if line.startswith(RESULT_TAG):
                result = tuple(int(x) for x in line.split()[1:4])
            else:
                print(line, end="", flush=True)
        proc.wait()
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if proc.poll() is None:
            proc.kill()
    if result is None:
        why = f"killed after {timeout:g} s (--timeout)" if timeout and proc.returncode != 0 else \
            f"exit code {proc.returncode}"
        print(f"FAIL  {path.name}: the test process ended without a result ({why})")
        return 0, 1, 0
    return result


# run the chosen modules; exit code 1 when a test failed
def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args[:1] == ["--one"]:  # child process of run_isolated
        for k, v in TEST_ENV.items():
            os.environ.setdefault(k, v)
        p, f, s = run_module(Path(args[1]))
        print(f"{RESULT_TAG} {p} {f} {s}", flush=True)
        return 0
    in_process = "--in-process" in args
    timeout = None
    if "--timeout" in args:
        i = args.index("--timeout")
        timeout = float(args[i + 1])
        del args[i:i + 2]
    words = [w.lower() for w in args if not w.startswith("--")]
    files = sorted(HERE.glob("test_*.py"))
    if words:
        files = [f for f in files if any(w in f.stem.lower() for w in words)]
    if in_process:
        for k, v in TEST_ENV.items():
            os.environ.setdefault(k, v)
    totals = [0, 0, 0]
    for f in files:
        counts = run_module(f) if in_process else run_isolated(f, timeout)
        for i, n in enumerate(counts):
            totals[i] += n
    print(f"\n{totals[0]} passed, {totals[1]} failed, {totals[2]} skipped ({len(files)} test files)")
    return 1 if totals[1] else 0


if __name__ == "__main__":
    sys.exit(main())
