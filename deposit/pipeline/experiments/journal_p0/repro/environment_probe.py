#!/usr/bin/env python3
"""Emit machine-readable runtime facts without guessing the embedded HiGHS version."""
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys

import matplotlib
import numpy
import pandas
import scipy


def _embedded_highs_version():
    candidates = (
        "scipy.optimize._highs._highs_wrapper",
        "scipy.optimize._highs._highs_constants",
        "scipy.optimize._highspy._core",
    )
    names = (
        "HIGHS_VERSION",
        "HIGHS_VERSION_MAJOR",
        "HIGHS_VERSION_MINOR",
        "HIGHS_VERSION_PATCH",
        "__version__",
    )
    observed = {}
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            observed[module_name] = {"import_error": type(exc).__name__}
            continue
        values = {name: getattr(module, name) for name in names if hasattr(module, name)}
        observed[module_name] = values
        if values:
            return {"status": "reported", "module": module_name, "values": values}
    return {
        "status": "not_recoverable_from_scipy_runtime",
        "reason": "SciPy 1.13.1 exposes milp/HiGHS status semantics but no embedded HiGHS "
                  "version constant in the inspected runtime modules; no version was guessed.",
        "inspected": observed,
    }


def main():
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    payload = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "os": {
            "platform": platform.platform(),
            "uname": list(platform.uname()),
        },
        "packages": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pip_freeze_all": freeze,
        },
        "solver": {
            "api": "scipy.optimize.milp",
            "embedded_highs": _embedded_highs_version(),
            "formal_options": {"presolve": False},
        },
        "container": {
            "base_amd64_manifest": "sha256:70fbdeb75b0d071778ea1df4353ac959118ae8031bc5be33a95d1b8ea1fb8f08",
            "image_digest": os.environ.get("DCHHR_IMAGE_DIGEST", "RECORDED_BY_HOST_AFTER_BUILD"),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
