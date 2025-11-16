"""Local shim for `fer` package to avoid ImportError when projects expect
`from fer import FER` but installed package API differs.

This module will try to load the real `fer` package from site-packages
while avoiding importing the local shim itself. If the real package or a
FER-like class isn't available, the shim exposes a lightweight fallback
`FER` class that raises a helpful error when used.
"""
import importlib.util
import importlib.machinery
import sys
import os


def _load_real_fer():
    # Search sys.path entries excluding the current project directory
    cwd = os.path.abspath(os.getcwd())
    for p in sys.path:
        try:
            if not p:
                # empty entry means cwd
                continue
            if os.path.abspath(p) == cwd:
                continue
        except Exception:
            continue
        try:
            spec = importlib.machinery.PathFinder.find_spec('fer', [p])
            if spec is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception:
            continue
    return None


_real_fer = _load_real_fer()


class _FERStub:
    def __init__(self, *args, **kwargs):
        self._available = False

    def detect_emotions(self, frame):
        raise RuntimeError(
            "FER functionality is not available in this environment. "
            "Install a compatible `fer` package or update the project to use `src.detect_emotion.get_emotion`."
        )

    # provide friendly repr
    def __repr__(self):
        return "<FER stub - real fer package not available>"


if _real_fer is not None:
    # try to expose a class named FER (or other common names) from the real package
    FER = getattr(_real_fer, 'FER', None) or getattr(_real_fer, 'FERDetector', None) or getattr(_real_fer, 'Detector', None)
    if FER is None:
        # real package loaded but no FER-like class found -> fallback to stub
        FER = _FERStub
else:
    FER = _FERStub
