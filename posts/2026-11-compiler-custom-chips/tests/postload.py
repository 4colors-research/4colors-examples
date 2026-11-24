"""Load a post's script (run.py, plot.py, ...) under a module name unique to that post (tests only).

Several posts have a run.py, a plot.py or a model.py. A plain `import run` puts the first one into sys.modules
and every later post's tests would silently get it. `load(post_dir, "run")` imports <post_dir>/run.py as
`blogpost_<post>_<hash>__run`, where <hash> comes from the absolute post path, so two posts never share a name
(not even "a-b" and "a_b"). The module is registered in sys.modules before it runs (dataclasses and pickling
need that). Each call runs the file afresh, as a plain loader would; the newest copy is the registered one.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from types import ModuleType


def module_name(post_dir, name: str) -> str:
    post_dir = os.path.abspath(os.fspath(post_dir))
    slug = re.sub(r"\W", "_", os.path.basename(post_dir))
    tag = hashlib.sha1(post_dir.encode()).hexdigest()[:8]
    return f"blogpost_{slug}_{tag}__" + re.sub(r"\W", "_", name)


def load(post_dir, name: str) -> ModuleType:
    """Import <post_dir>/<name>.py under a name unique to the post."""
    path = os.path.join(os.path.abspath(os.fspath(post_dir)), f"{name}.py")
    mod_name = module_name(post_dir, name)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    return mod
