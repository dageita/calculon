#!/usr/bin/env python3
"""BW1100 Phase 2 catalog.

The H20 catalog is hardware-neutral Calculon graph introspection; reuse it
while selecting BW1100.json by default.  This avoids maintaining two copies
of the DeepSeek-V3 layer-to-shape recovery logic.
"""
from __future__ import annotations

import runpy
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H20_CATALOG = ROOT / "test" / "h20" / "phase2_dsv3_op_catalog.py"

# Re-export the hardware-neutral catalog API for BW1100 Phase3 imports.
_spec = importlib.util.spec_from_file_location("bw1100_catalog_base", H20_CATALOG)
_base = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = _base
_saved_sys_path = list(sys.path)
try:
    _spec.loader.exec_module(_base)
finally:
    # The reused H20 module prepends test/h20 to sys.path.  Restore search
    # order so later Phase3 imports resolve BW1100 modules, not H20 namesakes.
    sys.path[:] = _saved_sys_path
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", str(ROOT / "models" / "deepseek-v3-671b.json")])
    if "--system" not in sys.argv:
        sys.argv.extend(["--system", str(ROOT / "systems" / "BW1100.json")])
    runpy.run_path(str(H20_CATALOG), run_name="__main__")
