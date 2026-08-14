#!/usr/bin/env python3
"""Check that the BW1100 JSON narrow-Linear model is consumed by Calculon."""
from __future__ import annotations

import json
from pathlib import Path

from calculon.llm.layers import Linear
from calculon.system import System


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    system = System(json.loads((ROOT / "systems" / "BW1100.json").read_text()))
    system.set_datatype("float16")
    tiny = Linear("tiny-n", system, 1024, 4096, 16)
    generic = Linear("narrow", system, 1024, 4096, 256)
    tiny_time = tiny.compute_flops_time("fw")
    generic_time = generic.compute_flops_time("fw")
    floor = system.get_linear_small_n_time(1024, 4096, 16)
    assert floor > 0 and tiny_time >= floor
    assert system.get_linear_small_n_time(1024, 4096, 256) == 0
    print(f"N=16 floor={floor * 1e6:.2f} us; Linear fw={tiny_time * 1e6:.2f} us")
    print(f"N=256 uses generic curve; Linear fw={generic_time * 1e6:.2f} us")


if __name__ == "__main__":
    main()
