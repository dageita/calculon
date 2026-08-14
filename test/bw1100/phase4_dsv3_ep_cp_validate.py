#!/usr/bin/env python3
"""BW1100 Phase4 EP/CP semantic validation.

The full implementation lives in phase4_dsv3_parallel_validate.py.  This
canonical name matches the H20 experiment and validates Calculon byte formulas,
C++ timeline events, and EP/CP ablations.  It does not replace the physical
RCCL calibration in phase4_rccl_flow_calibrate.py.
"""
from phase4_dsv3_parallel_validate import main


if __name__ == '__main__':
    main()
