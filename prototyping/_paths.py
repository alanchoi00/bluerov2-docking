"""Put the shared control law and the BlueRovSim submodule on sys.path.

The prototype is dev-only scaffolding, not an installed package, so it imports
`control.pbvs` (the SHIPPED law, tuned here and reused verbatim by #30) and
BlueRovSim's `bluerov_model` directly from source. Import this module first.
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent

for _p in (_ROOT / "src" / "control", _ROOT / "third_party" / "BlueRovSim"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
