"""Regression guard on the committed gains: every scenario must converge < 10 s.

Skips cleanly until both prerequisites exist: the BlueRovSim submodule (the
plant) and a tuned results/coarse_pbvs_gains.yaml.
"""

import pathlib

import _paths  # noqa: F401
import pytest

pytest.importorskip("bluerov_model", reason="BlueRovSim submodule not initialized")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import metrics  # noqa: E402
from control.pbvs import CoarsePbvsParams  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from simulate import run  # noqa: E402

GAINS = pathlib.Path(__file__).resolve().parent / "results" / "coarse_pbvs_gains.yaml"

TOL_POS = 0.05
TOL_YAW = np.deg2rad(5.0)
T_LIMIT = 10.0


@pytest.mark.skipif(not GAINS.exists(), reason="no tuned gains committed yet")
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_converges_within_limit(scenario):
    gains = yaml.safe_load(GAINS.read_text())
    # handoff_range_m / surge_taper_range_m are prototyping-sim concepts (the
    # convergence target range), not production CoarsePbvsParams gains...
    handoff_range_m = gains.pop("handoff_range_m")
    gains.pop("surge_taper_range_m", None)
    params = CoarsePbvsParams(**gains)
    tr = run(params, scenario, handoff_range_m)
    assert metrics.converged(tr.t, tr.forward, handoff_range_m, TOL_POS, T_LIMIT)
    assert metrics.converged(tr.t, tr.left, 0.0, TOL_POS, T_LIMIT)
    assert metrics.converged(tr.t, tr.up, 0.0, TOL_POS, T_LIMIT)
    assert metrics.converged(tr.t, tr.yaw_err, 0.0, TOL_YAW, T_LIMIT)
