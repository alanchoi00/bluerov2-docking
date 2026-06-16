"""Hand-tune entry point for the coarse-approach PBVS gains.

Workflow: edit the gains (in results/coarse_pbvs_gains.yaml, or the STARTING_GAINS
seed below if no YAML yet) -> run this -> read the metrics table + results/*.png ->
repeat -> commit the YAML + plots. The YAML keys match CoarsePbvsParams exactly,
"""

import _paths  # noqa: F401

import pathlib

import numpy as np
import yaml

import metrics
import plots
from control.pbvs import CoarsePbvsParams
from scenarios import SCENARIOS
from simulate import run

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
GAINS_YAML = RESULTS / "coarse_pbvs_gains.yaml"

# Seed used only if results/coarse_pbvs_gains.yaml does not exist yet.
# NOTE: commands are normalized EFFORT (see plan), so v_max_* are effort caps.
STARTING_GAINS = dict(
    kp_surge=0.5,
    kp_sway=0.8,
    kd_sway=0.3,
    kp_heave=0.8,
    kd_heave=0.3,
    kp_yaw=1.0,
    kd_yaw=0.3,
    handoff_range_m=0.5,
    surge_taper_range_m=1.0,
    v_max_surge=0.8,
    v_max_sway=0.6,
    v_max_heave=0.6,
    v_max_yaw=0.8,
)

TOL_POS = 0.05  # m
TOL_YAW = np.deg2rad(5.0)  # rad
T_LIMIT = 10.0  # s convergence target


def load_gains() -> dict:
    if GAINS_YAML.exists():
        return yaml.safe_load(GAINS_YAML.read_text())
    return dict(STARTING_GAINS)


def save_gains(gains: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    GAINS_YAML.write_text(yaml.safe_dump(gains, sort_keys=False))


def _row(name: str, tr, handoff: float) -> str:
    # report each axis against its target
    st_r = metrics.settling_time(tr.t, tr.forward, handoff, TOL_POS)
    st_l = metrics.settling_time(tr.t, tr.left, 0.0, TOL_POS)
    st_v = metrics.settling_time(tr.t, tr.up, 0.0, TOL_POS)
    st_y = metrics.settling_time(tr.t, tr.yaw_err, 0.0, TOL_YAW)
    sts = [s for s in (st_r, st_l, st_v, st_y) if s is not None]
    worst = max(sts) if sts else None
    ok = all(s is not None and s <= T_LIMIT for s in (st_r, st_l, st_v, st_y))
    worst_str = f"{worst:5.2f}" if worst is not None else "  inf"
    return f"  {name:9s}  settle(worst)={worst_str}s  converged<{T_LIMIT:g}s={ok}"


def main() -> None:
    gains = load_gains()
    params = CoarsePbvsParams(**gains)

    trajs = {}
    print("\ncoarse PBVS tuning -- per-scenario metrics:")
    for sc in SCENARIOS:
        tr = run(params, sc)
        trajs[sc.name] = tr
        print(_row(sc.name, tr, params.handoff_range_m))

    plots.plot_all(trajs, params.handoff_range_m, RESULTS)
    save_gains(gains)
    print(f"\nplots -> {RESULTS}/step_*.png")
    print(f"gains -> {GAINS_YAML}\n")


if __name__ == "__main__":
    main()
