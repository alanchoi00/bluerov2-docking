"""Step-response plots: per-axis error + command vs time, one figure per scenario."""

import matplotlib

matplotlib.use("Agg")  # headless (container has no display)
import matplotlib.pyplot as plt  # noqa: E402


def plot_all(trajs: dict, handoff_range_m: float, outdir) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, tr in trajs.items():
        fig, (ax_err, ax_cmd) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax_err.plot(tr.t, tr.forward - handoff_range_m, label="range err")
        ax_err.plot(tr.t, tr.left, label="lateral")
        ax_err.plot(tr.t, tr.up, label="vertical")
        ax_err.plot(tr.t, tr.yaw_err, label="yaw err (rad)")
        ax_err.axhline(0.0, color="k", lw=0.5)
        ax_err.set_ylabel("error")
        ax_err.legend(loc="upper right", fontsize=8)

        ax_cmd.plot(tr.t, tr.cmd_surge, label="surge")
        ax_cmd.plot(tr.t, tr.cmd_sway, label="sway")
        ax_cmd.plot(tr.t, tr.cmd_heave, label="heave")
        ax_cmd.plot(tr.t, tr.cmd_yaw, label="yaw")
        ax_cmd.set_ylabel("command (effort)")
        ax_cmd.set_xlabel("t (s)")
        ax_cmd.legend(loc="upper right", fontsize=8)

        fig.suptitle(f"coarse PBVS step response: {name}")
        fig.tight_layout()
        fig.savefig(outdir / f"step_{name}.png", dpi=110)
        plt.close(fig)
