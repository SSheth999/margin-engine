"""Distribution plots (build step 7).

The story is the SHAPE of the tail, so we plot cost-per-outcome histograms per arm
(overlaid) plus the margin ceiling reference. Matplotlib is imported lazily and with
a non-interactive backend so this runs headless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def cost_distributions(runs: list[dict], out_path: Path, arms: list[str]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    costs_by_arm = {
        arm: np.array([r["final_cost"] for r in runs if r["arm"] == arm], dtype=float)
        for arm in arms
    }
    all_costs = np.concatenate([c for c in costs_by_arm.values() if c.size]) if runs else np.array([])
    if all_costs.size == 0:
        return None
    bins = np.linspace(0, float(all_costs.max()) * 1.05 + 1e-9, 30)

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in arms:
        c = costs_by_arm[arm]
        if c.size:
            ax.hist(c, bins=bins, alpha=0.45, label=f"{arm} (n={c.size})")
    ax.set_xlabel("cost per outcome ($)")
    ax.set_ylabel("count")
    ax.set_title("Cost-per-outcome distribution by arm")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
