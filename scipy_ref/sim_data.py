import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from main_disorder_v2 import main_disorder_v2


# 1. Define baseline coefficients (4 spring families)
c2_base = np.array([-70.0, -100.0, -5.0, -35.0])
c3_base = np.array([1200.0, 2000.0, 1500.0, 3000.0])


# 2. Define perturbation range: -50% to +50% around each baseline value
#    21 points from 0.5x to 1.5x of the baseline coefficient
num_steps = 21
scale_factors = np.linspace(0.5, 1.5, num_steps)
pct_changes = (scale_factors - 1.0) * 100


# Arrays to store results: shape (4 families, num_steps)
ke_ratio_c2_sweep = np.zeros((4, num_steps))
ke_ratio_c3_sweep = np.zeros((4, num_steps))


# Rows that will be saved to CSV
rows = []


# Compute baseline ratio once for reference
baseline_ratio = main_disorder_v2(c2_base, c3_base)

print(f"Baseline max KE ratio: {baseline_ratio:.6f}")


# 3. Sweep each c2[i] and c3[i] one at a time
#    All other coefficients remain fixed at their baseline values
for i in range(4):
    print(f"Running perturbations for Spring Family {i + 1}/4...")

    for j, scale in enumerate(scale_factors):

        # --------------------------------------------------------------
        # Perturb c2[i] only
        # --------------------------------------------------------------
        c2_pert = c2_base.copy()
        c2_pert[i] = c2_base[i] * scale

        ke_ratio = main_disorder_v2(c2_pert, c3_base)
        ke_ratio_c2_sweep[i, j] = ke_ratio

        rows.append(
            {
                "varied_parameter": f"c2_{i + 1}",
                "family": i + 1,
                "percent_change": pct_changes[j],
                "c2_1": c2_pert[0],
                "c2_2": c2_pert[1],
                "c2_3": c2_pert[2],
                "c2_4": c2_pert[3],
                "c3_1": c3_base[0],
                "c3_2": c3_base[1],
                "c3_3": c3_base[2],
                "c3_4": c3_base[3],
                "KE_ratio": ke_ratio,
            }
        )

        # --------------------------------------------------------------
        # Perturb c3[i] only
        # --------------------------------------------------------------
        c3_pert = c3_base.copy()
        c3_pert[i] = c3_base[i] * scale

        ke_ratio = main_disorder_v2(c2_base, c3_pert)
        ke_ratio_c3_sweep[i, j] = ke_ratio

        rows.append(
            {
                "varied_parameter": f"c3_{i + 1}",
                "family": i + 1,
                "percent_change": pct_changes[j],
                "c2_1": c2_base[0],
                "c2_2": c2_base[1],
                "c2_3": c2_base[2],
                "c2_4": c2_base[3],
                "c3_1": c3_pert[0],
                "c3_2": c3_pert[1],
                "c3_3": c3_pert[2],
                "c3_4": c3_pert[3],
                "KE_ratio": ke_ratio,
            }
        )


# ==============================================================================
# SAVE DATA TO TIMESTAMPED CSV
# ==============================================================================

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = output_dir / f"sensitivity_{timestamp}.csv"

fieldnames = [
    "varied_parameter",
    "family",
    "percent_change",
    "c2_1",
    "c2_2",
    "c2_3",
    "c2_4",
    "c3_1",
    "c3_2",
    "c3_3",
    "c3_4",
    "KE_ratio",
]

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} data points to: {csv_path}")


# ==============================================================================
# PLOT 1: Overlaid Relative Perturbation Curves (-50% to +50%)
# ==============================================================================

fig1, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    sharey=True,
)

family_labels = [
    "Family 1 (Cells 1–5)",
    "Family 2 (Cells 6–10)",
    "Family 3 (Cells 11–15)",
    "Family 4 (Cells 16–20)",
]

for i in range(4):
    ax1.plot(
        pct_changes,
        ke_ratio_c2_sweep[i, :],
        marker="o",
        markersize=4,
        label=f"{family_labels[i]} (base $c_2={c2_base[i]:.0f}$)",
    )

    ax2.plot(
        pct_changes,
        ke_ratio_c3_sweep[i, :],
        marker="s",
        markersize=4,
        label=f"{family_labels[i]} (base $c_3={c3_base[i]:.0f}$)",
    )


for ax, title in zip(
    (ax1, ax2),
    (
        "Quadratic Coefficient ($c_2$) Perturbations",
        "Cubic Coefficient ($c_3$) Perturbations",
    ),
):
    ax.axvline(
        0,
        color="k",
        linestyle="--",
        alpha=0.5,
        label="Baseline (0%)",
    )

    ax.axhline(
        baseline_ratio,
        color="gray",
        linestyle=":",
        alpha=0.6,
    )

    ax.set_title(title)
    ax.set_xlabel("Perturbation from Baseline (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


ax1.set_ylabel(
    "Max KE Ratio at Last Cell ($KE_{non} / KE_{lin}$)"
)

fig1.suptitle(
    "One-At-A-Time Sensitivity Analysis of Nonlinear Spring Families",
    fontsize=13,
)

fig1.tight_layout()


# ==============================================================================
# PLOT 2: Individual Parameter Sweeps
# ==============================================================================

fig2, axes = plt.subplots(
    2,
    4,
    figsize=(16, 7),
    sharey=True,
)


for i in range(4):

    # --------------------------------------------------------------
    # Top row: c2 values
    # --------------------------------------------------------------
    c2_vals = c2_base[i] * scale_factors

    axes[0, i].plot(
        c2_vals,
        ke_ratio_c2_sweep[i, :],
        "b-o",
        markersize=4,
    )

    axes[0, i].axvline(
        c2_base[i],
        color="r",
        linestyle="--",
        label=f"Base={c2_base[i]:.0f}",
    )

    axes[0, i].set_title(
        f"$c_{{2,{i + 1}}}$ ({family_labels[i]})"
    )

    axes[0, i].set_xlabel(
        f"$c_{{2,{i + 1}}}$ value"
    )

    axes[0, i].grid(
        True,
        alpha=0.3,
    )

    axes[0, i].legend(
        fontsize=8,
    )


    # --------------------------------------------------------------
    # Bottom row: c3 values
    # --------------------------------------------------------------
    c3_vals = c3_base[i] * scale_factors

    axes[1, i].plot(
        c3_vals,
        ke_ratio_c3_sweep[i, :],
        "g-s",
        markersize=4,
    )

    axes[1, i].axvline(
        c3_base[i],
        color="r",
        linestyle="--",
        label=f"Base={c3_base[i]:.0f}",
    )

    axes[1, i].set_title(
        f"$c_{{3,{i + 1}}}$ ({family_labels[i]})"
    )

    axes[1, i].set_xlabel(
        f"$c_{{3,{i + 1}}}$ value"
    )

    axes[1, i].grid(
        True,
        alpha=0.3,
    )

    axes[1, i].legend(
        fontsize=8,
    )


axes[0, 0].set_ylabel("Max KE Ratio")
axes[1, 0].set_ylabel("Max KE Ratio")

fig2.suptitle(
    "Exact Parameter Value Sweeps (All Other Coefficients Fixed)",
    fontsize=13,
)

fig2.tight_layout()


# Show both figures
plt.show()