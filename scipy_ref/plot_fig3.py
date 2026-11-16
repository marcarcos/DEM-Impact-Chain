"""Reproduce Fig. 3 (a-c) of Tamur et al. -- or Fig. 4 with --mass 1.01.

Paper conventions (Fig. 3 caption and text):
  * columns: linear chain and the three optimized designs of SI Table S1
    (homogeneous bistable, heterogeneous, heterogeneous excluding bistability)
  * (a) kinetic energy of each unit cell normalized by the impactor energy,
        KE/IE with IE = M V^2 / 2, log colour scale 1e-4 .. 1
  * (b) potential (strain) energy of each unit cell's spring, log scale 1e-6 .. 1e-4
  * (c) local stiffness df/dDx of each spring, saturated to show negative values only
  * time on the y axis as t = t_sim / 25, shown for t in [0, 4] (Fig. 4: [0, 6])
  * star = time of maximum KE at the last cell (unit cell 20)
  * (d) |2D FFT| of the KE map over the whole run (t up to 20), wavenumber kx (rad per
        cell) against fn = f* / sqrt(c1*/m*) (cycles per non-dimensional time), log 1e-2 .. 1

    python plot_fig3.py                      # Fig. 3 (rows a-d)
    python plot_fig3.py --fft-of velocity    # row d from velocity instead of KE
    python plot_fig3.py --mass 1.01          # Fig. 4 (impactor mass 1.01 M, rows a-c)

Row d caveat: the paper's linear panel peaks near fn = 0.3 at kx = 3, which is the
linear chain's velocity dispersion fn = sin(|kx|/2)/pi. The KE spectrum (what the caption
says is transformed) reaches about 0.43 there, because KE = v^2 doubles wavenumber and
frequency. The nonlinear panels look closer with KE, so KE is the default.

Caveat: SI Table S1 prints the coefficients to 2 decimals. The homogeneous bistable
design is so sensitive that the rounded values give KE ratio 0.0231 instead of the
paper's 0.0117, so that column can differ visibly from the paper; the others match.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot_matrices import simulate

DESIGNS = [  # (column title, c2 per family A-D, c3 per family A-D); SI Table S1
    ("Linear", None, None),
    ("Homogeneous\nBistable", [-116.12] * 4, [3052.97] * 4),
    ("Heterogeneous", [-84.36, -94.08, -10.11, -33.35], [1589.77, 2426.41, 1682.13, 3300.14]),
    ("Heterogeneous\nExcluding Bistability", [-59.83, -69.15, -100.21, -53.23],
     [2715.10, 1472.62, 3374.10, 1371.83]),
]
T_SCALE = 25.0   # paper: t = t_sim / 25


def cell_fields(X, c2, c3, m_imp, v_imp):
    """Per-unit-cell KE/IE, spring PE and local stiffness from a MATLAB-layout state X."""
    N = 20
    C1 = np.ones(N)
    C2 = np.repeat(np.asarray(c2, float), 5) if c2 is not None else np.zeros(N)
    C3 = np.repeat(np.asarray(c3, float), 5) if c3 is not None else np.zeros(N)

    v = X[:, 3::2]                       # chain velocities, masses 1..N
    u = X[:, 2::2]                       # chain displacements, masses 1..N
    ke = 0.5 * 1.0 * v ** 2 / (0.5 * m_imp * v_imp ** 2)

    # spring j joins mass j and mass j+1 (the last one joins mass N to the wall)
    d = np.hstack([u[:, 1:], np.zeros((len(u), 1))]) - u      # u_right - u_left
    dx = -d                                                     # compression > 0 (paper's Dx)
    comp = dx > 0
    pe = np.where(comp, C1 * dx ** 2 / 2 + C2 * dx ** 3 / 3 + C3 * dx ** 4 / 4, C1 * d ** 2 / 2)
    k = np.where(comp, C1 + 2 * C2 * dx + 3 * C3 * dx ** 2, C1)
    return ke, pe, k


def spectrum(field, t, nx_pad=256, dt_target=0.05):
    """Row d: 2D FFT of a space-time field over the whole run (paper: t up to 20 = t_sim 500).

    Returns kx (rad per unit cell, -pi..pi), fn (cycles per unit of non-dimensional time,
    which is the paper's f* / sqrt(c1*/m*)) for 0 <= fn <= 1, and |FFT| normalized to max 1.
    The 20 cells are zero-padded to nx_pad points so the kx axis is smooth, as in the paper.
    Time is thinned to steps of ~dt_target (Nyquist ~10, far above the plotted fn <= 1).
    """
    dt = t[1] - t[0]
    step = max(1, int(round(dt_target / dt)))
    A = field[::step]
    F = np.fft.fftshift(np.fft.fft2(A, s=(len(A), nx_pad)), axes=1)
    fn = np.fft.fftfreq(len(A), dt * step)
    kx = np.fft.fftshift(np.fft.fftfreq(nx_pad)) * 2 * np.pi
    keep = (fn >= 0) & (fn <= 1)
    order = np.argsort(fn[keep])
    S = np.abs(F[keep])[order]
    return kx, fn[keep][order], S / S.max()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mass", type=float, default=1.0, help="impactor mass factor (1.01 = Fig. 4)")
    ap.add_argument("--tmax", type=float, default=None, help="time window in paper units (default 4, or 6 with --mass)")
    ap.add_argument("--fft-of", choices=["ke", "velocity"], default="ke",
                    help="field transformed in row d (paper caption: ke; velocity shows the linear dispersion)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tmax = a.tmax if a.tmax is not None else (4.0 if a.mass == 1.0 else 6.0)
    out = a.out or ("fig3.png" if a.mass == 1.0 else "fig4.png")
    with_fft = a.mass == 1.0          # the paper shows row d in Fig. 3 only

    m_imp, v_imp = 0.01 * 10 * a.mass, 0.25
    nrows = 4 if with_fft else 3
    fig, axes = plt.subplots(nrows, 4, figsize=(13, 3.1 * nrows + 0.6), constrained_layout=True)
    ims = [None] * nrows
    ke_lin_max = None
    for col, (title, c2, c3) in enumerate(DESIGNS):
        # simulate() runs the linear and a nonlinear chain; the linear column uses its linear run
        t, X_lin, X_non = simulate(c2 if c2 is not None else [0] * 4, c3 if c3 is not None else [0] * 4,
                                   mass_factor=a.mass)
        X = X_lin if c2 is None else X_non
        ke, pe, k = cell_fields(X, c2, c3, m_imp, v_imp)
        tp = t / T_SCALE

        ke_last = ke[:, -1]
        t_star = tp[np.argmax(ke_last)]          # time of max KE at the last cell (whole run)
        if c2 is None:
            ke_lin_max = ke_last.max()
        ratio = ke_last.max() / ke_lin_max
        print(f"{title.replace(chr(10), ' '):38s} KE ratio {ratio:.4f}   peak at t = {t_star:.2f}")

        keep = tp <= tmax
        extent = [0.5, 20.5, 0, tp[keep][-1]]
        panels = [
            (np.clip(ke[keep], 1e-4, 1), dict(cmap="hot", norm=LogNorm(1e-4, 1))),
            (np.clip(pe[keep], 1e-6, 1e-4), dict(cmap="hot", norm=LogNorm(1e-6, 1e-4))),
            (np.clip(k[keep], -0.1, 0), dict(cmap="hot_r", vmin=-0.1, vmax=0)),
        ]
        for row, (M, style) in enumerate(panels):
            ax = axes[row, col]
            ims[row] = ax.imshow(M, aspect="auto", origin="lower", extent=extent,
                                 interpolation="nearest", **style)
            if t_star <= tmax:
                ax.plot(20, t_star, marker="*", ms=14, mfc="#00d5e0", mec="k", mew=0.8, clip_on=False)
            ax.set_xticks([5, 10, 15, 20])
            ax.set_ylim(0, tmax)
            ax.set_xlabel("Unit Cell")
            if col == 0:
                ax.set_ylabel("Time")
        axes[0, col].set_title(f"{title}\nKE ratio {ratio:.4f}", fontsize=11)

        if with_fft:
            field = ke if a.fft_of == "ke" else X[:, 3::2]
            kx, fn, S = spectrum(field, t)
            ax = axes[3, col]
            ims[3] = ax.pcolormesh(kx, fn, np.clip(S, 1e-2, 1), norm=LogNorm(1e-2, 1), cmap="hot",
                                   shading="auto", rasterized=True)
            ax.set_xlim(-3, 3)
            ax.set_xticks(range(-3, 4))
            ax.set_ylim(0, 1)
            ax.set_xlabel(r"$k_x$")
            if col == 0:
                ax.set_ylabel(r"$f_n$")

    rows = [("a.", "KE / IE"), ("b.", "PE"), ("c.", r"$\partial f/\partial \Delta x$")]
    if with_fft:
        rows.append(("d.", "|FFT|"))
    for row, (lab, name) in enumerate(rows):
        fig.colorbar(ims[row], ax=axes[row, :], label=name, shrink=0.9, pad=0.01)
        axes[row, 0].text(-0.3, 1.02, lab, transform=axes[row, 0].transAxes, fontsize=14, fontweight="bold")
    fig.suptitle(("Fig. 3" if a.mass == 1.0 else f"Fig. 4 (impactor mass {a.mass} M)")
                 + " reproduction: Tamur et al.; time t = t_sim / 25", fontsize=12)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
