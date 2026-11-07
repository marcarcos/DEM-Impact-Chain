"""Reference (slow) KE-ratio evaluation: scipy port of matlab/main_disorder_fn.m.

Simulates one heterogeneous chain design and the matching linear chain with
scipy's RK45 (same Dormand-Prince method as MATLAB's ode45), then returns

    KE ratio = max_t KE_N,non(t) / max_t KE_N,lin(t),   KE_N = v_N^2 / 2,

the objective J of the paper (Methods 5.2). Takes ~6 s per design; use
fast/fast_dem.py::main_disorder_v2 for anything bigger than a spot check.

Kept deliberately identical to the MATLAB code (including its output grid, see the
note in the function) so it can arbitrate between MATLAB and the fast version.
"""
import numpy as np
from scipy.integrate import solve_ivp

from impact_equation_of_motion_hetero import impact_equation_of_motion_hetero


def main_disorder_v2(c2, c3):
    """KE ratio of one design.

    Parameters
    ----------
    c2, c3 : array-like, shape (4,)
        Quadratic / cubic coefficients of spring families A, B, C, D (5 springs each,
        families ordered from the impacted end to the wall).

    Returns
    -------
    float : max KE of the last mass, nonlinear chain / linear chain.
    """
    c2 = np.asarray(c2, dtype=float)
    c3 = np.asarray(c3, dtype=float)

    c1 = np.ones(4)  # linear stiffness, uniform along the chain

    # family coefficients (4,) -> per-spring coefficients (20,): AAAAA BBBBB CCCCC DDDDD
    C1 = np.repeat(c1, 5)
    C2 = np.repeat(c2, 5)
    C3 = np.repeat(c3, 5)

    zeta = 0.02        # damping ratio (SI-1.2)

    N = 20             # unit cells
    m_layer = 1.0      # unit-cell mass

    # Reference impactor mass M0 = half the chain mass. MATLAB writes N/2, which is the
    # same number because m_layer = 1; this form stays correct if m_layer changes.
    M0_impactor = N * m_layer / 2
    C_impact = 1.2e4   # Hertzian contact coefficient (SI-1.3)

    M_impactor = 0.01 * M0_impactor   # design point M/M0 = 0.01
    V_impactor = 0.25                 # design point V/V0 = 0.25 (V0 = 1, linear sound speed)

    # Solver setup.
    # NOTE on the output grid: the paper (Methods 5.1) states dt_out = 1/(100 f0). The
    # MATLAB code instead uses dt = 1/(100 f) with f built from a stiffness
    # k_f = max(1, 1 + 2 c2 + 3 c3) (the tangent stiffness f'(1), a heuristic upper
    # bound), which is ~50-100x finer. We keep the MATLAB grid to stay identical to it;
    # the max-KE ratio differs by < 0.3 % between the two grids (see REPLICATION.md).
    k_f = np.max(np.concatenate(([1.0], 1 + 2 * c2 + 3 * c3)))

    cycles = 25
    output_per_cycle = 100

    f0 = (1 / (2 * np.pi)) * np.sqrt(1 / M0_impactor)
    dt_cyc0 = 1 / f0

    f = (1 / (2 * np.pi)) * np.sqrt(k_f / M0_impactor)
    dt_cyc = 1 / f

    dt = dt_cyc / output_per_cycle     # output spacing
    T = dt_cyc0 * cycles               # end time T = 25 / f0 ~ 496.7

    num_points = int(np.floor(T / dt)) + 1
    t_eval = np.arange(num_points) * dt

    # Material info and initial conditions
    material_info = np.array(
        [m_layer, M_impactor, C_impact, zeta],
        dtype=float,
    )

    # Interleaved state [u0, v0, u1, v1, ...]: all at rest except the impactor velocity
    initial_vals = np.zeros(2 * (N + 1))
    initial_vals[1] = V_impactor

    atol = np.full(2 * (N + 1), 1e-12)
    rtol = 1e-12

    # Linear reference system (c2 = c3 = 0)
    linear_spring_info = np.column_stack(
        (
            C1,
            np.zeros(N),
            np.zeros(N),
        )
    )

    linear_solution = solve_ivp(
        impact_equation_of_motion_hetero,
        (0, t_eval[-1]),
        initial_vals,
        method="RK45",
        t_eval=t_eval,
        args=(linear_spring_info, material_info, 0.0),
        rtol=rtol,
        atol=atol,
    )

    X_lin = linear_solution.y.T

    velocity_x_lin = X_lin[:, 1::2]        # (time, N+1) incl. impactor in column 0
    displacement_x_lin = X_lin[:, 2::2]    # (time, N) chain masses only (unused here)

    # Nonlinear system
    nonlinear_spring_info = np.column_stack(
        (
            C1,
            C2,
            C3,
        )
    )

    nonlinear_solution = solve_ivp(
        impact_equation_of_motion_hetero,
        (0, t_eval[-1]),
        initial_vals,
        method="RK45",
        t_eval=t_eval,
        args=(nonlinear_spring_info, material_info, 0.0),
        rtol=rtol,
        atol=atol,
    )

    X_non = nonlinear_solution.y.T

    velocity_x_non = X_non[:, 1::2]
    displacement_x_non = X_non[:, 2::2]

    # Kinetic energy of the last mass (column N; column 0 is the impactor)
    KE_non = 0.5 * velocity_x_non[:, N] ** 2
    KE_lin = 0.5 * velocity_x_lin[:, N] ** 2

    max_KE_non = np.max(KE_non)
    max_KE_lin = np.max(KE_lin)

    max_ke_ratio = max_KE_non / max_KE_lin

    return max_ke_ratio