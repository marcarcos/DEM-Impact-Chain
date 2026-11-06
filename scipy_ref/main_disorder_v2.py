import numpy as np
from scipy.integrate import solve_ivp

from impact_equation_of_motion_hetero import impact_equation_of_motion_hetero


def main_disorder_v2(c2, c3):
    c2 = np.asarray(c2, dtype=float)
    c3 = np.asarray(c3, dtype=float)

    c1 = np.ones(4)

    C1 = np.repeat(c1, 5)
    C2 = np.repeat(c2, 5)
    C3 = np.repeat(c3, 5)

    zeta = 0.02

    N = 20
    m_layer = 1.0

    M0_impactor = N * m_layer / 2  # More accurate than matlab's implementation
    C_impact = 1.2e4

    M_impactor = 0.01 * M0_impactor
    V_impactor = 0.25

    # Solver setup
    k_f = np.max(np.concatenate(([1.0], 1 + 2 * c2 + 3 * c3)))

    cycles = 25
    output_per_cycle = 100

    f0 = (1 / (2 * np.pi)) * np.sqrt(1 / M0_impactor)
    dt_cyc0 = 1 / f0

    f = (1 / (2 * np.pi)) * np.sqrt(k_f / M0_impactor)
    dt_cyc = 1 / f

    dt = dt_cyc / output_per_cycle
    T = dt_cyc0 * cycles

    num_points = int(np.floor(T / dt)) + 1
    t_eval = np.arange(num_points) * dt

    # Material info and initial conditions
    material_info = np.array(
        [m_layer, M_impactor, C_impact, zeta],
        dtype=float,
    )

    initial_vals = np.zeros(2 * (N + 1))
    initial_vals[1] = V_impactor

    atol = np.full(2 * (N + 1), 1e-12)
    rtol = 1e-12

    # Linear system
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

    velocity_x_lin = X_lin[:, 1::2]
    displacement_x_lin = X_lin[:, 2::2]

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

    # Kinetic energy at the last mass
    KE_non = 0.5 * velocity_x_non[:, N] ** 2
    KE_lin = 0.5 * velocity_x_lin[:, N] ** 2

    max_KE_non = np.max(KE_non)
    max_KE_lin = np.max(KE_lin)

    max_ke_ratio = max_KE_non / max_KE_lin

    return max_ke_ratio