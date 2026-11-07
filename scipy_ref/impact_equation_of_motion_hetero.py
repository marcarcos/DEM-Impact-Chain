"""Equations of motion of the impacted heterogeneous chain (scipy port of the MATLAB code).

Line-by-line port of matlab/impact_equation_of_motion_hetero.m, kept deliberately
close to the original so it can serve as a reference. The fast Numba version
(fast/fast_dem.py::_rhs) is checked against this function and MATLAB by fast/rhs_check.py.

Model (paper Eq. 5 / SI Eq. S1.14), all non-dimensional:

    M x0''  = -f_H(x0, x1)                                                   impactor
    m x1''  =  F_1(x2 - x1) + f_H(x0, x1) + 2 zeta (v2 - v1)                 first mass
    m xi''  =  F_i(x_{i+1} - x_i) - F_{i-1}(x_i - x_{i-1})
               + 2 zeta (v_{i+1} - 2 v_i + v_{i-1})                          2 <= i <= N-1
    m xN''  =  F_N(0 - xN) - F_{N-1}(xN - x_{N-1})
               - 2 zeta (vN - v_{N-1}) - 2 zeta vN                           last mass (wall)

with the spring force (d = elongation = u_right - u_left; d < 0 is compression)

    F(d) = c1 d                       if d >= 0
    F(d) = c1 d - c2 d^2 + c3 d^3     if d <  0      (= -f(-d), f from paper Eq. 1)

State layout (interleaved, as in MATLAB):  x = [u0, v0, u1, v1, ..., uN, vN],
index 0 = impactor, 1..N = chain masses. Spring k (0-based, k = 0..N-1) sits to the
right of chain mass k+1, so mass i has right spring i-1 and left spring i-2.
"""
import numpy as np

from impact_force import impact_force


def impact_equation_of_motion_hetero(
    t,
    x,
    nonlinear_spring_info,
    material_info,
    F,
):
    """Right-hand side dx/dt for scipy.integrate.solve_ivp.

    Parameters
    ----------
    t : float
        Time (unused; the system is autonomous). Kept for the solve_ivp signature.
    x : (2(N+1),) array
        Interleaved state [u0, v0, u1, v1, ..., uN, vN]; index 0 is the impactor.
    nonlinear_spring_info : (N, 3) array
        Columns c1, c2, c3 of each chain spring (row k = spring right of mass k+1).
    material_info : (4,) array
        [m (unit-cell mass), M (impactor mass), C_imp (contact coeff.), zeta (damping)].
    F : float
        External force on the impactor (0 in all paper simulations).

    Returns
    -------
    dx : (2(N+1),) array, interleaved [v0, a0, v1, a1, ...].
    """
    num_mass = len(x) // 2  # N + 1 (chain masses + impactor)
    dx = np.zeros_like(x, dtype=float)

    c1 = nonlinear_spring_info[:, 0]
    c2 = nonlinear_spring_info[:, 1]
    c3 = nonlinear_spring_info[:, 2]

    m = material_info[0]
    m_impact = material_info[1]
    C_impact = material_info[2]
    zeta = material_info[3]

    # Impactor (i = 0): only the contact force acts on it
    i = 0

    dx[2 * i] = x[2 * i + 1]

    dx[2 * i + 1] = (
        F
        - impact_force(
            C_impact,
            x[2 * i],
            x[2 * (i + 1)],
        )
    ) / m_impact

    # First mass in the chain (i = 1): right spring + contact force + right damper only
    i = 1

    dx[2 * i] = x[2 * i + 1]

    if x[2 * (i + 1)] >= x[2 * i]:
        spring_force = c1[i - 1] * (
            x[2 * (i + 1)] - x[2 * i]
        )
    else:
        displacement = (
            x[2 * (i + 1)] - x[2 * i]
        )

        spring_force = (
            c1[i - 1] * displacement
            - c2[i - 1] * displacement**2
            + c3[i - 1] * displacement**3
        )

    dx[2 * i + 1] = (
        spring_force
        + impact_force(
            C_impact,
            x[2 * (i - 1)],
            x[2 * i],
        )
        - 2 * zeta * (
            -x[2 * (i + 1) + 1]
            + x[2 * i + 1]
        )
    ) / m

    # Middle masses (i = 2..N-1): right spring c[i-1], left spring c[i-2], two dampers
    for i in range(2, num_mass - 1):
        dx[2 * i] = x[2 * i + 1]

        # Force from right spring
        right_displacement = x[2 * (i + 1)] - x[2 * i]

        if x[2 * (i + 1)] >= x[2 * i]:
            force_spring1 = c1[i - 1] * right_displacement
        else:
            force_spring1 = (
                c1[i - 1] * right_displacement
                - c2[i - 1] * right_displacement**2
                + c3[i - 1] * right_displacement**3
            )

        # Force from left spring
        left_displacement = x[2 * i] - x[2 * (i - 1)]

        if x[2 * i] >= x[2 * (i - 1)]:
            force_spring2 = -c1[i - 2] * left_displacement
        else:
            force_spring2 = (
                -c1[i - 2] * left_displacement
                + c2[i - 2] * left_displacement**2
                - c3[i - 2] * left_displacement**3
            )

        dx[2 * i + 1] = (
            force_spring1
            + force_spring2
            - 2 * zeta * (
                -x[2 * (i + 1) + 1]
                + 2 * x[2 * i + 1]
                - x[2 * (i - 1) + 1]
            )
        ) / m

    # Last mass (i = N): right spring c[N-1] is anchored to the fixed wall at u = 0
    i = num_mass - 1

    dx[2 * i] = x[2 * i + 1]

    # Force from right spring (wall)
    right_displacement = -x[2 * i]

    if 0 >= x[2 * i]:
        force_spring1 = c1[i - 1] * right_displacement
    else:
        force_spring1 = (
            c1[i - 1] * right_displacement
            - c2[i - 1] * right_displacement**2
            + c3[i - 1] * right_displacement**3
        )

    # Force from left spring
    left_displacement = x[2 * i] - x[2 * (i - 1)]

    if x[2 * i] >= x[2 * (i - 1)]:
        force_spring2 = -c1[i - 2] * left_displacement
    else:
        force_spring2 = (
            -c1[i - 2] * left_displacement
            + c2[i - 2] * left_displacement**2
            - c3[i - 2] * left_displacement**3
        )

    dx[2 * i + 1] = (
        force_spring1
        + force_spring2
        - 2 * zeta * (
            x[2 * i + 1]
            - x[2 * (i - 1) + 1]
        )
        - 2 * zeta * x[2 * i + 1]
    ) / m

    return dx