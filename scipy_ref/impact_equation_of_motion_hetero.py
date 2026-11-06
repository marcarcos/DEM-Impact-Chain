import numpy as np

from impact_force import impact_force


def impact_equation_of_motion_hetero(
    t,
    x,
    nonlinear_spring_info,
    material_info,
    F,
):
    num_mass = len(x) // 2
    dx = np.zeros_like(x, dtype=float)

    c1 = nonlinear_spring_info[:, 0]
    c2 = nonlinear_spring_info[:, 1]
    c3 = nonlinear_spring_info[:, 2]

    m = material_info[0]
    m_impact = material_info[1]
    C_impact = material_info[2]
    zeta = material_info[3]

    # Impactor
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

    # First mass in the chain
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

    # Middle masses
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

    # Last mass
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