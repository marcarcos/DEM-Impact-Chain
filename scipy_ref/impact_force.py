"""Hertzian-like contact force between the impactor and the first chain mass.

Port of matlab/impact_force.m. Paper Methods 5.1 / SI Eq. S1.13:

    f_H(x0, x1) = C_imp * [x0 - x1]_+^(3/2)

The contact only pushes (compression, x0 > x1) and carries no tension, so the impactor
is free to rebound and separate from the chain.
"""


def impact_force(C, x1, x2):
    """Contact force for impactor position ``x1`` and first-mass position ``x2``.

    Parameters
    ----------
    C  : contact coefficient C_imp (1.2e4 in the paper).
    x1 : displacement of the impactor (left body).
    x2 : displacement of the first chain mass (right body).

    Returns the (non-negative) repulsive force magnitude; 0 when the bodies separate.
    """
    if x1 > x2:
        return C * (x1 - x2) ** 1.5
    return 0
