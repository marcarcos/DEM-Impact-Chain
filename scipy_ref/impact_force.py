def impact_force(C, x1, x2):
    if x1 > x2:
        return C * (x1 - x2) ** 1.5
    return 0