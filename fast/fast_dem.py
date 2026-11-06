"""
Fast batched version of main_disorder_v2 (same physics as the scipy reference).

Why it is fast:
  1. Adaptive Dormand-Prince 5(4) (the same method as scipy's "RK45") compiled with
     Numba, one simulation per CPU thread. The impact lasts ~0.03 time units, so a
     fixed step sized for it (what the torch version does) wastes >95% of the work
     on the long, soft part of the response. The adaptive step only goes small
     during the impact and in stiff compression.
  2. The linear reference system does not depend on c2/c3, so its max KE is
     computed once and reused for every row.
  3. The max KE of the last mass is taken from the solver's dense output (refined
     with a small search inside the step that holds the peak), so no t_eval grid
     with ~240k output points is stored.

Physics is identical to impact_equation_of_motion_hetero.py:
  index 0 = impactor, 1..N = chain masses, wall after mass N.
  spring j (0..N-1) connects mass j+1 to mass j+2 (or the wall for j = N-1);
  force f(d) = c1 d for d >= 0, c1 d - c2 d^2 + c3 d^3 for d < 0, d = u_right - u_left.
  Hertzian contact impactor/mass 1: C (overlap)^1.5. Dampers 2*zeta on each spring.
"""
import math
import numpy as np
from numba import njit, prange

# ---- model constants (same as main_disorder_v2) ----
N = 20
LAYERS = 4
M_LAYER = 1.0
M0_IMPACTOR = N * M_LAYER / 2.0
M_IMPACTOR = 0.01 * M0_IMPACTOR
C_IMPACT = 1.2e4
ZETA = 0.02
V_IMPACTOR = 0.25
CYCLES = 25
T_END = (2.0 * np.pi * np.sqrt(M0_IMPACTOR)) * CYCLES  # dt_cyc0 * cycles

# ---- Dormand-Prince 5(4) tableau (identical to scipy RK45) ----
A21 = 1 / 5
A31, A32 = 3 / 40, 9 / 40
A41, A42, A43 = 44 / 45, -56 / 15, 32 / 9
A51, A52, A53, A54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
A61, A62, A63, A64, A65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
B1, B3, B4, B5, B6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
E1, E3, E4, E5, E6, E7 = -71 / 57600, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40

# dense output matrix (scipy RK45.P), rows = stages 1..7, cols = theta^1..theta^4
P = np.array([
    [1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
    [0, 0, 0, 0],
    [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
    [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
    [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
    [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
    [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
])


@njit(cache=True, fastmath=False, inline="always")
def _spring(c1, c2, c3, d):
    if d >= 0.0:
        return c1 * d
    return c1 * d - c2 * d * d + c3 * d * d * d


@njit(cache=True, fastmath=False)
def _rhs(y, c1, c2, c3, m, m_imp, C_imp, zeta, out):
    """y = [u0..uN, v0..vN]; out = dy/dt. n = N+1 masses."""
    n = y.shape[0] // 2
    # velocities
    for i in range(n):
        out[i] = y[n + i]
    # impact
    ov = y[0] - y[1]
    f_imp = C_imp * ov * math.sqrt(ov) if ov > 0.0 else 0.0
    out[n] = -f_imp / m_imp
    # chain: spring j between mass j+1 and j+2 (wall for j = n-2)
    f_left = 0.0       # force on current mass from its left spring
    damp_left = 0.0    # 2 zeta (v_i - v_{i-1}) for left damper (0 for mass 1)
    for i in range(1, n):
        j = i - 1
        if i < n - 1:
            d = y[i + 1] - y[i]
            dv = y[n + i + 1] - y[n + i]
        else:
            d = -y[i]
            dv = -y[n + i]
        f_r = _spring(c1[j], c2[j], c3[j], d)
        damp_r = 2.0 * zeta * dv
        acc = f_r + f_left + damp_r - damp_left
        if i == 1:
            acc += f_imp
        out[n + i] = acc / m
        f_left = -f_r
        damp_left = damp_r


@njit(cache=True, fastmath=False)
def _dense_v(P, K, h, idx, y0, theta):
    """Dense-output value of component idx at theta in [0,1]."""
    t1 = theta
    t2 = t1 * theta
    t3 = t2 * theta
    t4 = t3 * theta
    s = 0.0
    for st in range(7):
        q = P[st, 0] * t1 + P[st, 1] * t2 + P[st, 2] * t3 + P[st, 3] * t4
        s += K[st, idx] * q
    return y0[idx] + h * s


@njit(cache=True, fastmath=False)
def simulate_max_ke(c1, c2, c3, t_end, rtol, atol, max_steps, n_sub):
    """Integrate one system; return (max KE of last mass, accepted steps, status).
    status 0 = ok, 1 = hit max_steps, 2 = step size underflow / non-finite."""
    n = c1.shape[0] + 1
    dim = 2 * n
    idx_v = 2 * n - 1  # velocity of last mass
    m, m_imp, C_imp, zeta = M_LAYER, M_IMPACTOR, C_IMPACT, ZETA

    y = np.zeros(dim)
    y[n] = V_IMPACTOR
    ynew = np.zeros(dim)
    ytmp = np.zeros(dim)
    K = np.zeros((7, dim))

    _rhs(y, c1, c2, c3, m, m_imp, C_imp, zeta, K[0])

    # initial step (scipy select_initial_step, order 4)
    d0 = 0.0
    d1 = 0.0
    for k in range(dim):
        sc = atol + rtol * abs(y[k])
        d0 += (y[k] / sc) ** 2
        d1 += (K[0, k] / sc) ** 2
    d0 = math.sqrt(d0 / dim)
    d1 = math.sqrt(d1 / dim)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    for k in range(dim):
        ytmp[k] = y[k] + h0 * K[0, k]
    _rhs(ytmp, c1, c2, c3, m, m_imp, C_imp, zeta, K[1])
    d2 = 0.0
    for k in range(dim):
        sc = atol + rtol * abs(y[k])
        d2 += ((K[1, k] - K[0, k]) / sc) ** 2
    d2 = math.sqrt(d2 / dim) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / 5.0)
    h = min(100 * h0, h1)

    t = 0.0
    max_v2 = 0.0
    steps = 0
    status = 0
    while t < t_end:
        if steps >= max_steps:
            status = 1
            break
        if t + h > t_end:
            h = t_end - t
        # stages
        for k in range(dim):
            ytmp[k] = y[k] + h * (A21 * K[0, k])
        _rhs(ytmp, c1, c2, c3, m, m_imp, C_imp, zeta, K[1])
        for k in range(dim):
            ytmp[k] = y[k] + h * (A31 * K[0, k] + A32 * K[1, k])
        _rhs(ytmp, c1, c2, c3, m, m_imp, C_imp, zeta, K[2])
        for k in range(dim):
            ytmp[k] = y[k] + h * (A41 * K[0, k] + A42 * K[1, k] + A43 * K[2, k])
        _rhs(ytmp, c1, c2, c3, m, m_imp, C_imp, zeta, K[3])
        for k in range(dim):
            ytmp[k] = y[k] + h * (A51 * K[0, k] + A52 * K[1, k] + A53 * K[2, k] + A54 * K[3, k])
        _rhs(ytmp, c1, c2, c3, m, m_imp, C_imp, zeta, K[4])
        for k in range(dim):
            ytmp[k] = y[k] + h * (A61 * K[0, k] + A62 * K[1, k] + A63 * K[2, k]
                                  + A64 * K[3, k] + A65 * K[4, k])
        _rhs(ytmp, c1, c2, c3, m, m_imp, C_imp, zeta, K[5])
        for k in range(dim):
            ynew[k] = y[k] + h * (B1 * K[0, k] + B3 * K[2, k] + B4 * K[3, k]
                                  + B5 * K[4, k] + B6 * K[5, k])
        _rhs(ynew, c1, c2, c3, m, m_imp, C_imp, zeta, K[6])
        # error norm (RMS, scipy style)
        err = 0.0
        for k in range(dim):
            e = h * (E1 * K[0, k] + E3 * K[2, k] + E4 * K[3, k] + E5 * K[4, k]
                     + E6 * K[5, k] + E7 * K[6, k])
            sc = atol + rtol * max(abs(y[k]), abs(ynew[k]))
            err += (e / sc) ** 2
        err = math.sqrt(err / dim)

        if not math.isfinite(err):
            h *= 0.2
            if h < 1e-14:
                status = 2
                break
            continue

        if err <= 1.0:
            # ---- accepted: track max KE of last mass using dense output ----
            best = 0.0
            best_j = 0
            for j in range(1, n_sub + 1):
                v = _dense_v(P, K, h, idx_v, y, j / n_sub)
                if v * v > best:
                    best = v * v
                    best_j = j
            if best > max_v2:
                # golden-section refine around the best sample
                a = (best_j - 1) / n_sub
                b = min(best_j + 1, n_sub) / n_sub
                gr = 0.6180339887498949
                x1 = b - gr * (b - a)
                x2 = a + gr * (b - a)
                v = _dense_v(P, K, h, idx_v, y, x1)
                f1 = v * v
                v = _dense_v(P, K, h, idx_v, y, x2)
                f2 = v * v
                for _ in range(30):
                    if f1 > f2:
                        b = x2
                        x2 = x1
                        f2 = f1
                        x1 = b - gr * (b - a)
                        v = _dense_v(P, K, h, idx_v, y, x1)
                        f1 = v * v
                    else:
                        a = x1
                        x1 = x2
                        f1 = f2
                        x2 = a + gr * (b - a)
                        v = _dense_v(P, K, h, idx_v, y, x2)
                        f2 = v * v
                    if b - a < 1e-9:
                        break
                best = max(best, f1, f2)
                max_v2 = best
            t += h
            for k in range(dim):
                y[k] = ynew[k]
                K[0, k] = K[6, k]  # FSAL
            steps += 1
            fac = 10.0 if err == 0.0 else min(10.0, 0.9 * err ** -0.2)
            h *= fac
        else:
            h *= max(0.2, 0.9 * err ** -0.2)
            if h < 1e-14:
                status = 2
                break
    return 0.5 * max_v2, steps, status  # KE = 0.5 v^2, as in the reference


@njit(cache=True, parallel=True)
def _batch(C1, C2, C3, t_end, rtol, atol, max_steps, n_sub, out_ke, out_steps, out_status):
    for b in prange(C2.shape[0]):
        ke, st, ss = simulate_max_ke(C1[b], C2[b], C3[b], t_end, rtol, atol, max_steps, n_sub)
        out_ke[b] = ke
        out_steps[b] = st
        out_status[b] = ss


def _expand(c):
    """(B, 4) layer coefficients -> (B, 20) per-spring coefficients."""
    c = np.atleast_2d(np.asarray(c, dtype=np.float64))
    return np.ascontiguousarray(np.repeat(c, N // c.shape[1], axis=1))


def batch_max_ke(c2, c3, rtol=1e-12, atol=1e-12, max_steps=2_000_000, n_sub=8):
    C2 = _expand(c2)
    C3 = _expand(c3)
    B = C2.shape[0]
    C1 = np.ones_like(C2)
    ke = np.empty(B)
    steps = np.empty(B, dtype=np.int64)
    status = np.empty(B, dtype=np.int64)
    _batch(C1, C2, C3, T_END, rtol, atol, max_steps, n_sub, ke, steps, status)
    return ke, steps, status


_LIN_CACHE = {}


def linear_max_ke(rtol=1e-11, atol=1e-13):
    """Max KE of the linear chain. Independent of c2/c3, so compute once."""
    key = (rtol, atol)
    if key not in _LIN_CACHE:
        ke, _, st = batch_max_ke(np.zeros((1, LAYERS)), np.zeros((1, LAYERS)), rtol, atol)
        assert st[0] == 0
        _LIN_CACHE[key] = float(ke[0])
    return _LIN_CACHE[key]


def main_disorder_v2(c2, c3, rtol=1e-12, atol=1e-12, return_info=False):
    """Drop-in replacement. c2, c3: shape (4,) or (B, 4). Returns max KE ratio (scalar or (B,))."""
    single = np.ndim(c2) == 1
    ke, steps, status = batch_max_ke(c2, c3, rtol, atol)
    ratio = ke / linear_max_ke()
    ratio[status != 0] = np.nan
    if single:
        ratio = ratio[0]
    if return_info:
        return ratio, steps, status
    return ratio