"""
fast_dem -- fast, batched discrete-element model (DEM) of an impacted nonlinear chain.

This is the production simulator used to generate datasets. It solves exactly the same
equations as the original MATLAB code (``matlab/impact_equation_of_motion_hetero.m``) and
its line-by-line scipy port (``scipy_ref/``), but is ~100-1000x faster per design.
Agreement with MATLAB is checked by ``rhs_check.py`` (equations of motion, ~1e-14) and
``validate_all.py`` (final KE ratio, median ~6e-7 relative).

Physical model  (Tamur et al., main text Eq. 1, 2, 5; SI Eq. S1.6, S1.13, S1.14)
-------------------------------------------------------------------------------------
All quantities are non-dimensional (SI-1): unit-cell mass m = 1, linear spring stiffness
c1 = 1, lattice spacing a = 1.

* N = 20 unit cells (masses 1..N), fixed wall to the right of mass N.
* Mass 0 is a rigid impactor of mass M moving right with velocity V at t = 0.
  Impactor and mass 1 interact through a Hertzian, tension-free contact spring
      f_H = C_imp * [u0 - u1]_+^(3/2),          C_imp = 1.2e4.
* Spring j (j = 0..N-1) joins mass j+1 to mass j+2 (j = N-1 joins mass N to the wall).
  With d = u_right - u_left (d < 0 means compression, Delta x = -d > 0), the force it
  pulls the left mass with is
      F(d) = c1 d                         for d >= 0  (tension: linear only)
      F(d) = c1 d - c2 d^2 + c3 d^3       for d <  0  (compression)
  which is just  -f(Delta x)  with the paper's  f(Dx) = Dx + c2 Dx^2 + c3 Dx^3  (Eq. 1).
  The right mass feels -F(d).
* A linear viscous damper 2*zeta*(v_right - v_left), zeta = 0.02 (SI-1.2), sits in
  parallel with every chain spring (including the wall spring). There is no damper
  between impactor and mass 1.
* Spring families: the 20 springs are split into 4 consecutive families A,B,C,D of 5
  springs each, so a design is just (c2, c3) with shape (4,) each -> repeated to (20,).

Performance metric
------------------
    KE ratio = max_t KE_N(t)_nonlinear / max_t KE_N(t)_linear,    KE_N = v_N^2 / 2,

where the linear chain has c2 = c3 = 0 (same impactor, damping, contact). Lower is better.

Time window and output grid
---------------------------
f0 = sqrt(1/M0) / (2 pi) with M0 = N/2 (half the chain mass) and T = 25 / f0 ~ 496.7.
The paper/MATLAB take the max of KE on a discrete output grid. Here the max is taken
continuously from the RK45 dense-output polynomial inside every accepted step (with a
golden-section refinement), i.e. the limit of an infinitely fine grid. We checked
that on the paper's designs the grid choice changes the ratio by < 0.3 %
(see REPLICATION.md).

Why it is fast
--------------
1. Adaptive Dormand-Prince 5(4) (the same method, tableau, error norm and step-size
   controller as scipy's ``RK45``) compiled with Numba, one simulation per CPU thread.
   The impact itself lasts ~0.03 time units, so a fixed step sized for it would waste
   >95 % of the work on the long, soft part of the response.
2. The linear reference system does not depend on c2/c3, so its max KE is computed
   once per (impactor mass, impactor velocity, tolerance) and cached.
3. No dense t_eval grid is stored (MATLAB/scipy store ~240k output points per run).

State layout
------------
fast_dem uses y = [u0..uN, v0..vN] (all displacements, then all velocities).
MATLAB / scipy_ref use the interleaved layout x = [u0, v0, u1, v1, ...].
In both, index 0 is the impactor.

Public API
----------
main_disorder_v2(c2, c3)            -> KE ratio (drop-in for scipy_ref.main_disorder_v2)
batch_max_ke(c2, c3, ...)           -> raw max KE of the last mass (+ step counts, status)
linear_max_ke(...)                  -> max KE of the linear reference chain (cached)
"""
import math

import numpy as np
from numba import njit, prange

# --------------------------------------------------------------------------------------
# Model constants (identical to matlab/main_disorder.m and scipy_ref/main_disorder_v2.py)
# --------------------------------------------------------------------------------------
N = 20                              # number of unit cells (masses) in the chain
LAYERS = 4                          # spring families A, B, C, D (N // LAYERS springs each)
M_LAYER = 1.0                       # non-dimensional unit-cell mass m
M0_IMPACTOR = N * M_LAYER / 2.0     # reference impactor mass M0 = half the chain mass
M_IMPACTOR = 0.01 * M0_IMPACTOR     # design-point impactor mass  M / M0 = 0.01
C_IMPACT = 1.2e4                    # Hertzian contact coefficient C_imp (SI-1.3)
ZETA = 0.02                         # damping ratio zeta of every chain damper (SI-1.2)
V_IMPACTOR = 0.25                   # design-point impactor velocity V / V0 = 0.25 (V0 = 1)
CYCLES = 25                         # simulated duration in periods of f0
T_END = (2.0 * np.pi * np.sqrt(M0_IMPACTOR)) * CYCLES  # T = 25 / f0 ~ 496.7

# --------------------------------------------------------------------------------------
# Dormand-Prince 5(4) Butcher tableau (identical to scipy.integrate.RK45)
#   A**: stage coefficients, B*: 5th-order solution weights (B2 = B7 = 0),
#   E*: difference between 5th- and 4th-order weights -> local error estimate.
# --------------------------------------------------------------------------------------
A21 = 1 / 5
A31, A32 = 3 / 40, 9 / 40
A41, A42, A43 = 44 / 45, -56 / 15, 32 / 9
A51, A52, A53, A54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
A61, A62, A63, A64, A65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
B1, B3, B4, B5, B6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
E1, E3, E4, E5, E6, E7 = -71 / 57600, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40

# Dense-output (continuous extension) matrix, identical to scipy RK45.P.
# Rows = stages 1..7, cols = coefficients of theta^1..theta^4, where theta in [0, 1] is the
# fractional position inside the step:  y(t + theta h) = y + h * sum_s K_s * (P[s] . theta^k)
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
    """Force a spring exerts on its LEFT mass for elongation d = u_right - u_left.

    d >= 0 (tension):     c1 d                      (linear only, paper Eq. S1.6)
    d <  0 (compression): c1 d - c2 d^2 + c3 d^3    (= -f(Dx) with Dx = -d > 0, Eq. 1)
    The right mass feels the opposite force.
    """
    if d >= 0.0:
        return c1 * d
    return c1 * d - c2 * d * d + c3 * d * d * d


@njit(cache=True, fastmath=False)
def _rhs(y, c1, c2, c3, m, m_imp, C_imp, zeta, out):
    """Equations of motion (paper Eq. 5 / SI Eq. S1.14), written into ``out`` in place.

    Parameters
    ----------
    y      : (2n,) state [u0..u_{n-1}, v0..v_{n-1}], n = N + 1 masses, index 0 = impactor.
    c1,c2,c3 : (N,) per-spring polynomial coefficients; spring j joins mass j+1 to j+2
             (spring N-1 joins mass N to the wall at u = 0).
    m      : chain unit-cell mass.     m_imp : impactor mass.
    C_imp  : Hertzian contact coefficient.  zeta : damping ratio.
    out    : (2n,) output array, receives dy/dt = [v, a].

    Mathematically identical to matlab/impact_equation_of_motion_hetero.m (checked by
    rhs_check.py) but organised as a single sweep in which each spring force is
    computed once and reused (with opposite sign) for the next mass.
    """
    n = y.shape[0] // 2
    # du/dt = v
    for i in range(n):
        out[i] = y[n + i]
    # Hertzian contact: only acts while the impactor overlaps mass 1 (u0 > u1).
    ov = y[0] - y[1]
    f_imp = C_imp * ov * math.sqrt(ov) if ov > 0.0 else 0.0   # C * ov^(3/2)
    out[n] = -f_imp / m_imp                                    # impactor is pushed back
    # Chain masses i = 1..n-1. Carry the left-spring force / damper term from the
    # previous iteration so every spring is evaluated exactly once.
    f_left = 0.0       # force on current mass from its left spring (0 for mass 1: contact instead)
    damp_left = 0.0    # 2 zeta (v_i - v_{i-1}) of the left damper (0 for mass 1: no damper)
    for i in range(1, n):
        j = i - 1      # index of the spring to the RIGHT of mass i
        if i < n - 1:
            d = y[i + 1] - y[i]            # elongation of right spring
            dv = y[n + i + 1] - y[n + i]   # elongation rate of right damper
        else:
            d = -y[i]                      # last spring is anchored to the wall (u = 0)
            dv = -y[n + i]
        f_r = _spring(c1[j], c2[j], c3[j], d)
        damp_r = 2.0 * zeta * dv
        acc = f_r + f_left + damp_r - damp_left
        if i == 1:
            acc += f_imp                   # contact force pushes mass 1 forward
        out[n + i] = acc / m
        f_left = -f_r                      # Newton's 3rd law for the next mass
        damp_left = damp_r


@njit(cache=True, fastmath=False)
def _dense_v(P, K, h, idx, y0, theta):
    """Value of state component ``idx`` at t + theta*h from the RK45 dense-output polynomial.

    K holds the 7 stage derivatives of the current (accepted) step, y0 the state at its
    start. theta in [0, 1]. Same 4th-order interpolant as scipy's RK45 dense output.
    """
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
def simulate_max_ke(c1, c2, c3, m_imp, v_imp, t_end, rtol, atol, max_steps, n_sub):
    """Integrate one impact simulation and return the peak KE of the last mass.

    Parameters
    ----------
    c1, c2, c3 : (N,) per-spring coefficients.
    m_imp, v_imp : impactor mass and initial velocity.
    t_end      : end time of the simulation.
    rtol, atol : RK45 relative / absolute tolerance (scipy-style mixed error norm).
    max_steps  : safety cap on the number of accepted steps.
    n_sub      : samples per step used to locate the KE peak in the dense output before
                 golden-section refinement.

    Returns
    -------
    (max_ke, steps, status) where max_ke = max_t 0.5 * v_N(t)^2 and
    status 0 = ok, 1 = hit max_steps, 2 = step size underflow / non-finite error.

    The integrator replicates scipy.integrate.RK45 (initial-step heuristic, RMS error
    norm, safety factor 0.9, step growth limits [0.2, 10]) so results track the scipy
    reference to roughly the requested tolerance.
    """
    n = c1.shape[0] + 1          # masses including impactor
    dim = 2 * n
    idx_v = 2 * n - 1            # index of v_N (velocity of last chain mass)
    m, C_imp, zeta = M_LAYER, C_IMPACT, ZETA

    y = np.zeros(dim)
    y[n] = v_imp                 # impactor starts moving, everything else at rest
    ynew = np.zeros(dim)
    ytmp = np.zeros(dim)
    K = np.zeros((7, dim))       # RK stage derivatives (K[0] reused via FSAL)

    _rhs(y, c1, c2, c3, m, m_imp, C_imp, zeta, K[0])

    # ---- initial step size: scipy.integrate._ivp.common.select_initial_step (order 4) ----
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
    max_v2 = 0.0     # running max of v_N^2
    steps = 0
    status = 0
    while t < t_end:
        if steps >= max_steps:
            status = 1
            break
        if t + h > t_end:
            h = t_end - t          # land exactly on t_end
        # ---- Dormand-Prince stages 2..6 ----
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
        # ---- 5th-order solution and 7th stage (first stage of next step, FSAL) ----
        for k in range(dim):
            ynew[k] = y[k] + h * (B1 * K[0, k] + B3 * K[2, k] + B4 * K[3, k]
                                  + B5 * K[4, k] + B6 * K[5, k])
        _rhs(ynew, c1, c2, c3, m, m_imp, C_imp, zeta, K[6])
        # ---- RMS error norm, scipy style ----
        err = 0.0
        for k in range(dim):
            e = h * (E1 * K[0, k] + E3 * K[2, k] + E4 * K[3, k] + E5 * K[4, k]
                     + E6 * K[5, k] + E7 * K[6, k])
            sc = atol + rtol * max(abs(y[k]), abs(ynew[k]))
            err += (e / sc) ** 2
        err = math.sqrt(err / dim)

        if not math.isfinite(err):
            # blow-up inside the trial step: shrink hard and retry
            h *= 0.2
            if h < 1e-14:
                status = 2
                break
            continue

        if err <= 1.0:
            # ---- step accepted: look for the KE peak of the last mass inside it ----
            # 1) coarse scan of the dense output at n_sub points
            best = 0.0
            best_j = 0
            for j in range(1, n_sub + 1):
                v = _dense_v(P, K, h, idx_v, y, j / n_sub)
                if v * v > best:
                    best = v * v
                    best_j = j
            if best > max_v2:
                # 2) golden-section refine around the best sample (only when it could
                #    raise the running max, so this is rare and cheap)
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
                K[0, k] = K[6, k]  # FSAL: last stage of this step = first of the next
            steps += 1
            # step growth, scipy: factor = min(10, 0.9 * err^(-1/5))
            fac = 10.0 if err == 0.0 else min(10.0, 0.9 * err ** -0.2)
            h *= fac
        else:
            # step rejected: shrink, scipy: factor = max(0.2, 0.9 * err^(-1/5))
            h *= max(0.2, 0.9 * err ** -0.2)
            if h < 1e-14:
                status = 2
                break
    return 0.5 * max_v2, steps, status  # KE = 0.5 m v^2 with m = 1, as in the reference


@njit(cache=True, parallel=True)
def _batch(C1, C2, C3, m_imp, v_imp, t_end, rtol, atol, max_steps, n_sub,
           out_ke, out_steps, out_status):
    """Run simulate_max_ke for every row of C1/C2/C3 ((B, N) arrays) in parallel."""
    for b in prange(C2.shape[0]):
        ke, st, ss = simulate_max_ke(C1[b], C2[b], C3[b], m_imp, v_imp, t_end,
                                     rtol, atol, max_steps, n_sub)
        out_ke[b] = ke
        out_steps[b] = st
        out_status[b] = ss


def _expand(c):
    """(B, 4) family coefficients -> (B, 20) per-spring coefficients (each repeated 5x).

    Also accepts (B, 20) (already per-spring; repeated 1x) or a single (4,) design.
    """
    c = np.atleast_2d(np.asarray(c, dtype=np.float64))
    return np.ascontiguousarray(np.repeat(c, N // c.shape[1], axis=1))


def batch_max_ke(c2, c3, rtol=1e-12, atol=1e-12, max_steps=2_000_000, n_sub=8,
                 m_imp=M_IMPACTOR, v_imp=V_IMPACTOR, t_end=T_END):
    """Peak KE of the last mass for a batch of designs (all CPU threads).

    Parameters
    ----------
    c2, c3 : (B, 4) or (4,) family coefficients (or (B, 20) per-spring).
    rtol, atol : integrator tolerances (default 1e-12, same as MATLAB/scipy reference).
    max_steps, n_sub : see simulate_max_ke.
    m_imp, v_imp : impactor mass and velocity (default: paper design point
             M/M0 = 0.01, V/V0 = 0.25). Vary these for the Fig. 2 impact-condition maps.
    t_end  : simulated duration (default T = 25/f0).

    Returns
    -------
    ke (B,), steps (B,) int, status (B,) int  -- see simulate_max_ke for status codes.
    """
    C2 = _expand(c2)
    C3 = _expand(c3)
    if C2.shape != C3.shape:
        raise ValueError(f"c2 and c3 shapes differ: {C2.shape} vs {C3.shape}")
    B = C2.shape[0]
    C1 = np.ones_like(C2)        # linear stiffness is uniform (c1 = 1) in the paper
    ke = np.empty(B)
    steps = np.empty(B, dtype=np.int64)
    status = np.empty(B, dtype=np.int64)
    _batch(C1, C2, C3, float(m_imp), float(v_imp), float(t_end), rtol, atol,
           max_steps, n_sub, ke, steps, status)
    return ke, steps, status


_LIN_CACHE = {}


def linear_max_ke(rtol=1e-11, atol=1e-13, m_imp=M_IMPACTOR, v_imp=V_IMPACTOR, t_end=T_END):
    """Peak KE of the last mass of the LINEAR chain (c2 = c3 = 0), the KE-ratio denominator.

    It does not depend on the design, only on the impact condition, so it is computed
    once per (tolerances, m_imp, v_imp, t_end) and cached. Slightly tighter default
    tolerances than the nonlinear run because every ratio in a dataset shares it.
    """
    key = (rtol, atol, float(m_imp), float(v_imp), float(t_end))
    if key not in _LIN_CACHE:
        ke, _, st = batch_max_ke(np.zeros((1, LAYERS)), np.zeros((1, LAYERS)), rtol, atol,
                                 m_imp=m_imp, v_imp=v_imp, t_end=t_end)
        if st[0] != 0:
            raise RuntimeError(f"linear reference simulation failed (status {st[0]})")
        _LIN_CACHE[key] = float(ke[0])
    return _LIN_CACHE[key]


def main_disorder_v2(c2, c3, rtol=1e-12, atol=1e-12, return_info=False,
                     m_imp=M_IMPACTOR, v_imp=V_IMPACTOR):
    """KE ratio max(KE_non) / max(KE_lin) at the last mass -- the paper's objective J.

    Drop-in replacement for scipy_ref/main_disorder_v2.py (and matlab/main_disorder_fn.m).

    Parameters
    ----------
    c2, c3 : (4,) for one design -> scalar result, or (B, 4) -> (B,) array.
    rtol, atol : tolerances of the nonlinear runs.
    return_info : also return the accepted-step counts and status codes.
    m_imp, v_imp : impact condition (default: paper design point).

    Failed integrations (status != 0) are returned as NaN.
    """
    single = np.ndim(c2) == 1
    ke, steps, status = batch_max_ke(c2, c3, rtol, atol, m_imp=m_imp, v_imp=v_imp)
    ratio = ke / linear_max_ke(m_imp=m_imp, v_imp=v_imp)
    ratio[status != 0] = np.nan
    if single:
        ratio = ratio[0]
    if return_info:
        return ratio, steps, status
    return ratio
