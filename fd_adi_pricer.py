"""
fd_adi_pricer.py -- finite-difference (MCS-ADI) American-put pricer under the Heston model.

Prices European and American puts under the Heston stochastic-volatility model with
constant coefficients, on a non-uniform (S, v) grid using the Modified Craig-Sneyd
ADI time-integration scheme.  American options are handled either by Bermudan
projection onto the payoff after each step (policy=0, default) or by a penalized
policy-iteration on the linear-complementarity problem (policy=1).

The class holds the contract/model parameters as attributes (defaults below), which
can be overridden through a dict passed to the constructor (unknown keys raise
KeyError).  All work that depends only on the parameters -- the grids, the sparse
operators and the two LU factorizations of the implicit ADI stages -- is done once in
the constructor, so the European and the American value share a single setup and the
solved grids are cached; repeated price(S0, V0) queries then cost only an
interpolation.

    from fd_adi_pricer import HestonFDSolver
    solver = HestonFDSolver({"K": 100, "T2": 1.0, "rd": 0.03, "rf": 0.0,
                             "theta": 0.04, "xi": 0.5, "rho": -0.6, "kappa": 2.0})
    eu = solver.price(S0=100, V0=0.04, amer=0)
    am = solver.price(S0=100, V0=0.04, amer=1)   # reuses grids/operators/LUs
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.interpolate import CubicSpline


class HestonFDSolver:
    """MCS-ADI finite-difference solver for European/American Heston puts (constant coefficients)."""

    # Contract / model parameters
    K:     float = 100.0
    T2:    float = 1.0
    rd:    float = 0.03      # domestic / risk-free rate
    rf:    float = 0.0       # foreign rate / dividend yield
    theta: float = 0.04      # long-run variance
    xi:    float = 0.5       # vol-of-vol
    rho:   float = -0.6      # spot/variance correlation
    kappa: float = 2.0       # mean-reversion speed

    # Numerical parameters
    N:         int   = 100           # time steps
    m1:        int   = 200           # S-intervals (v uses m1 // 2)
    B:         float = 0.0           # lower S boundary
    Smax_mult: float = 8.0           # Smax = Smax_mult * K
    Vmax:      float = 5.0
    theta_mcs: float = 1.0 / 3.0     # MCS parameter
    policy:    int   = 1             # American treatment: 1 = penalized policy (default), 0 = Bermudan

    # Penalized-policy constants (used only when policy == 1)
    rho_AM:     float = 1e9
    pp_maxiter: int   = 100

    def __init__(self, params=None):
        if params is not None:
            allowed = {k for klass in type(self).__mro__ for k in getattr(klass, "__annotations__", {})}
            for key, value in params.items():
                if key not in allowed:
                    raise KeyError(f"unknown parameter '{key}'")
                setattr(self, key, value)

        self.Smax = self.Smax_mult * self.K
        self.M1 = self.m1 + 1            # nodes in S
        self.m2 = self.m1 // 2
        self.M2 = self.m2 + 1            # nodes in v
        self.pp_tol = 1.0 / self.rho_AM

        self._build_time_grid()
        self._build_space_grids()
        self._build_operators()
        self._build_payoff()
        self._factorize()

        self._U_cache = {}              # solved value grids keyed by the amer flag

    @staticmethod
    def _spdiags_dense(cols, diags, m, n):
        """Dense m x n matrix from diagonal bands: column `col[k]` is placed on diagonal `d[k]`,
        i.e. A[i, i + d] = col[i + d]."""
        A = np.zeros((m, n))
        for col, d in zip(cols, diags):
            col = np.asarray(col, dtype=float)
            i = np.arange(max(0, -d), min(m, n - d))
            A[i, i + d] = col[i + d]
        return A

    # -- grids -----------------------------------------------------------------
    def _build_time_grid(self):
        self.tt = np.linspace(0.0, self.T2, self.N + 1)
        self.dt = self.T2 / self.N

    def _build_space_grids(self):
        K, B, M1, M2 = self.K, self.B, self.M1, self.M2

        self.Theta_vec = self.theta * np.ones(self.N + 1)

        # non-uniform grid in S, refined around the strike
        T = self.T2
        c = K / 10.0
        Sleft = max(0.5, np.exp(-0.25 * T)) * K
        Sright = K
        xmin = np.arcsinh((B - Sleft) / c)
        xint = (Sright - Sleft) / c
        xmax = xint + np.arcsinh((self.Smax - Sright) / c)
        x = np.linspace(xmin, xmax, M1)
        x1 = x[x <= 0]
        x2 = x[(0 < x) & (x < xint)]
        x3 = x[x >= xint]
        s1 = Sleft + np.sinh(x1) * c
        s2 = Sleft + x2 * c
        s3 = Sright + np.sinh(x3 - xint) * c
        s = np.concatenate([s1, s2, s3])
        s[0] = B
        self.s = s

        # non-uniform grid in v, refined near v = 0
        d = self.Vmax / 500.0
        ymax = np.arcsinh(self.Vmax / d)
        y = np.linspace(0.0, ymax, M2)
        v = d * np.sinh(y)
        v[0] = 0.0
        self.v = v

        self.j2_idx = np.where(v < np.max(self.Theta_vec))[0][-1]
        self.nrep = self.j2_idx + 1               # rows using the central v-derivative
        self.ii = np.arange(1, M1)                # interior S indices
        self.jj = np.arange(0, M2)                # all v indices
        self.mi = self.ii.size
        self.mj = self.jj.size

        self.Xt = np.diag(s)                      # M1 x M1
        self.Yt = np.diag(v)                      # M2 x M2
        self.X = np.diag(s[self.ii])              # mi x mi
        self.Y = np.diag(v[self.jj])              # mj x mj

    # -- finite-difference operators -------------------------------------------
    def _build_operators(self):
        self._build_s_matrices()
        self._build_v_matrices()
        self._build_global_operators()

    def _build_s_matrices(self):
        """Difference matrices in the S-direction: Dst, Dsst, Dsmt."""
        m1, M1 = self.m1, self.M1
        s = self.s

        ds = s[1:m1 + 1] - s[0:m1]
        hh = ds[0:m1 - 1] + ds[1:m1]

        betsl = -ds[1:m1] / (ds[0:m1 - 1] * hh)
        betsp = ds[0:m1 - 1] / (ds[1:m1] * hh)
        betsm = -betsl - betsp

        delsl = 2.0 / (ds[0:m1 - 1] * hh)
        delsp = 2.0 / (ds[1:m1] * hh)
        delsm = -delsl - delsp

        betsl = np.concatenate([betsl, [0.0, 0.0]])
        betsm = np.concatenate([[0.0], betsm, [0.0]])
        betsp = np.concatenate([[0.0, 0.0], betsp])

        delsl = np.concatenate([delsl, [0.0, 0.0]])
        delsm = np.concatenate([[0.0], delsm, [0.0]])
        delsp = np.concatenate([[0.0, 0.0], delsp])

        # first derivative (central), upper boundary row zeroed
        Dst = self._spdiags_dense([betsl, betsm, betsp], [-1, 0, 1], M1, M1)
        Dst[M1 - 1, :] = 0.0

        # second derivative, one-sided closure at S = Smax
        Dsst = self._spdiags_dense([delsl, delsm, delsp], [-1, 0, 1], M1, M1)
        Dsst[M1 - 1, :] = 0.0
        Dsst[M1 - 1, M1 - 2] = 2.0 / (ds[m1 - 1] ** 2)
        Dsst[M1 - 1, M1 - 1] = -2.0 / (ds[m1 - 1] ** 2)

        # first derivative used by the mixed term
        Dsmt = self._spdiags_dense([betsl, betsm, betsp], [-1, 0, 1], M1, M1)
        Dsmt[M1 - 1, :] = 0.0

        self.Dst, self.Dsst, self.Dsmt = Dst, Dsst, Dsmt

    def _build_v_matrices(self):
        """Difference matrices in the v-direction: Dvt, Dvvt, Dvmt."""
        m2, M2 = self.m2, self.M2
        v = self.v
        nrep = self.nrep

        dv = v[1:m2 + 1] - v[0:m2]
        hh = dv[0:m2 - 1] + dv[1:m2]

        betvl = -dv[1:m2] / (dv[0:m2 - 1] * hh)
        betvp = dv[0:m2 - 1] / (dv[1:m2] * hh)
        betvm = -betvl - betvp

        alpvk = -betvl
        alpvl = -hh / (dv[0:m2 - 1] * dv[1:m2])
        alpvm = -alpvk - alpvl

        delvl = 2.0 / (dv[0:m2 - 1] * hh)
        delvp = 2.0 / (dv[1:m2] * hh)
        delvm = -delvl - delvp

        alpvk = np.concatenate([alpvk, [0.0, 0.0]])
        alpvl = np.concatenate([[0.0], alpvl, [0.0]])
        alpvm = np.concatenate([[0.0, 0.0], alpvm])

        betvl = np.concatenate([betvl, [0.0, 0.0]])
        betvm = np.concatenate([[0.0], betvm, [0.0]])
        betvp = np.concatenate([[0.0, 0.0], betvp])

        delvl = np.concatenate([delvl, [0.0, 0.0]])
        delvm = np.concatenate([[0.0], delvm, [0.0]])
        delvp = np.concatenate([[0.0, 0.0], delvp])

        # first derivative: central below theta, backward above; one-sided second order at v = 0
        Dvc = self._spdiags_dense([betvl, betvm, betvp], [-1, 0, 1], M2, M2)
        Dvb = self._spdiags_dense([alpvk, alpvl, alpvm], [-2, -1, 0], M2, M2)
        Dvt = Dvb.copy()
        Dvt[0:nrep, :] = Dvc[0:nrep, :]
        Dvt[M2 - 1, :] = 0.0
        Dvt[0, :] = 0.0
        Dvt[0, 0] = -(2 * dv[0] + dv[1]) / (dv[0] * (dv[0] + dv[1]))
        Dvt[0, 1] = (dv[0] + dv[1]) / (dv[0] * dv[1])
        Dvt[0, 2] = -dv[0] / (dv[1] * (dv[0] + dv[1]))

        # second derivative, one-sided closure at v = Vmax
        Dvvt = self._spdiags_dense([delvl, delvm, delvp], [-1, 0, 1], M2, M2)
        Dvvt[M2 - 1, :] = 0.0
        Dvvt[M2 - 1, M2 - 2] = 2.0 / (dv[m2 - 1] ** 2)
        Dvvt[M2 - 1, M2 - 1] = -2.0 / (dv[m2 - 1] ** 2)

        # first derivative used by the mixed term
        Dvmt = self._spdiags_dense([betvl, betvm, betvp], [-1, 0, 1], M2, M2)
        Dvmt[M2 - 1, :] = 0.0

        self.Dvt, self.Dvvt, self.Dvmt = Dvt, Dvvt, Dvmt
        self.DvmT_Yt = Dvmt.T @ self.Yt           # reused by _compute_g

    def _build_global_operators(self):
        """Kronecker-product operators A0 (mixed), A1 (S-direction), A2 (v-direction) and A."""
        M1 = self.M1

        Ds = self.Dst[1:M1, 1:M1]
        Dv = self.Dvt
        Dss = self.Dsst[1:M1, 1:M1]
        Dvv = self.Dvvt
        Dsm = self.Dsmt[1:M1, 1:M1]
        Dvm = self.Dvmt

        def csc(M):
            return sp.csc_matrix(M)

        mi, mj = self.mi, self.mj
        X, Y = self.X, self.Y
        Is = sp.identity(mi, format="csc")
        Iv = sp.identity(mj, format="csc")
        Ibig = sp.identity(mi * mj, format="csc")
        self.Ibig = Ibig

        A0_base = sp.kron(csc(Y @ Dvm), csc(X @ Dsm), format="csc")
        A1_diff = sp.kron(csc(Y), csc(0.5 * (X @ X @ Dss)), format="csc")
        A1_conv = sp.kron(Iv, csc(X @ Ds), format="csc")
        A2_diff_base = sp.kron(csc(Y @ Dvv), Is, format="csc")
        A2_dv_base = sp.kron(csc(Dv), Is, format="csc")
        A2_Ydv_base = sp.kron(csc(Y @ Dv), Is, format="csc")

        xi_v, rho_v = self.xi, self.rho
        rd_v, rf_v = self.rd, self.rf
        Theta_v = self.Theta_vec[0]
        kappa = self.kappa

        self.A0 = (rho_v * xi_v) * A0_base
        self.A1 = A1_diff + (rd_v - rf_v) * A1_conv - rd_v * Ibig
        self.A2 = (0.5 * xi_v ** 2) * A2_diff_base \
            + (kappa * Theta_v) * A2_dv_base - kappa * A2_Ydv_base
        self.A = (self.A0 + self.A1 + self.A2).tocsc()

    # -- payoff / obstacle -----------------------------------------------------
    def _build_payoff(self):
        K, s, M1, M2 = self.K, self.s, self.M1, self.M2
        # u = max(K - S, 0) on every v column, with cell averaging at the kink.
        u_mat = np.repeat(np.maximum(K - s, 0.0)[:, None], M2, axis=1)
        ind = np.where(s < K)[0][-1]
        if abs(s[ind] - K) > abs(s[ind + 1] - K):
            ind = ind + 1
        sl = (s[ind - 1] + s[ind]) / 2.0
        sr = (s[ind] + s[ind + 1]) / 2.0
        u_mat[ind, :] = 0.5 * (K - sl) ** 2 / (sr - sl)
        self.phi = u_mat[1:M1, 0:M2].flatten(order="F")

    # -- LU factorizations (time-constant -> factor once, reuse every step) -----
    def _factorize(self):
        th, dt, Ibig = self.theta_mcs, self.dt, self.Ibig
        self.lu1 = splu((Ibig - th * dt * self.A1).tocsc())

        # A2 = kron(B2, I_mi) is S-independent, so the A2-implicit stage decouples into
        # (I - th*dt*B2) X^T = R^T with one small (mj x mj) factorization and mi right-hand
        # sides -- identical to, and much faster than, factoring the full (mi*mj) system.
        B2 = 0.5 * self.xi ** 2 * (self.Yt @ self.Dvvt) \
            + self.kappa * self.theta * self.Dvt \
            - self.kappa * (self.Yt @ self.Dvt)
        self.lu2s = splu(sp.csc_matrix(np.eye(self.mj) - th * dt * B2))

        # Constant coefficients -> precompute the corrector and predictor operator combinations.
        self.Acorr = (th * dt * self.A0 + (0.5 - th) * dt * self.A).tocsc()
        self.A_dt = (dt * self.A).tocsc()

    def _solve2(self, b):
        """Solve (I - theta_mcs*dt*A2) x = b via the small v-direction factorization."""
        R = b.reshape(self.mi, self.mj, order="F")
        return self.lu2s.solve(np.ascontiguousarray(R.T)).T.ravel(order="F")

    # -- boundary contribution (only the S=0 row is non-zero; the A2 part vanishes) -
    def _boundary_val(self, i, amer):
        """Dirichlet value at S = 0 for time index i (1-based into tt)."""
        if amer:
            return self.K
        tau = self.T2 - self.tt[i - 1]
        return self.K * np.exp(-self.rd * tau)

    def _compute_g(self, i, amer):
        """Return (g0, g1, g) at time index i (mixed and S-direction boundary contributions)."""
        M1, M2 = self.M1, self.M2
        Gd = np.zeros((M1, M2))
        Gd[0, :] = self._boundary_val(i, amer)
        g0 = (self.rho * self.xi) * (self.Xt @ (self.Dsmt @ (Gd @ self.DvmT_Yt)))
        g1 = 0.5 * (self.Xt @ self.Xt @ (self.Dsst @ Gd) @ self.Yt) \
            + (self.rd - self.rf) * (self.Xt @ (self.Dst @ Gd))
        g0i = g0[1:M1, 0:M2].flatten(order="F")
        g1i = g1[1:M1, 0:M2].flatten(order="F")
        return g0i, g1i, g0i + g1i

    # -- MCS-ADI single-step building blocks -----------------------------------
    def _predictor(self, u, g_old, g_new):
        dt, th = self.dt, self.theta_mcs
        _g0_o, g1_o, _g_o = g_old
        _g0_n, g1_n, g_n = g_new
        z0 = self.A_dt.dot(u) + dt * g_n
        z = self.lu1.solve(z0 + th * dt * (g1_n - g1_o))
        return z0, self._solve2(z)

    def _step_mcs(self, u, g_old, g_new, amer):
        """One MCS-ADI step; for American options with policy=0 the result is projected
        onto the payoff (Bermudan treatment)."""
        dt, th = self.dt, self.theta_mcs
        g0_o, g1_o, g_o = g_old
        g0_n, g1_n, g_n = g_new

        z0, z = self._predictor(u, g_old, g_new)
        z = z0 + self.Acorr.dot(z) \
            + th * dt * (g0_n - g0_o) + (0.5 - th) * dt * (g_n - g_o)
        z = self.lu1.solve(z + th * dt * (g1_n - g1_o))
        z = self._solve2(z)
        u = u + z
        if amer:
            u = np.maximum(u, self.phi)
        return u

    def _step_penalized(self, u, Pi_AM, g_old, g_new):
        """One MCS-ADI step with penalized policy iteration (policy=1).  Returns (u, Pi_AM)."""
        dt, th = self.dt, self.theta_mcs
        g0_o, g1_o, g_o = g_old
        g0_n, g1_n, g_n = g_new

        z0, z = self._predictor(u, g_old, g_new)
        z = z0 + self.Acorr.dot(z) \
            + th * dt * (g0_n - g0_o) + (0.5 - th) * dt * (g_n - g_o)
        Yhat1 = self.lu1.solve(z + th * dt * (g1_n - g1_o))
        return self._policy_iteration(u, Yhat1, Pi_AM)

    def _policy_iteration(self, u, Yhat1, Pi_AM_iter):
        """Penalized policy iteration for the A2-implicit stage until the active set stabilizes."""
        dt, th = self.dt, self.theta_mcs
        Ibig, A2, phi = self.Ibig, self.A2, self.phi
        rho_AM = self.rho_AM
        n_dof = self.mi * self.mj

        rhs_const = (Ibig - th * dt * A2).dot(u)
        U_iter_km1 = u
        U_iter = u
        for _k in range(1, self.pp_maxiter + 1):
            Pk1 = sp.diags(Pi_AM_iter, 0, format="csc")
            M_pen = (Ibig - th * dt * A2 + rho_AM * Pk1).tocsc()
            rhs = Yhat1 + rhs_const + rho_AM * (Pk1.dot(phi))
            U_iter = splu(M_pen).solve(rhs)
            Pi_AM_new = np.zeros(n_dof)
            Pi_AM_new[(phi - U_iter) > 0] = 1
            crit = np.max(np.abs(U_iter - U_iter_km1) / np.maximum(1.0, np.abs(U_iter)))
            if crit < self.pp_tol or np.all(Pi_AM_new == Pi_AM_iter):
                break
            Pi_AM_iter = Pi_AM_new
            U_iter_km1 = U_iter
        return U_iter, Pi_AM_iter

    # -- MCS-ADI time integration ----------------------------------------------
    def solve(self, amer):
        """Run the MCS-ADI time stepping and return the full (M1 x M2) value grid on (s, v).
        Results are cached per option style."""
        amer = int(bool(amer))
        if amer in self._U_cache:
            return self._U_cache[amer]

        u = self.phi.copy()
        use_policy = bool(amer and self.policy)
        if use_policy:
            Pi_AM = np.zeros(self.mi * self.mj)
            Pi_AM[(self.phi - u) > 0] = 1

        # For American options the S=0 boundary value is the constant K, so the boundary
        # vectors are time-independent and their increments vanish.
        g_old = self._compute_g(1, amer)
        for n in range(1, self.N + 1):
            g_new = g_old if amer else self._compute_g(n + 1, amer)
            if use_policy:
                u, Pi_AM = self._step_penalized(u, Pi_AM, g_old, g_new)
            else:
                u = self._step_mcs(u, g_old, g_new, amer)
            g_old = g_new

        U = np.zeros((self.M1, self.M2))
        U[0, :] = self.K
        U[1:self.M1, :] = u.reshape(self.mi, self.mj, order="F")
        self._U_cache[amer] = U
        return U

    def price(self, S0, V0, amer):
        """Price at (S0, V0); amer = 0 (European) or 1 (American).  The PDE is solved at most
        once per style; the value is read off by a separable not-a-knot cubic spline."""
        U = self.solve(amer)
        cs_s = CubicSpline(self.s, U, axis=0, bc_type="not-a-knot")
        tmp = cs_s(float(S0))
        cs_v = CubicSpline(self.v, tmp, bc_type="not-a-knot")
        return float(cs_v(float(V0)))


def price_put_fd(S0, K, T, rd, rf, theta, xi, rho, V0, kappa, amer=1, **kwargs):
    """Convenience one-shot pricer: build a solver for these parameters and return one price."""
    solver = HestonFDSolver({"K": K, "T2": T, "rd": rd, "rf": rf,
                             "theta": theta, "xi": xi, "rho": rho, "kappa": kappa, **kwargs})
    return solver.price(S0, V0, amer)
