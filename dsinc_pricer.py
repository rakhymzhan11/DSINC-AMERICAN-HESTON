"""
dsinc_pricer.py -- spectral (damped-sinc) American-put pricer under the Heston model.

Prices European and American puts under the Heston stochastic-volatility model with
constant coefficients.  The early-exercise premium is obtained by Fourier inversion
of an affine tilted-CIR transform summed over a piecewise-linear (M>1) representation
of the exercise boundary x*(v) = a_m + b_m v on the variance grid; no conditional
characteristic-function inversion and no variance-direction quadrature are required.

The class holds the contract/model parameters as attributes (defaults below), which
can be overridden through a dict passed to the constructor (unknown keys raise
KeyError).  All query-independent work -- grids, the per-lag transform pieces and the
CIR transition band masses -- is done once in the constructor; price() runs the
backward induction over the exercise boundary and returns (PE, PA, EEP).

    from dsinc_pricer import DSINCPut
    PE, PA, EEP = DSINCPut({"S": 100, "K": 95, "T": 0.25, "rd": 0.025, "rf": 0.05,
                            "theta": 0.05, "sigma": 0.9, "rho": 0.0,
                            "v0": 0.05, "kappa": 2.0}).price()
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.stats import ncx2


class DSINCPut:
    """Spectral DSINC (M>1) American-put solver under the constant-coefficient Heston model."""

    # Contract / model parameters
    S:     float = 100.0
    K:     float = 100.0
    T:     float = 0.25
    rd:    float = 0.03      # domestic / risk-free rate
    rf:    float = 0.025     # foreign rate / dividend yield (0 = none)
    theta: float = 0.6       # long-run variance
    sigma: float = 0.8       # vol-of-vol
    rho:   float = -0.4      # spot/variance correlation
    v0:    float = 0.5       # initial variance
    kappa: float = 0.5       # mean-reversion speed

    # Numerical parameters
    Nv:     int   = 16        # variance nodes (M = Nv-1 boundary segments)
    Nt:     object = None     # induction time steps (None -> max(16, round(14+6T)))
    Nom:    int   = 600       # Gauss-Legendre frequency nodes
    om_max: float = 180.0     # frequency half-range [-om_max, om_max]
    mode:   str   = "partial"  # 'partial' = band-restricted segment transform (consistent for M>1)

    def __init__(self, params=None):
        if params is not None:
            allowed = {k for klass in type(self).__mro__
                       for k in getattr(klass, "__annotations__", {})}
            for key, value in params.items():
                if key not in allowed:
                    raise KeyError(f"unknown parameter '{key}'")
                setattr(self, key, value)

        self.chi = -1.0
        if self.Nt is None:
            self.Nt = max(16, round(14 + 6 * self.T))
        self.M = self.Nv - 1
        self.x0 = np.log(self.S / self.K)
        self.partial = (str(self.mode).lower() == "partial")

        s = self.sigma
        self.mu1 = self.rho / s
        self.lam1 = self.kappa * self.rho / s - 0.5
        self.lam2 = 1 - self.rho ** 2
        self.R = 0.5 * s ** 2
        self.P = -self.kappa
        self.cofA = -(2 * self.kappa * self.theta / s ** 2)
        self.rtx = self.rho * self.theta / s

        self.PE = self.PA = self.EEP = None
        self.xstar = self.vgrid = self.tt = None

        self._build_grids()
        self._precompute_lags()

    # -- grids and frequency-domain constants ----------------------------------
    def _build_grids(self):
        Nt, Nv = self.Nt, self.Nv
        self.tt = self.T * np.linspace(0.0, 1.0, Nt)
        kappa, theta, s, v0 = self.kappa, self.theta, self.sigma, self.v0
        ekT = np.exp(-kappa * self.T)
        df = 4 * kappa * theta / s ** 2
        sc = 4 * kappa / (s ** 2 * (1 - ekT))
        ncp = sc * ekT * v0
        bv = ncx2.ppf(0.9999, df, ncp) / sc
        bv = max(bv, theta + 6 * np.sqrt(theta * s ** 2 / (2 * kappa)) + v0)
        self.vgrid = bv * (np.linspace(0.0, 1.0, Nv) ** 1.6)   # nodes clustered near v=0
        self.vgrid[0] = 1e-6

        x, w = leggauss(self.Nom)
        a, b = -self.om_max, self.om_max
        self.om = 0.5 * (b - a) * x + 0.5 * (b + a)
        self.wq = 0.5 * (b - a) * w
        self.iw = 1j * self.om - self.chi
        self.wc = -self.om - 1j * self.chi
        gam = 0.5 * self.wc ** 2 * self.lam2 - 1j * self.wc * self.lam1
        self.Q = -gam
        self.dd = np.sqrt(self.P ** 2 - 4.0 * self.R * self.Q)
        self.Dcd = 1j * self.wc * self.mu1

    def _precompute_lags(self):
        # On the uniform time grid the transition lag tau = lag*Dt fully determines the
        # transform pieces and the band masses, so they are built once per lag and reused.
        Nt, Nv, v0 = self.Nt, self.Nv, self.v0
        Dt = self.tt[1] - self.tt[0]
        self.Dt = Dt
        self._glag = [None] * Nt
        self._WM = [None] * Nt                       # (Nv, M): conditioning on the variance grid
        self._WM0 = [None] * Nt                       # (M,):    conditioning on v0
        vc_all = np.concatenate([self.vgrid, [v0]])
        for L in range(1, Nt):
            tau = L * Dt
            self._glag[L] = self._tilt_tau_parts(tau)
            if self.partial:
                wm_all = self._bandmass(vc_all, tau)
                self._WM[L] = wm_all[:Nv, :]
                self._WM0[L] = wm_all[Nv, :]

    def _tilt_tau_parts(self, tau):
        """Closed-form 2x2 propagator pieces g11..g22 and logF for a lag tau (vectors over omega)."""
        P, R, Q, dd = self.P, self.R, self.Q, self.dd
        z = dd * tau / 2.0
        em2 = np.exp(-2 * z); c0 = 1 + em2; s0 = 1 - em2
        g11 = c0 - (P / dd) * s0
        g12 = (-2 * R / dd) * s0
        g21 = (2 * Q / dd) * s0
        g22 = c0 + (P / dd) * s0
        logF = P * tau / 2.0 + z - np.log(2.0)
        return g11, g12, g21, g22, logF

    # -- per-segment affine transform coefficients -----------------------------
    def _segcoef(self, xs, L):
        """Affine coefficients (E,F) and prefactors (e1,e2), shape (Nom, M), for boundary xs at lag L."""
        vg = self.vgrid
        bm = (xs[1:] - xs[:-1]) / (vg[1:] - vg[:-1])
        am = xs[:-1] - bm * vg[:-1]
        g11, g12, g21, g22, logF = self._glag[L]
        iw = self.iw[:, None]; Dcd = self.Dcd[:, None]
        g11 = g11[:, None]; g12 = g12[:, None]
        g21 = g21[:, None]; g22 = g22[:, None]; logF = logF[:, None]
        b = bm[None, :]; a = am[None, :]
        L1 = Dcd + iw * b
        Y1 = g11 + g12 * L1
        F1 = (g21 + g22 * L1) / Y1
        E1 = self.cofA * (logF + self._clog(Y1))
        L2 = Dcd + (1 + iw) * b
        Y2 = g11 + g12 * L2
        F2 = (g21 + g22 * L2) / Y2
        E2 = self.cofA * (logF + self._clog(Y2))
        e1 = self.rd * np.exp(iw * a) / iw
        e2 = self.rf * np.exp((1 + iw) * a) / (1 + iw)
        return E1, F1, E2, F2, e1, e2

    @staticmethod
    def _clog(X):
        """Complex log with the phase unwrapped along the frequency axis."""
        return np.log(np.abs(X)) + 1j * np.unwrap(np.angle(X), axis=0)

    def _bandmass(self, vconds, tau):
        """CIR transition mass P(v_u in [v_m, v_{m+1}] | v_cond) over the segments (rows = v_cond)."""
        kappa, theta, s = self.kappa, self.theta, self.sigma
        vconds = np.atleast_1d(vconds)
        ek = np.exp(-kappa * tau)
        c = s ** 2 * (1 - ek) / (4 * kappa)
        df = 4 * kappa * theta / s ** 2
        lam = 4 * kappa * vconds * ek / (s ** 2 * (1 - ek))
        cdf = ncx2.cdf((self.vgrid / c)[None, :], df, lam[:, None])
        return np.diff(cdf, axis=1)

    # -- European leg: closed-form marginal characteristic function + COS payoff -
    def _heston_cf(self, u, tau, v):
        """Marginal Heston characteristic function in x=log(S_T/K) at maturity tau, variance v."""
        kappa, theta, s, rho = self.kappa, self.theta, self.sigma, self.rho
        a = kappa * theta; b = kappa; i = 1j
        bb = rho * s * i * u - b; s2 = s ** 2
        d = np.sqrt(bb ** 2 + s2 * (i * u + u ** 2))
        lam2 = -bb - d; lam1 = -bb + d
        g = lam2 / lam1
        bad = np.abs(lam1) < 1e-12
        if np.any(bad):
            g = g.copy()
            g[bad] = 1 - 2 * d[bad] / (lam1[bad] + 1e-12)
        edt = np.exp(-d * tau)
        C = (a / s2) * (lam2 * tau - 2 * np.log((1 - g * edt) / (1 - g)))
        D = (1 - edt) / (1 - g * edt) * (b - rho * s * i * u - d) / s2
        cf = np.exp(i * u * ((self.rd - self.rf) * tau) + C + D * v)
        cf[~np.isfinite(cf)] = 0.0
        return cf

    def _european_coeffs(self, v, tau):
        """Precompute the COS European-put coefficients for fixed (v, tau); the log-moneyness x stays free."""
        if tau <= 1e-12:
            return None
        rd, theta, s = self.rd, self.theta, self.sigma
        c1 = (rd - self.rf) * tau - 0.5 * v * tau
        w = 14 * np.sqrt(v * tau + theta * tau + s ** 2 * tau ** 2 / 8 + 1e-4)
        a, b = c1 - w, c1 + w
        N = 320
        k = np.arange(N)
        freqs = k * np.pi / (b - a)
        cf = self._heston_cf(freqs, tau, v)
        U = 2.0 / (b - a) * (-self._xi(k, a, b, a, 0.0) + self._psi1(k, a, b, a, 0.0))
        unit = np.concatenate([[0.5], np.ones(N - 1)])
        DF = self.K * np.exp(-rd * tau)
        return a, freqs, cf * unit * U, DF

    def _european_value(self, x, coeffs):
        """European put at log-moneyness x from precomputed coefficients (intrinsic if tau~0)."""
        if coeffs is None:
            return max(self.K * (1 - np.exp(x)), 0.0)
        a, freqs, cfU, DF = coeffs
        val = DF * np.sum(np.real(cfU * np.exp(1j * freqs * (x - a))))
        return max(val, 0.0)

    @staticmethod
    def _xi(k, a, b, c, d):
        kk = k * np.pi / (b - a)
        return 1.0 / (1.0 + kk ** 2) * (
            np.cos(kk * (d - a)) * np.exp(d) - np.cos(kk * (c - a)) * np.exp(c)
            + kk * (np.sin(kk * (d - a)) * np.exp(d) - np.sin(kk * (c - a)) * np.exp(c)))

    @staticmethod
    def _psi1(k, a, b, c, d):
        out = np.empty(k.shape, dtype=float)
        out[0] = d - c
        kk = k[1:]
        out[1:] = (np.sin(kk * np.pi * (d - a) / (b - a))
                   - np.sin(kk * np.pi * (c - a) / (b - a))) * (b - a) / (kk * np.pi)
        return out

    # -- time weights and boundary root-find -----------------------------------
    @staticmethod
    def _trapw(x):
        x = np.atleast_1d(x).astype(float)
        n = x.size
        w = np.zeros(n)
        if n == 1:
            return w
        w[0] = (x[1] - x[0]) / 2
        w[-1] = (x[-1] - x[-2]) / 2
        if n > 2:
            w[1:-1] = (x[2:] - x[:-2]) / 2
        return w

    @staticmethod
    def _safe(f, x):
        try:
            v = f(x)
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    def _solve_boundary(self, res, x0):
        # Bracketing root-find warm-started at x0: expand a two-sided bracket about x0 (sqrt(2)
        # growth) and bisect the first sign change; a wide scan over [log(1e-4), 0.02] is the
        # fallback.  Confined to the put-sensible range x* <= ~0.05.
        lo_lim, hi_lim = np.log(1e-6), 0.05
        x0 = float(np.clip(x0, lo_lim, hi_lim))
        f0 = self._safe(res, x0)
        if np.isfinite(f0) and f0 == 0:
            return x0
        if np.isfinite(f0):
            dx = (abs(x0) / 50.0) if x0 != 0 else (1.0 / 50.0)
            a = b = x0
            fb = f0
            for _ in range(100):
                dx *= np.sqrt(2.0)
                a = x0 - dx
                fa = self._safe(res, a)
                if np.isfinite(fa) and (fa > 0) != (fb > 0):
                    return brentq(res, a, b, xtol=1e-12)
                b = x0 + dx
                fb = self._safe(res, b)
                if np.isfinite(fa) and np.isfinite(fb) and (fa > 0) != (fb > 0):
                    return brentq(res, a, b, xtol=1e-12)
                if a < lo_lim and b > hi_lim:
                    break
        xs = np.linspace(np.log(1e-4), 0.02, 60)
        rv = np.array([self._safe(res, xi) for xi in xs])
        ok = np.isfinite(rv)
        idx = np.where(ok[:-1] & ok[1:] & (rv[:-1] * rv[1:] < 0))[0]
        if idx.size:
            s = idx[0]
            return brentq(res, xs[s], xs[s + 1], xtol=1e-12)
        rv = np.where(ok, np.abs(rv), np.inf)
        return float(xs[int(np.argmin(rv))])

    def _assemble_acc(self, E1, F1, E2, F2, e1, e2, wm):
        """Sum the segment transforms over m at every variance node: acc has shape (Nom, Nv)."""
        Nom, M, Nv = self.Nom, self.M, self.Nv
        vcr = self.vgrid.reshape(1, 1, Nv)
        P1 = e1[:, :, None] * np.exp(E1[:, :, None] + F1[:, :, None] * vcr)
        P2 = e2[:, :, None] * np.exp(E2[:, :, None] + F2[:, :, None] * vcr)
        if self.partial:
            Wr = wm.T.reshape(1, M, Nv)
        else:
            Wr = np.ones((1, M, 1))
        return np.sum(Wr * (P1 - P2), axis=1)

    # -- backward induction over the exercise boundary, then the premium --------
    def _backward_solve(self):
        Nt, Nv, Nom, K = self.Nt, self.Nv, self.Nom, self.K
        wc, wq, tt, vgrid = self.wc, self.wq, self.tt, self.vgrid
        rd, rf, mu1, kappa, rtx = self.rd, self.rf, self.mu1, self.kappa, self.rtx

        xT = np.log(min(1.0, rd / rf)) if rf > 0 else 0.0
        xstar = np.full((Nt, Nv), xT, dtype=float)
        mu0r_base = -mu1 * vgrid

        for k in range(Nt - 2, -1, -1):
            t = tt[k]
            iul = np.arange(k + 1, Nt)
            Wt = self._trapw(np.concatenate([[t], tt[iul]]))
            twf = Wt[1:]
            B = np.zeros((Nom, Nv), dtype=complex)
            for jj, iu in enumerate(iul):
                L = iu - k
                tau = tt[iu] - t
                Du = np.exp(-rd * tau)
                E1, F1, E2, F2, e1, e2 = self._segcoef(xstar[iu, :], L)
                wm = self._WM[L] if self.partial else None
                acc = self._assemble_acc(E1, F1, E2, F2, e1, e2, wm)
                mu0r = mu0r_base + (rd - rf) * tau - kappa * rtx * tau
                ph = np.exp(1j * wc[:, None] * mu0r[None, :])
                B += (Du * twf[jj]) * wq[:, None] * (ph * acc)

            tau_eu = self.T - t
            for iv in range(Nv):
                Bcol = B[:, iv]
                coeffs = self._european_coeffs(vgrid[iv], tau_eu)

                def res(xs, Bcol=Bcol, coeffs=coeffs):
                    eepf = K / (2 * np.pi) * np.real(np.sum(Bcol * np.exp(1j * wc * xs)))
                    return K * (1 - np.exp(xs)) - self._european_value(xs, coeffs) - eepf

                xstar[k, iv] = self._solve_boundary(res, xstar[k + 1, iv])

        self.xstar = xstar
        return self._final_eep(xstar)

    def _final_eep(self, xstar):
        Nt, K, wc, wq = self.Nt, self.K, self.wc, self.wq
        tt, v0, x0 = self.tt, self.v0, self.x0
        rd, rf, mu1, kappa, rtx = self.rd, self.rf, self.mu1, self.kappa, self.rtx
        Wt = self._trapw(tt)
        EEP = 0.0
        for iu in range(1, Nt):
            tau = tt[iu]
            L = iu
            Du = np.exp(-rd * tau)
            E1, F1, E2, F2, e1, e2 = self._segcoef(xstar[iu, :], L)
            wm0 = self._WM0[L] if self.partial else np.ones(self.M)
            acc = np.sum(wm0 * (e1 * np.exp(E1 + F1 * v0) - e2 * np.exp(E2 + F2 * v0)), axis=1)
            mu0 = x0 + (rd - rf) * tau - mu1 * v0 - kappa * rtx * tau
            term = np.exp(1j * wc * mu0) * acc
            EEP += Du * Wt[iu] * (1.0 / (2 * np.pi)) * np.real(np.sum(wq * term))
        return max(K * EEP, 0.0)

    def price(self):
        """Return (PE, PA, EEP): European put, American put, early-exercise premium."""
        self.EEP = self._backward_solve()
        self.PE = self._european_value(self.x0, self._european_coeffs(self.v0, self.T))
        self.PA = max(self.PE + self.EEP, self.K - self.S)
        return self.PE, self.PA, self.EEP
