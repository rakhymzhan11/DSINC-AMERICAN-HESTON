# DSINC-AMERICAN-HESTON
Two independent Python pricers for American/European puts under the Heston model: a spectral damped-sinc (DSINC) integral-equation method with a piecewise-linear exercise boundary, and an MCS-ADI finite-difference benchmark with policy iteration. Cross-validated against standard benchmarks. 



# American-put pricing under the Heston model

Two independent, self-contained Python pricers for **European and American put
options** under the Heston stochastic-volatility model (constant coefficients).
The two methods rest on entirely different principles, a semi-analytical spectral
integral equation and a finite-difference PDE solver, so running both is a strong
mutual cross-check, and each can serve as a reference for the other.

This code accompanies a research paper on valuing American options and flexible-forward
(window-forward) FX contracts; a flexible forward is, in essence, an American-style
option on the timing of delivery, and is priced with exactly these techniques.

---

## The two methods

### Spectral pricer — `DSINCPut` (`dsinc_pricer.py`)

A local-basis spectral method. The early-exercise premium is written as a Fourier
integral of an affine *tilted-CIR* transform, summed over a **piecewise-linear (M>1)**
representation of the early-exercise boundary `x*(v) = a_m + b_m v` on the variance
grid. Unlike a global cosine (COS) basis — whose coefficients all react to the kink at
the exercise boundary and so suffer Gibbs oscillations — the **damped-sinc** basis is
locally supported, so a move of the boundary perturbs only nearby coefficients. The
method needs no conditional-CF inversion and no quadrature in the variance direction;
the exercise boundary is solved explicitly and separately from the option value.

### Finite-difference pricer — `HestonFDSolver` (`fd_adi_pricer.py`)

A PDE benchmark on a non-uniform `(S, v)` grid, refined around the strike and near
`v = 0`, integrated in time with the **Modified Craig–Sneyd (MCS) ADI** scheme. The
free-boundary (linear-complementarity) problem of American exercise is handled by a
**penalized policy iteration** (default), or, optionally, by simple Bermudan
projection onto the payoff after each step. All operators and the implicit-stage
factorizations depend only on the model parameters, so they are built once and reused
across the European and the American solve.

---

## Installation

Only NumPy and SciPy are required:

```bash
pip install numpy scipy
```

Then copy the three modules (`dsinc_pricer.py`, `fd_adi_pricer.py`, `benchmark.py`)
into your project, or clone the repository and run from its directory.

---

## Quick start

```python
# --- spectral pricer: returns (European, American, early-exercise premium) ---
from dsinc_pricer import DSINCPut

PE, PA, EEP = DSINCPut({
    "S": 100, "K": 100, "T": 0.25,
    "rd": 0.03, "rf": 0.025,          # domestic rate, dividend / foreign rate
    "theta": 0.6,                     # long-run variance
    "sigma": 0.8,                     # vol-of-vol
    "rho": -0.4, "v0": 0.5, "kappa": 0.5,
}).price()

# --- finite-difference pricer: one price per call, setup reused across calls ---
from fd_adi_pricer import HestonFDSolver

fd = HestonFDSolver({
    "K": 100, "T2": 0.25, "rd": 0.03, "rf": 0.025,
    "theta": 0.6, "xi": 0.8, "rho": -0.4, "kappa": 2.0,
})
european = fd.price(S0=100, V0=0.5, amer=0)
american = fd.price(S0=100, V0=0.5, amer=1)   # reuses grids / operators / factorizations
```

A one-shot helper is also provided:

```python
from fd_adi_pricer import price_put_fd
american = price_put_fd(S0=100, K=100, T=0.25, rd=0.03, rf=0.025,
                        theta=0.6, xi=0.8, rho=-0.4, V0=0.5, kappa=2.0, amer=1)
```

Run the bundled validation:

```bash
python3 benchmark.py
```

---

## Parameters

Both classes expose every setting as a constructor parameter with a sensible default;
override any subset through the dict. An unknown key raises `KeyError` to catch typos.

**Model** (shared): `K` strike, `T` / `T2` maturity, `rd` domestic rate, `rf`
dividend/foreign rate, `theta` long-run variance, `sigma` / `xi` vol-of-vol, `rho`
spot–variance correlation, `v0` / `V0` initial variance, `kappa` mean-reversion speed.

**`DSINCPut` numerics:** `Nv` variance nodes (`M = Nv-1` boundary segments, default 16),
`Nt` induction time steps (default `max(16, round(14+6T))`), `Nom` frequency nodes
(default 600), `om_max` frequency half-range (default 180), `mode` (`'partial'` =
band-restricted segment transform, the consistent choice for `M > 1`).

**`HestonFDSolver` numerics:** `N` time steps (default 100), `m1` spot intervals
(default 200; the variance grid uses `m1 // 2`), `Smax_mult`, `Vmax`, `theta_mcs`
(MCS parameter), `policy` (`1` = penalized policy iteration, default; `0` = Bermudan
projection), and the penalty constants `rho_AM`, `pp_maxiter`.

---

## Validation

`benchmark.py` reproduces two standard tests; representative output (defaults):

**Paper test case** — `S=100, K=100, T=0.25, rd=0.03, rf=0.025, theta=0.6,
vol-of-vol=0.8, rho=-0.4, v0=0.5, kappa=0.5`:

| | European | American |
|---|---|---|
| spectral | 13.6633 | 13.6854 |
| finite difference | 13.6622 | 13.6839 |
| reference (integral equation) | 13.6633 | 13.6749 |
| reference (finite difference) | 13.6611 | 13.6837 |

**American-put benchmark** (Haentjens & in 't Hout, 2015; reference prices from
Ikonen & Toivanen, 2009) — `kappa=5, theta=0.16, vol-of-vol=0.9, rho=0.1, r=0.10,
K=10, T=0.25`:

| v0 | RMS error, spectral | RMS error, finite difference |
|----|---------------------|------------------------------|
| 0.0625 | 0.0003 | 0.0002 |
| 0.25   | 0.0005 | 0.0004 |

Both methods reproduce the published prices to a few parts in `1e3`, and they agree
with each other to the same order — a genuine cross-validation, since they share no
code or numerical machinery.

---

## Performance notes

- The spectral pricer takes roughly **0.8 s** per price at the default resolution
  (`Nv=16, Nom=600`); cost grows about `Nt²` with maturity (`Nt = max(16, 14+6T)`).
- The finite-difference pricer with **penalized policy iteration** (`policy=1`, the default) is
  more faithful to the linear-complementarity formulation and it takes around **5 s** per price,
  because the penalized stage is refactorized as the active set
  updates.




