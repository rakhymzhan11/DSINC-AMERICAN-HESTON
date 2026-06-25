# American-option pricing under the Heston model

Two independent, self-contained Python pricers for **European and American options**
under the Heston stochastic-volatility model (constant coefficients). The two methods
rest on entirely different principles — a semi-analytical integral-equation method
(DSINC) and a finite-difference PDE solver — so running both is a strong mutual
cross-check, and either can serve as a reference for the other.

The examples and the implementation are written for **put** options; the same method
also prices **call** options (e.g. through put–call duality).

This code accompanies the paper:

> L. Andersen, A. Itkin and R. Kazbek,
> *Valuing American options and Flexible Forwards contracts in time-dependent models.*
> SSRN:https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6991498 · arXiv: *to be added*.

A flexible forward (window forward) is, in essence, an American-style option on the
timing of delivery, and is priced with exactly these techniques.

---

## The two methods

### DSINC pricer — `DSINCPut` (`dsinc_pricer.py`)

A local-basis integral-equation method that uses **damped-sinc (DSINC)** basis
functions. The early-exercise boundary `x*(v) = a_m + b_m v` is represented as a
piecewise-linear function over `M = Nv − 1` variance segments and solved **explicitly
and separately** from the option value; the early-exercise premium is then evaluated
in the DSINC basis from an affine tilted-CIR transform. No conditional
characteristic-function inversion is required.

The DSINC basis is a local (compactly supported) alternative to the global cosine
(COS) basis. The COS method assumes the density is effectively zero outside a
truncation interval `[a, b]`; when the Feller condition is violated, a significant part
of the variance density sits near `v = 0`, and the truncation removes a vital part of
the distribution. This causes slow convergence — one needs an exponentially larger
number of cosine terms for even basic precision — and can mis-price options,
particularly deep in-the-money or short-maturity ones. The variance density is
therefore handled here by numerical integration of the inverse CDF to a prescribed
tolerance.

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

Then copy the modules into your project, or clone the repository and run from its
directory.

---

## Quick start

```python
# --- DSINC pricer: returns (European, American, early-exercise premium) ---
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

**`DSINCPut` numerics:** `Nv` variance nodes (`M = Nv − 1` boundary segments, default
16), `Nt` induction time steps (default `max(16, round(14 + 6T))`), `Nom` frequency
nodes (default 600), `om_max` frequency half-range (default 180), `mode` (`'partial'` =
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
| DSINC | 13.6633 | 13.6854 |
| finite difference | 13.6622 | 13.6839 |
| reference (integral equation) | 13.6633 | 13.6749 |
| reference (finite difference) | 13.6611 | 13.6837 |

**American-put benchmark** (Haentjens & in 't Hout, 2015; reference prices from
Ikonen & Toivanen, 2009) — `kappa=5, theta=0.16, vol-of-vol=0.9, rho=0.1, r=0.10,
K=10, T=0.25`:

| v0 | RMS error, DSINC | RMS error, finite difference |
|----|------------------|------------------------------|
| 0.0625 | 0.0003 | 0.0002 |
| 0.25   | 0.0005 | 0.0004 |

Both methods reproduce the published prices to a few parts in `1e3`, and they agree
with each other to the same order — a genuine cross-validation, since they share no
code or numerical machinery.

---

## Performance notes

- The DSINC pricer takes roughly **0.8 s** per price at the default resolution
  (`Nv=16, Nom=600`); cost grows about `Nt²` with maturity (`Nt = max(16, 14 + 6T)`).
- The finite-difference pricer with **Bermudan projection** (`policy=0`) is very fast
  (about 0.1–0.3 s). The **penalized policy iteration** (`policy=1`, the default) is
  more faithful to the linear-complementarity formulation but several times slower
  (a few seconds), because the penalized stage is refactorized as the active set
  updates. The two American treatments converge to the same price; switch with
  `{"policy": 0}` when speed matters.


---

## License

MIT — see [`LICENSE`](LICENSE).
