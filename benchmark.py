"""
benchmark.py -- validate both pricers on two standard test sets.

  1. The paper's constant-coefficient test case, against its published
     integral-equation (IE) and finite-difference (FD) reference values.
  2. The Haentjens & in 't Hout (2015) American-put benchmark set, against the
     Ikonen & Toivanen (2009) reference prices.

Each case is priced with the DSINC method and the finite-difference (MCS-ADI)
method, so the two independent solvers also cross-check each other.

    python3 benchmark.py
"""
import time
import numpy as np

from dsinc_pricer import DSINCPut
from fd_adi_pricer import HestonFDSolver


def price_both(S, K, T, rd, rf, theta, vov, rho, v0, kappa):
    """Return (DSINC PE, PA, EEP), (FD PE, PA), and the two timings."""
    t = time.time()
    sPE, sPA, sEEP = DSINCPut({"S": S, "K": K, "T": T, "rd": rd, "rf": rf, "theta": theta,
                               "sigma": vov, "rho": rho, "v0": v0, "kappa": kappa}).price()
    st = time.time() - t
    t = time.time()
    fd = HestonFDSolver({"K": K, "T2": T, "rd": rd, "rf": rf, "theta": theta,
                         "xi": vov, "rho": rho, "kappa": kappa})
    fPE, fPA = fd.price(S, v0, amer=0), fd.price(S, v0, amer=1)
    ft = time.time() - t
    return (sPE, sPA, sEEP), (fPE, fPA), (st, ft)


# ---------------------------------------------------------------------------
# 1. Paper test case (constant coefficients)
# ---------------------------------------------------------------------------
print("\n== Paper test case (constant coefficients) ==")
print("   S=100, K=100, T=0.25, r_d=0.03, r_f=0.025, theta=0.6, vol-of-vol=0.8,")
print("   rho=-0.4, v0=0.5, kappa=0.5\n")
(sPE, sPA, sEEP), (fPE, fPA), (st, ft) = price_both(
    100, 100, 0.25, 0.03, 0.025, 0.6, 0.8, -0.4, 0.5, 0.5)
print(f"   {'':10}{'European':>11}{'American':>11}{'EEP':>10}{'sec':>8}")
print(f"   {'DSINC':10}{sPE:>11.4f}{sPA:>11.4f}{sEEP:>10.4f}{st:>8.2f}")
print(f"   {'FD-ADI':10}{fPE:>11.4f}{fPA:>11.4f}{fPA-fPE:>10.4f}{ft:>8.2f}")
print(f"   {'ref (IE)':10}{13.6633:>11.4f}{13.6749:>11.4f}")
print(f"   {'ref (FD)':10}{13.6611:>11.4f}{13.6837:>11.4f}")

# ---------------------------------------------------------------------------
# 2. Haentjens & in 't Hout (2015) benchmark set
# ---------------------------------------------------------------------------
print("\n== Haentjens & in 't Hout (2015) benchmark ==")
print("   kappa=5, theta=0.16, vol-of-vol=0.9, rho=0.1, r=0.10, K=10, T=0.25\n")
K, T, r, kappa, theta, vov, rho = 10, 0.25, 0.10, 5, 0.16, 0.9, 0.1
S_list = [8, 9, 10, 11, 12]
ref = {0.0625: [2.0000, 1.1076, 0.5199, 0.2135, 0.0820],
       0.25:   [2.0785, 1.3336, 0.7959, 0.4482, 0.2427]}

for v0 in (0.0625, 0.25):
    print(f"   v0 = {v0}")
    print(f"   {'S':>4}{'DSINC':>11}{'FD-ADI':>11}{'reference':>11}"
          f"{'err(DS)':>10}{'err(FD)':>10}")
    es, ef = [], []
    for S, refS in zip(S_list, ref[v0]):
        (_, sPA, _), (_, fPA), _ = price_both(S, K, T, r, 0.0, theta, vov, rho, v0, kappa)
        es.append(sPA - refS); ef.append(fPA - refS)
        print(f"   {S:>4}{sPA:>11.4f}{fPA:>11.4f}{refS:>11.4f}{es[-1]:>+10.4f}{ef[-1]:>+10.4f}")
    print(f"   {'RMS':>4}{'':22}{'':11}{np.sqrt(np.mean(np.square(es))):>10.4f}"
          f"{np.sqrt(np.mean(np.square(ef))):>10.4f}\n")
