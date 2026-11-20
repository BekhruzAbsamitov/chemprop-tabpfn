"""scripts/gp_mordred_geometry.py — why the RBF GP collapses on Mordred (D2R pool, n = 100). Press Run.

Prints, per representation: effective rank, pairwise-distance scale and spread (CV, p99/p1),
the length scale and signal variance the benchmark GP fits, the resulting off-diagonal kernel
entries, and the spread of the predicted mean; for Mordred also the per-column scales and a
re-fit on standardised features. The values are copied into appendix_c_tables.py (Appendix C of the thesis).
"""
import sys
import warnings

import numpy as np
import torch

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, f"{REPO}/src/scripts")
warnings.filterwarnings("ignore")


def eff_rank(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    import benchmark_al as ba
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    smiles, y, _ = ba.load_pool(ba.TARGETS["d2r"])
    y = y.flatten().numpy().astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(smiles), 1000, replace=False)
    tr = idx[:100]                                   # an n = 100 labelled set, as in the benchmark
    print(f"{'representation':20s} {'d':>5} {'eff.rank':>8} {'median dist':>11} {'CV dist':>8} "
          f"{'p99/p1 dist':>11} {'fitted l':>9} {'sig_f^2':>8} {'k_offdiag mean':>14} {'k_offdiag sd':>12} "
          f"{'pred sd/y sd':>12}")
    for name in ["morgan", "physchem", "mordred3d", "chemeleon", "chemprop_untrained"]:
        kind, ckpt = ba.REPRESENTATIONS[name]
        X = ba.featurize_pool(kind, ckpt, smiles, torch.device("cpu")).float().numpy().astype(np.float64)
        S = X[idx]
        D = np.sqrt(np.maximum(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1), 0))
        off = D[np.triu_indices(len(S), 1)]
        off = off[off > 0]
        gp = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), alpha=0.1,
                                      n_restarts_optimizer=2, random_state=42)
        gp.fit(X[tr], y[tr])
        sig2, ell = gp.kernel_.k1.constant_value, gp.kernel_.k2.length_scale
        K = gp.kernel_(X[tr]) / sig2
        koff = K[np.triu_indices(len(tr), 1)]
        mu = gp.predict(X[idx[100:]])
        print(f"{name:20s} {X.shape[1]:5d} {eff_rank(S):8.1f} {np.median(off):11.3g} {off.std() / off.mean():8.3f} "
              f"{np.percentile(off, 99) / np.percentile(off, 1):11.2f} {ell:9.3g} {sig2:8.3g} {koff.mean():14.3g} "
              f"{koff.std():12.3g} {mu.std() / y[idx[100:]].std():12.3f}")
        if name == "mordred3d":
            col_sd = S.std(0)
            top = np.argsort(-col_sd)[:5]
            share = (col_sd[top] ** 2).sum() / (col_sd ** 2).sum()
            print(f"    mordred: largest column sds {np.round(col_sd[top], 1).tolist()}; top-5 columns carry "
                  f"{share:.1%} of the total variance (squared-distance budget)")
            Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
            gz = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), alpha=0.1,
                                          n_restarts_optimizer=2, random_state=42).fit(Z[tr], y[tr])
            Dz = np.sqrt(np.maximum(((Z[idx][:, None, :] - Z[idx][None, :, :]) ** 2).sum(-1), 0))
            oz = Dz[np.triu_indices(len(idx), 1)]
            oz = oz[oz > 0]
            muz = gz.predict(Z[idx[100:]])
            rho = np.corrcoef(np.argsort(np.argsort(muz)), np.argsort(np.argsort(y[idx[100:]])))[0, 1]
            rho_raw = np.corrcoef(np.argsort(np.argsort(mu)), np.argsort(np.argsort(y[idx[100:]])))[0, 1]
            print(f"    mordred standardised: median dist {np.median(oz):.3g}, CV {oz.std() / oz.mean():.3f}, "
                  f"fitted l {gz.kernel_.k2.length_scale:.3g}; Spearman of the mean on 900 held-out "
                  f"molecules: raw {rho_raw:+.3f} -> standardised {rho:+.3f}")


if __name__ == "__main__":
    main()
