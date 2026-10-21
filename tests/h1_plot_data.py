#!/usr/bin/env python3
"""Publication figures for the one-pass H1 scan (PNG only).

Input: ``tests/h1_full_scan/scan_all.h5`` -- the output of the one-pass scan
(axis + walk + eps on every surface).  Optional: ``eps_refine.h5`` with the
full-tier re-measurement of the best configurations; when present its numbers
are printed for reference.

Terminology:

  * a CONFIGURATION is one set of coil currents (one Sobol sample);
  * a MAGNETIC SURFACE (or just surface) is one flux surface of a
    configuration.  A usable configuration contributes a set of continuously
    nested surfaces, the outermost being the LCFS.

ε shown here comes from the walk tier (nturn=200, npart=500): it is meant for
ranking and for ε(a) maps; absolute values are ~20 % below the full tier, which
``eps_refine.h5`` corrects for the best candidates.

Figures
  fig_axis_reach      6 panels: any two currents vs the real-axis reach
  fig_axis_summary    reach histogram, fixed-point count, axis radius
  fig_depth_surface   6 panels: any two currents vs the LCFS minor radius
  fig_depth_physics   LCFS a distribution and a vs dr / iota / mirror ratio
  fig_eps_surface     6 panels: any two currents vs eps at a = A0
  fig_eps_physics     eps distribution, V vs iota, a vs elongation
  fig_qc              gate and resolution diagnostics

  python -u tests/h1_plot_data.py                # all figures
  python -u tests/h1_plot_data.py --only eps     # one group
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from ripplepy.plotting import PUB_COLORS, setup_publication_style

setup_publication_style()                       # must precede pyplot (Agg)
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.cm import ScalarMappable                                 # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# the scan is written to h1_full_scan on the server and downloaded to
# h1_radius_scan locally; take whichever exists
_CANDIDATES = [BASE / "tests" / "h1_full_scan" / "scan_all.h5",
               BASE / "tests" / "h1_radius_scan" / "scan_all.h5"]
SCAN_H5 = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
DATA_DIR = SCAN_H5.parent
REFINE_H5 = DATA_DIR / "eps_refine.h5"
OUTDIR = DATA_DIR / "figs"

BOUNDS = np.array([
    [50000.0, 50000.0],      # coil 0: TF, fixed
    [0.0, 10000.0],          # coil 1
    [0.0, 10000.0],          # coil 2
    [-100000.0, 0.0],        # coil 3
    [-100000.0, 0.0],        # coil 4
])
FREE = np.array([1, 2, 3, 4])
PAIRS = list(itertools.combinations(range(len(FREE)), 2))

A0 = 0.10               # reference minor radius for the config-level eps
REACH_CLASSES = np.array([0.0, 0.02, 0.04])
DPI = 300
QC_MAX = 1.5
N_TREND = 8

CMAP_LEN = "viridis"
CMAP_EPS = "viridis_r"          # low eps = bright
NORM_LEN = Normalize(0.0, 0.25)
CMAP_IOTA = "viridis"
NORM_IOTA = Normalize(0.4, 1.8)
CMAP_V = "viridis"
NORM_V = Normalize(0.0, 1.2)
REACH_CMAP = ListedColormap(["0.84",
                             plt.get_cmap(CMAP_LEN)(0.55),
                             plt.get_cmap(CMAP_LEN)(0.95)])

COIL = lambda k: f"$I_{k+1}$ [kA]"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_data(path=SCAN_H5):
    with h5py.File(path, "r") as f:
        S = f["samples"][:]
        sc = {k: f["scalars"][k][:] for k in f["scalars"]}
        prof = {k: f["profile"][k][:] for k in f["profile"]}
    ok = prof["status"] == 0
    ci, _ = np.nonzero(ok)
    surf = {k: v[ok] for k, v in prof.items()}
    surf["config"] = ci

    # config-level reference surface: the valid face closest to a = A0
    n = len(S)
    ref = {k: np.full(n, np.nan) for k in
           ("eps", "a", "dr", "iota", "volume", "drift", "elongation")}
    for i in range(n):
        # require the column to reach A0 (otherwise the closest face is a
        # smaller plasma's LCFS and the radii would be mixed)
        m = ok[i] & np.isfinite(prof["a"][i]) & (prof["a"][i] >= 0.95 * A0)
        if not m.any():
            continue
        j = np.flatnonzero(m)[int(np.argmin(np.abs(prof["a"][i][m] - A0)))]
        for k in ref:
            ref[k][i] = prof[k][i, j] if k in prof else np.nan
    e = ref["eps"]
    good = np.isfinite(e) & (e > 0)
    lo, hi = np.percentile(e[good], [2, 98])
    norm_eps = LogNorm(max(lo, 1e-3), hi)
    return {"samples": S, "scalars": sc, "profile": prof, "surf": surf,
            "ref": ref, "ref_ok": good, "norm_eps": norm_eps}


def kA(S, idx):
    return S[:, FREE[idx]] / 1e3


def kA_range(idx):
    return (BOUNDS[FREE[idx], 0] / 1e3, BOUNDS[FREE[idx], 1] / 1e3)


def panel(ax, label):
    kw = dict(transform=ax.transAxes, ha="left", va="top", fontsize=11,
              fontweight="bold", zorder=10,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                        pad=1.2))
    if hasattr(ax, "text2D"):
        ax.text2D(0.015, 0.985, label, **kw)
    else:
        ax.text(0.015, 0.985, label, **kw)


def cbar(fig, mappable, ax, label, **kw):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03, **kw)
    cb.set_label(label, labelpad=6)
    cb.ax.yaxis.set_label_position("right")
    cb.outline.set_linewidth(0.6)
    return cb


def trend(x, y, n_bins=N_TREND):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < n_bins * 3:
        return np.array([]), np.array([])
    q = np.quantile(x[ok], np.linspace(0, 1, n_bins + 1))
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = ok & (x >= lo) & (x <= hi)
        if m.sum() >= 3:
            xs.append(np.median(x[m]))
            ys.append(np.median(y[m]))
    return np.array(xs), np.array(ys)


def save(fig, stem):
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTDIR) / f"{stem}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def pair_scatter(S, z, zlabel, cmap, norm, fname, title):
    fig, axs = plt.subplots(2, 3, figsize=(13.0, 7.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)
    for k, (i, j) in enumerate(PAIRS):
        ax = axs.flat[k]
        sc_ = ax.scatter(kA(S, i), kA(S, j), c=z, s=2.5, cmap=cmap,
                         norm=norm, linewidths=0, rasterized=True)
        ax.set_xlabel(COIL(i))
        ax.set_ylabel(COIL(j))
        panel(ax, f"({chr(97 + k)})")
    cb = fig.colorbar(sc_, ax=axs.ravel().tolist(), fraction=0.02, pad=0.02)
    cb.set_label(zlabel, labelpad=6)
    cb.ax.yaxis.set_label_position("right")
    cb.outline.set_linewidth(0.6)
    fig.suptitle(title, y=1.00, fontsize=12)
    save(fig, fname)


# ---------------------------------------------------------------------------
# figure 1: real-axis reach against any two currents
# ---------------------------------------------------------------------------
def fig_axis_reach(d):
    S, sc = d["samples"], d["scalars"]
    reach = np.where(np.isfinite(sc["axis_reach"]) & (sc["axis_reach"] > 0),
                     sc["axis_reach"], 0.0)
    norm = BoundaryNorm([-0.01, 0.01, 0.03, 0.05], REACH_CMAP.N)
    fig, axs = plt.subplots(2, 3, figsize=(13.0, 7.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)
    for k, (i, j) in enumerate(PAIRS):
        ax = axs.flat[k]
        ax.scatter(kA(S, i), kA(S, j), c=reach, cmap=REACH_CMAP, norm=norm,
                   s=1.5, linewidths=0, rasterized=True)
        ax.set_xlabel(COIL(i))
        ax.set_ylabel(COIL(j))
        panel(ax, f"({chr(97 + k)})")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=REACH_CMAP),
                      ax=axs.ravel().tolist(), fraction=0.02, pad=0.02,
                      ticks=REACH_CLASSES)
    cb.set_label("axis reach [m]", labelpad=6)
    cb.ax.yaxis.set_label_position("right")
    cb.outline.set_linewidth(0.6)
    fig.suptitle(f"real magnetic axis: {100 * np.mean(reach > 0):.1f}% of "
                 f"{len(S):,} configurations", y=1.00, fontsize=12)
    print(f"[axis_reach] {len(S):,} configurations")
    save(fig, "fig_axis_reach")


# ---------------------------------------------------------------------------
# figure 2: axis summary
# ---------------------------------------------------------------------------
def fig_axis_summary(d):
    S, sc = d["samples"], d["scalars"]
    reach = np.where(np.isfinite(sc["axis_reach"]), sc["axis_reach"], np.nan)
    reach = np.where(np.isnan(reach), -1.0, reach)
    n_fix = sc["axis_count"]
    axisR = sc["axis_R"]
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.6))
    fig.subplots_adjust(wspace=0.30)

    ax = axs[0]
    vals, cnts = np.unique(np.where(reach > 0, reach, 0.0), return_counts=True)
    bars = ax.bar([f"{v:.2f}" for v in vals], cnts,
                  color=[REACH_CMAP(i) for i in range(len(vals))],
                  edgecolor="k", linewidth=0.6)
    for b, c in zip(bars, cnts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{100.0 * c / len(S):.1f}%", ha="center", va="bottom",
                fontsize=9)
    ax.set_xlabel("axis reach [m]")
    ax.set_ylabel("configurations")
    ax.set_title(f"real axis: {100.0 * np.mean(reach > 0):.1f}%")
    panel(ax, "(a)")

    ax = axs[1]
    vals, cnts = np.unique(n_fix[np.isfinite(n_fix)], return_counts=True)
    ax.bar([f"{int(v)}" for v in vals], cnts, color=PUB_COLORS["blue"],
           edgecolor="k", linewidth=0.6)
    ax.set_xlabel("distinct one-turn fixed points")
    ax.set_ylabel("configurations")
    ax.set_title(f"multi-axis candidates: {100.0 * np.mean(n_fix > 1):.1f}%")
    panel(ax, "(b)")

    ax = axs[2]
    bins = np.linspace(1.08, 1.38, 31)
    ax.hist(axisR[np.isfinite(axisR)], bins=bins, color="0.75", edgecolor="k",
            linewidth=0.4, label="all")
    ax.hist(axisR[reach > 0], bins=bins, color=PUB_COLORS["blue"],
            edgecolor="k", linewidth=0.4, alpha=0.9, label="reach $>$ 0")
    ax.set_xlabel("axis $R$ at the outboard crossing [m]")
    ax.set_ylabel("configurations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    panel(ax, "(c)")
    print(f"[axis_summary] {len(S):,} configurations")
    save(fig, "fig_axis_summary")


# ---------------------------------------------------------------------------
# figure 3: LCFS minor radius against any two currents
# ---------------------------------------------------------------------------
def fig_depth_surface(d):
    a = d["scalars"]["lcfs_a"]
    good = np.isfinite(a) & (a > 0)
    print(f"[depth_surface] {int(good.sum()):,} configurations with an LCFS")
    pair_scatter(d["samples"][good], a[good], "$a_\\mathrm{LCFS}$ [m]",
                 CMAP_LEN, NORM_LEN, "fig_depth_surface",
                 f"LCFS minor radius of {int(good.sum()):,} configurations")


# ---------------------------------------------------------------------------
# figure 4: depth physics
# ---------------------------------------------------------------------------
def fig_depth_physics(d):
    surf = d["surf"]
    a, dr, io = surf["a"], surf["dr"], surf["iota"]
    bmin, bmax = surf["Bmin"], surf["Bmax"]
    mirror = (bmax - bmin) / (bmax + bmin)
    eps = surf["eps"]
    good = np.isfinite(a) & (a > 0)
    have = good & np.isfinite(eps) & (eps > 0)

    fig = plt.figure(figsize=(13.0, 7.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    ax = fig.add_subplot(2, 2, 1)
    lc = d["scalars"]["lcfs_a"]
    lc = lc[np.isfinite(lc) & (lc > 0)]
    ax.hist(lc, bins=np.linspace(0, 0.25, 41), color=PUB_COLORS["blue"],
            edgecolor="k", linewidth=0.3)
    ax.set_xlabel("$a_\\mathrm{LCFS}$ [m]")
    ax.set_ylabel("configurations")
    ax.set_title(f"{len(lc):,} LCFS")
    panel(ax, "(a)")

    panels = [(dr, "$dr$ [m]", "(b)"),
              (io, r"$\iota$", "(c)"),
              (mirror, "mirror ratio $(B_\\mathrm{max}-B_\\mathrm{min})/"
                       "(B_\\mathrm{max}+B_\\mathrm{min})$", "(d)")]
    for k, (y, ylabel, lab) in enumerate(panels):
        ax = fig.add_subplot(2, 2, k + 2)
        m = good & np.isfinite(y)
        ax.scatter(a[m & ~have], y[m & ~have], s=1.5, c="0.85",
                   linewidths=0, rasterized=True, zorder=1)
        sc_ = ax.scatter(a[m & have], y[m & have], s=2, c=eps[m & have],
                         cmap=CMAP_EPS, norm=d["norm_eps"], linewidths=0,
                         rasterized=True, zorder=2)
        ax.set_xlabel("minor radius $a$ [m]")
        ax.set_ylabel(ylabel)
        panel(ax, lab)
    cb = fig.colorbar(sc_, ax=[fig.axes[i] for i in (1, 2, 3)],
                      fraction=0.025, pad=0.02)
    cb.set_label(r"$\varepsilon_\mathrm{eff}^{3/2}$ (walk tier)",
                 labelpad=6)
    cb.ax.yaxis.set_label_position("right")
    cb.outline.set_linewidth(0.6)
    print(f"[depth_physics] {int(good.sum()):,} surfaces, "
          f"{int(have.sum()):,} with eps")
    save(fig, "fig_depth_physics")


# ---------------------------------------------------------------------------
# figure 5: eps at the reference surface against any two currents
# ---------------------------------------------------------------------------
def fig_eps_surface(d):
    ref, ok = d["ref"], d["ref_ok"]
    a_med = np.median(ref["a"][ok])
    print(f"[eps_surface] {int(ok.sum()):,} configurations with eps at "
          f"a ~ {A0} m")
    pair_scatter(d["samples"][ok], ref["eps"][ok],
                 r"$\varepsilon_\mathrm{eff}^{3/2}$", CMAP_EPS,
                 d["norm_eps"], "fig_eps_surface",
                 f"$\\varepsilon_\\mathrm{{eff}}^{{3/2}}$ on the "
                 f"$a={a_med:.3f}$ m surface, {int(ok.sum()):,} configurations")


# ---------------------------------------------------------------------------
# figure 6: eps against the other physical quantities
# ---------------------------------------------------------------------------
def fig_eps_physics(d):
    ref, ok = d["ref"], d["ref_ok"]
    sc = d["scalars"]
    eps = ref["eps"]
    V = sc["lcfs_volume"]
    a_lcfs = sc["lcfs_a"]
    elong = sc["lcfs_elongation"]
    iota = ref["iota"]

    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.4))
    fig.subplots_adjust(wspace=0.34)

    ax = axs[0]
    bins = np.linspace(0, min(np.nanmax(eps[ok]), 1.5), 40)
    ax.hist(eps[ok], bins=bins, color=PUB_COLORS["blue"], edgecolor="k",
            linewidth=0.3)
    ax.set_xlabel(r"$\varepsilon_\mathrm{eff}^{3/2}$")
    ax.set_ylabel("configurations")
    panel(ax, "(a)")

    panels = [(iota, V, rf"$\iota$ at the $a={np.median(ref['a'][ok]):.3f}$ m"
               " surface", "LCFS volume $V$ [m$^3$]", "(b)"),
              (a_lcfs, elong, "LCFS minor radius $a$ [m]",
               "elongation", "(c)")]
    for k, (x, y, xlabel, ylabel, lab) in enumerate(panels):
        ax = axs[k + 1]
        m = ok & np.isfinite(x) & np.isfinite(y)
        sc_ = ax.scatter(x[m], y[m], c=eps[m], cmap=CMAP_EPS,
                         norm=d["norm_eps"], s=6, linewidths=0,
                         rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        panel(ax, lab)
    cb = fig.colorbar(sc_, ax=axs[1:].tolist(), fraction=0.05, pad=0.03)
    cb.set_label(r"$\varepsilon_\mathrm{eff}^{3/2}$", labelpad=6)
    cb.ax.yaxis.set_label_position("right")
    cb.outline.set_linewidth(0.6)
    print(f"[eps_physics] {int(ok.sum()):,} configurations")
    save(fig, "fig_eps_physics")


# ---------------------------------------------------------------------------
# figure 7: gate and resolution diagnostics
# ---------------------------------------------------------------------------
def fig_qc(d):
    prof = d["profile"]
    ok = prof["status"] == 0
    res, gn = prof["residual"][ok], prof["gap_norm"][ok]
    drift = prof["gradpsi_drift"][ok]
    m = np.isfinite(res) & np.isfinite(gn)

    fig, axs = plt.subplots(2, 3, figsize=(13.0, 7.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.34)

    ax = axs[0, 0]
    ax.hist(res[m], bins=np.linspace(0, 0.05, 60), color=PUB_COLORS["blue"],
            edgecolor="k", linewidth=0.3)
    ax.axvline(0.05, color=PUB_COLORS["red"], ls="--", lw=1.0)
    ax.set_xlabel("Poincar\u00e9-section fit residual (gate 0.05)")
    ax.set_ylabel("surfaces")
    panel(ax, "(a)")

    ax = axs[0, 1]
    ax.hist(gn[m], bins=np.linspace(0, 15, 60), color=PUB_COLORS["blue"],
            edgecolor="k", linewidth=0.3)
    ax.axvline(15.0, color=PUB_COLORS["red"], ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"gap$_\mathrm{norm}$ (gate 15)")
    ax.set_ylabel("surfaces")
    panel(ax, "(b)")

    ax = axs[0, 2]
    h = ax.hist2d(gn[m], res[m], bins=(40, 40), cmap="Greys", cmin=0.5,
                  range=[[0, 16], [0, 0.06]])
    ax.axhline(0.05, color=PUB_COLORS["red"], ls="--", lw=1.0)
    ax.axvline(15.0, color=PUB_COLORS["red"], ls="--", lw=1.0)
    cb = cbar(fig, h[3], ax, "surfaces")
    cb.ax.set_title("")
    ax.set_xlabel(r"gap$_\mathrm{norm}$")
    ax.set_ylabel("residual")
    panel(ax, "(c)")

    ax = axs[1, 0]
    dv = drift[np.isfinite(drift)]
    ax.hist(dv, bins=np.linspace(0, 3, 60), color=PUB_COLORS["blue"],
            edgecolor="k", linewidth=0.3)
    ax.axvline(QC_MAX, color=PUB_COLORS["red"], ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("grad-$\\psi$ drift (last/first decile)")
    ax.set_ylabel("surfaces")
    panel(ax, "(d)")

    ax = axs[1, 1]
    ref, ok_ref = d["ref"], d["ref_ok"]
    S = d["samples"][ok_ref]
    U = (S[:, FREE] - BOUNDS[FREE, 0]) / (BOUNDS[FREE, 1] - BOUNDS[FREE, 0])
    _, jj = cKDTree(U).query(U, k=2)
    eps = ref["eps"]
    rel = np.abs(eps[ok_ref] - eps[ok_ref][jj[:, 1]]) / eps[ok_ref]
    ax.scatter(eps[ok_ref], rel, s=3, c="0.55", linewidths=0, rasterized=True)
    xs, ys = trend(eps[ok_ref], rel)
    if len(xs):
        ax.plot(xs, ys, "o-", color=PUB_COLORS["red"], lw=1.4, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon_\mathrm{eff}^{3/2}$")
    ax.set_ylabel(r"$|\Delta\varepsilon|/\varepsilon$ to nearest neighbour")
    panel(ax, "(e)")

    ax = axs[1, 2]
    names = ["ok", "trace", "residual", "gap", "geometry"]
    counts = [int((prof["status"] == k).sum()) for k in range(5)]
    ax.bar(names, counts, color=[PUB_COLORS["blue"], PUB_COLORS["orange"],
                                 PUB_COLORS["red"], PUB_COLORS["magenta"],
                                 PUB_COLORS["black"]], edgecolor="k",
           linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("surfaces (log)")
    ax.set_title(f"{int(ok.sum()):,} accepted / "
                 f"{int(np.isfinite(prof['status']).sum()):,} measured")
    panel(ax, "(f)")
    print(f"[qc] {int(ok.sum()):,} accepted surfaces")
    save(fig, "fig_qc")


# ---------------------------------------------------------------------------
def _refine_note():
    if not REFINE_H5.exists():
        return
    with h5py.File(REFINE_H5, "r") as f:
        e = f["eps_a0"][:]
        w = f["eps_a0_walk"][:]
        el = f["eps_lcfs"][:]
    m = np.isfinite(e) & np.isfinite(w) & (w > 0)
    if m.any():
        r = e[m] / w[m]
        print(f"[refine] {int(m.sum())} full-tier points: best eps(a0) = "
              f"{np.nanmin(e):.4f}, full/walk median {np.median(r):.3f}")
    if np.isfinite(el).any():
        print(f"[refine] eps at the LCFS: median {np.nanmedian(el):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "axis", "depth", "eps", "qc"])
    args = ap.parse_args()
    print(f"input: {SCAN_H5}")
    d = load_data()
    _refine_note()
    if args.only in ("all", "axis"):
        fig_axis_reach(d)
        fig_axis_summary(d)
    if args.only in ("all", "depth"):
        fig_depth_surface(d)
        fig_depth_physics(d)
    if args.only in ("all", "eps"):
        fig_eps_surface(d)
        fig_eps_physics(d)
    if args.only in ("all", "qc"):
        fig_qc(d)


if __name__ == "__main__":
    main()
