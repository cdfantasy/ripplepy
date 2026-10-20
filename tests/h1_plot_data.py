#!/usr/bin/env python3
"""Publication-quality figures for the staged current scan.

Two independent figures, one per stage:

  fig_axis   (stage 1, axis_pilot.h5)   where in the box a REAL magnetic axis
                                        exists, and how many branches there are
  fig_depth  (stage 2, depth_map.h5)    how deep each column survives, its
                                        geometry, the gate margins, and the
                                        depth map in a 2D slice of the box

  python -u tests/h1_plot_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import h5py

from ripplepy.plotting import PUB_COLORS, save_figure, setup_publication_style

setup_publication_style()          # must precede pyplot (Agg backend)
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
AXIS_H5 = BASE / "tests" / "h1_radial_scan" / "axis_pilot.h5"
DEPTH_H5 = BASE / "tests" / "h1_radial_scan" / "depth_map.h5"
EPS_H5 = BASE / "tests" / "h1_radial_scan" / "eps_ref.h5"
OUTDIR = BASE / "tests" / "h1_radial_scan" / "figs"

A0 = 0.10               # reference minor radius used by stage 3 (vertical line)
N_PROFILE_LINES = 60    # how many columns to draw in the profile panel
DPI = 300

COIL_LABELS = ("$I_1$ [kA]", "$I_2$ [kA]", "$I_3$ [kA]", "$I_4$ [kA]")
COIL_KEYS = (1, 2, 3, 4)          # free coils in the samples array


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _panel(ax, label):
    ax.text(0.015, 0.985, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold", zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))


def _cbar(fig, mappable, ax, label, as_title=False, **kw):
    """Colorbar with the label on the right (or on top when space is tight)."""
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03, **kw)
    if as_title:
        cb.ax.set_title(label, fontsize=8, pad=4)
    else:
        cb.set_label(label, labelpad=5)
        cb.ax.yaxis.set_label_position("right")
    cb.outline.set_linewidth(0.6)
    return cb


def _kA(x):
    return np.asarray(x, dtype=float) / 1e3


def _load_h5(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f.keys()}


# ---------------------------------------------------------------------------
# figure 1: stage 1, the axis region
# ---------------------------------------------------------------------------
def fig_axis(path=AXIS_H5, outdir=OUTDIR):
    d = _load_h5(Path(path))
    S = d["samples"]
    reach = d["reach"]
    n_axes = d["n_axes"]
    axisR = d["axis_R"]
    n = len(S)
    real = reach > 0

    fig, axs = plt.subplots(2, 3, figsize=(13.0, 7.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.38)

    # (a) reach classes
    ax = axs[0, 0]
    vals, cnts = np.unique(reach[np.isfinite(reach)], return_counts=True)
    bars = ax.bar([f"{v:.2f}" for v in vals], cnts,
                  color=[PUB_COLORS["red"], PUB_COLORS["orange"],
                         PUB_COLORS["green"]][:len(vals)],
                  edgecolor="k", linewidth=0.6)
    for b, c in zip(bars, cnts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{100.0 * c / n:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("column reach after the real-axis probe [m]")
    ax.set_ylabel("samples")
    ax.set_title(f"axis region: {100.0 * real.sum() / n:.1f}% reach $>$ 0")
    _panel(ax, "(a)")

    # (b) number of one-turn fixed points
    ax = axs[0, 1]
    vals, cnts = np.unique(n_axes[np.isfinite(n_axes)], return_counts=True)
    ax.bar([f"{int(v)}" for v in vals], cnts, color=PUB_COLORS["blue"],
           edgecolor="k", linewidth=0.6)
    ax.set_xlabel("distinct one-turn fixed points")
    ax.set_ylabel("samples")
    ax.set_title(f"multi-axis: {100.0 * np.mean(n_axes > 1):.1f}%")
    _panel(ax, "(b)")

    # (c) axis radius
    ax = axs[0, 2]
    bins = np.linspace(1.08, 1.38, 31)
    ax.hist(axisR[np.isfinite(axisR)], bins=bins, color="0.75",
            edgecolor="k", linewidth=0.4, label="all")
    ax.hist(axisR[real], bins=bins, color=PUB_COLORS["green"],
            edgecolor="k", linewidth=0.4, alpha=0.9, label="reach $>$ 0")
    ax.set_xlabel("axis $R$ [m]")
    ax.set_ylabel("samples")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title("where the axis sits")
    _panel(ax, "(c)")

    # (d)-(f) 2D slices of the 4D box, colour = reach class
    cmap = ListedColormap([PUB_COLORS["red"], PUB_COLORS["orange"],
                           PUB_COLORS["green"]])
    norm = BoundaryNorm([-0.01, 0.01, 0.03, 0.05], cmap.N)
    pairs = [(0, 2), (1, 3), (2, 3)]
    labels = ["(d)", "(e)", "(f)"]
    for k, ((i, j), lab) in enumerate(zip(pairs, labels)):
        ax = axs[1, k]
        bg = ~real
        ax.scatter(_kA(S[bg, COIL_KEYS[i]]), _kA(S[bg, COIL_KEYS[j]]),
                   s=1.4, c="0.86", linewidths=0, rasterized=True, zorder=1)
        sc = ax.scatter(_kA(S[real, COIL_KEYS[i]]), _kA(S[real, COIL_KEYS[j]]),
                        s=2.5, c=reach[real], cmap=cmap, norm=norm,
                        linewidths=0, rasterized=True, zorder=2)
        ax.set_xlabel(COIL_LABELS[COIL_KEYS[i] - 1])
        ax.set_ylabel(COIL_LABELS[COIL_KEYS[j] - 1])
        _panel(ax, lab)
    _cbar(fig, sc, axs[1, 2], "probe reach [m]", ticks=[0.0, 0.02, 0.04])

    print(f"[axis] {n} samples, real axis {int(real.sum())} "
          f"({100.0 * real.sum() / n:.1f}%), multi-axis "
          f"{100.0 * np.mean(n_axes > 1):.1f}%")
    out = Path(outdir) / "fig_axis"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, dpi=DPI)
    plt.close(fig)
    print(f"[axis] -> {out}.pdf / .png")


# ---------------------------------------------------------------------------
# figure 2: stage 2, the depth map
# ---------------------------------------------------------------------------
def fig_depth(path=DEPTH_H5, outdir=OUTDIR):
    from ripplepy.radial_scan import load_scan_h5
    d = load_scan_h5(Path(path))
    s, p = d["scalars"], d["profile"]
    S = d["samples"]
    dr, a, V = s["lcfs_dr"], s["lcfs_a"], s["lcfs_volume"]
    iota, gap = s["lcfs_iota"], s["lcfs_gap_norm"]
    ok = np.isfinite(dr) & (dr > 0) & np.isfinite(a)
    n = len(dr)

    fig, axs = plt.subplots(2, 3, figsize=(13.0, 7.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.38)

    # (a) depth histogram
    ax = axs[0, 0]
    ax.hist(dr[ok], bins=np.arange(0.0, 0.185, 0.01), color=PUB_COLORS["blue"],
            edgecolor="k", linewidth=0.4)
    ax.axvline(np.median(dr[ok]), color=PUB_COLORS["red"], ls="--", lw=1.2,
               label=f"median {np.median(dr[ok]):.3f} m")
    ax.set_xlabel("LCFS depth  $dr$ [m]")
    ax.set_ylabel("columns")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title(f"{int(ok.sum())}/{n} columns with a nested column")
    _panel(ax, "(a)")

    # (b) minor radius vs depth, coloured by iota
    ax = axs[0, 1]
    sc = ax.scatter(dr[ok], a[ok], c=iota[ok], s=6, cmap="viridis",
                    linewidths=0, rasterized=True)
    ax.set_xlabel("LCFS depth  $dr$ [m]")
    ax.set_ylabel("LCFS minor radius  $a$ [m]")
    _cbar(fig, sc, ax, r"$\iota$")
    _panel(ax, "(b)")

    # (c) size plane
    ax = axs[0, 2]
    sc = ax.scatter(a[ok], V[ok], c=dr[ok], s=6, cmap="magma",
                    linewidths=0, rasterized=True)
    ax.set_xlabel("LCFS minor radius  $a$ [m]")
    ax.set_ylabel("LCFS volume  $V$ [m$^3$]")
    _cbar(fig, sc, ax, "depth $dr$ [m]")
    _panel(ax, "(c)")

    # (d) radial profiles of a random subset
    ax = axs[1, 0]
    okrow = p["status"] == 0
    rng = np.random.default_rng(0)
    cols = rng.choice(np.flatnonzero(ok), size=min(N_PROFILE_LINES, ok.sum()),
                      replace=False)
    dr_grid = np.arange(0.02, 0.31, 0.02)
    curves = []
    for i in cols:
        x = p["dr"][i]
        y = p["a"][i]
        m = np.isfinite(x) & np.isfinite(y) & okrow[i]
        if m.sum() < 2:
            continue
        ax.plot(x[m], y[m], color="0.72", lw=0.5, zorder=1)
        yy = np.interp(dr_grid, x[m], y[m], left=np.nan, right=np.nan)
        if np.isfinite(yy).sum() >= 2:
            curves.append(yy)
    if curves:
        arr = np.array(curves)
        cnt = np.isfinite(arr).sum(axis=0)
        med = np.full(arr.shape[1], np.nan)
        med[cnt >= 3] = np.nanmedian(arr[:, cnt >= 3], axis=0)
        ax.plot(dr_grid, med, color=PUB_COLORS["red"], lw=2.0, zorder=3,
                label="median")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_xlabel("$dr$ [m]")
    ax.set_ylabel("$a$ [m]")
    ax.set_xlim(0, 0.19)
    ax.set_title(f"$a(dr)$ for {len(curves)} columns")
    _panel(ax, "(d)")

    # (e) gate margins of the accepted faces
    ax = axs[1, 1]
    res = p["residual"][okrow]
    gn = p["gap_norm"][okrow]
    m = np.isfinite(res) & np.isfinite(gn)
    nb = 32 if m.sum() > 500 else 16
    h = ax.hist2d(gn[m], res[m], bins=(nb, nb), cmap="Greys", cmin=0.5,
                  range=[[0, 16], [0, 0.06]])
    ax.axhline(0.05, color=PUB_COLORS["red"], ls="--", lw=1.0)
    ax.axvline(15.0, color=PUB_COLORS["red"], ls="--", lw=1.0)
    ax.set_xlabel(r"gap$_\mathrm{norm}$  (gate 15)")
    ax.set_ylabel(r"residual (gate 0.05)")
    _cbar(fig, h[3], ax, "faces", as_title=True)
    ax.set_title(f"{int(m.sum())} accepted faces")
    _panel(ax, "(e)")

    # (f) depth map in a 2D slice
    ax = axs[1, 2]
    ax.scatter(_kA(S[~ok, 1]), _kA(S[~ok, 3]), s=1.4, c="0.86",
               linewidths=0, rasterized=True, zorder=1)
    sc = ax.scatter(_kA(S[ok, 1]), _kA(S[ok, 3]), c=dr[ok], s=6,
                    cmap="viridis", linewidths=0, rasterized=True)
    _cbar(fig, sc, ax, "depth $dr$ [m]")
    ax.set_xlabel(COIL_LABELS[0])
    ax.set_ylabel(COIL_LABELS[2])
    ax.set_title("depth map (2 of 4 dimensions)")
    _panel(ax, "(f)")

    print(f"[depth] {int(ok.sum())}/{n} columns; depth med "
          f"{np.median(dr[ok]):.3f} m; a med {np.median(a[ok]):.3f} m; "
          f"V med {np.median(V[ok]):.3f} m^3")
    out = Path(outdir) / "fig_depth"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, dpi=DPI)
    plt.close(fig)
    print(f"[depth] -> {out}.pdf / .png")


# ---------------------------------------------------------------------------
# figure 3: stage 3, the eps field at the reference surface
# ---------------------------------------------------------------------------
def fig_eps(path=EPS_H5, outdir=OUTDIR):
    with h5py.File(Path(path), "r") as f:
        S = f["samples"][:]
        r = {k: f["ref"][k][:] for k in f["ref"]}
        a0 = float(f.attrs["a0"])
    e, V, io = r["eps"], r["volume"], r["iota"]
    qc, st = r["gradpsi_drift"], r["status"]
    has = np.isfinite(e)
    clean = has & np.isfinite(qc) & (qc <= 1.5)
    print(f"[eps] {len(e)} measured, {int(has.sum())} with eps, "
          f"{int(clean.sum())} clean (drift <= 1.5)")

    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.52)

    # (a) eps distribution, all vs clean
    ax = axs[0, 0]
    bins = np.linspace(0, min(np.nanmax(e[has]), 1.5), 40)
    ax.hist(e[has], bins=bins, color="0.75", edgecolor="k", linewidth=0.4,
            label="all")
    ax.hist(e[clean], bins=bins, color=PUB_COLORS["blue"], edgecolor="k",
            linewidth=0.4, alpha=0.85, label="drift $\\leq$ 1.5")
    ax.set_xlabel(rf"$\varepsilon_\mathrm{{eff}}^{{3/2}}$ at $a={a0:.2f}$ m")
    ax.set_ylabel("columns")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _panel(ax, "(a)")

    # (b) eps vs volume, with the lower envelope
    ax = axs[0, 1]
    ax.scatter(V[has], e[has], s=8, c="0.7", linewidths=0, rasterized=True)
    ax.scatter(V[clean], e[clean], s=9, c=PUB_COLORS["blue"], linewidths=0,
               rasterized=True)
    if clean.any():
        o = np.argsort(V[clean])
        vv, ee = V[clean][o], e[clean][o]
        env, best = [], np.inf
        for x, y in zip(vv, ee):
            if y < best:
                best = y
                env.append((x, y))
        env = np.array(env)
        ax.plot(env[:, 0], env[:, 1], "-", color=PUB_COLORS["red"], lw=1.6,
                label="lower envelope")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(r"LCFS volume  $V$ [m$^3$]")
    ax.set_ylabel(rf"$\varepsilon_\mathrm{{eff}}^{{3/2}}$ at $a={a0:.2f}$ m")
    _panel(ax, "(b)")

    # (c) QC vs eps
    ax = axs[0, 2]
    ax.scatter(qc[has], e[has], s=8, c="0.7", linewidths=0, rasterized=True)
    ax.axvline(1.5, color=PUB_COLORS["red"], ls="--", lw=1.0)
    ax.set_xlabel(r"grad-$\psi$ drift (last/first decile)")
    ax.set_ylabel(rf"$\varepsilon_\mathrm{{eff}}^{{3/2}}$")
    ax.set_title(f"{int(has.sum() - clean.sum())} flagged as corrupted")
    _panel(ax, "(c)")

    # (d) local smoothness: eps difference to the nearest neighbour
    ax = axs[1, 0]
    from scipy.spatial import cKDTree
    span = np.ptp(S[:, 1:], axis=0)
    span[span <= 0] = 1.0
    U = S[clean][:, 1:] / span
    Ec = e[clean]
    if len(U) > 10:
        d, j = cKDTree(U).query(U, k=2)
        d, j = d[:, 1], j[:, 1]
        diff = np.abs(Ec - Ec[j]) / np.maximum(Ec, 1e-9)
        ax.scatter(Ec, diff, s=8, c="0.6", linewidths=0, rasterized=True)
        q = np.quantile(Ec, np.linspace(0, 1, 7))
        xs, ys = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            m = (Ec >= lo) & (Ec < hi)
            if m.sum() > 3:
                xs.append(0.5 * (lo + hi))
                ys.append(np.median(diff[m]))
        ax.plot(xs, ys, "o-", color=PUB_COLORS["red"], lw=1.6,
                label="median")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
        print(f"[eps] nearest-neighbour eps difference: median "
              f"{np.median(diff) * 100:.1f}%, p90 {np.percentile(diff, 90) * 100:.1f}%")
    ax.set_xlabel(rf"$\varepsilon_\mathrm{{eff}}^{{3/2}}$ at $a={a0:.2f}$ m")
    ax.set_ylabel("relative difference to nearest neighbour")
    _panel(ax, "(d)")

    # (e) where the good columns are
    ax = axs[1, 1]
    ax.scatter(_kA(S[has, 1]), _kA(S[has, 3]), c="0.85", s=3,
               linewidths=0, rasterized=True, zorder=1)
    sc = ax.scatter(_kA(S[clean, 1]), _kA(S[clean, 3]), c=Ec, s=9,
                    cmap="viridis_r", linewidths=0, rasterized=True, zorder=2,
                    vmin=np.percentile(Ec, 2), vmax=np.percentile(Ec, 98))
    _cbar(fig, sc, ax, r"$\varepsilon_\mathrm{{eff}}^{{3/2}}$")
    ax.set_xlabel(COIL_LABELS[0])
    ax.set_ylabel(COIL_LABELS[2])
    _panel(ax, "(e)")

    # (f) eps vs iota
    ax = axs[1, 2]
    sc = ax.scatter(io[clean], Ec, c=V[clean], s=9, cmap="magma",
                    linewidths=0, rasterized=True)
    _cbar(fig, sc, ax, r"$V$ [m$^3$]")
    ax.set_xlabel(r"$\iota$ at the reference surface")
    ax.set_ylabel(rf"$\varepsilon_\mathrm{{eff}}^{{3/2}}$")
    _panel(ax, "(f)")

    out = Path(outdir) / "fig_eps"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, dpi=DPI)
    plt.close(fig)
    print(f"[eps] -> {out}.pdf / .png")
    if clean.any():
        k = int(np.nanargmin(np.where(clean, e, np.nan)))
        print(f"[eps] best: eps={e[k]:.6f} V={V[k]:.3f} iota={io[k]:.3f} "
              f"QC={qc[k]:.2f} extcur={np.round(S[k], 1).tolist()}")
    miss = ~has
    if miss.any():
        for code, name in enumerate(("ok", "trace_fail", "residual", "gap",
                                     "bad_geom")):
            c = int(np.sum(st[miss] == code))
            if c:
                print(f"[eps] missing-eps reason {name}: {c}")


# ---------------------------------------------------------------------------
def main():
    fig_axis()
    fig_depth()
    fig_eps()


if __name__ == "__main__":
    main()
