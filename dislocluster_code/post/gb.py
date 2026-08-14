"""Grain-boundary profiles from MoDELib2-NNL (Zr3d_ghoniem) output.

Radially averaged profiles of the mobile and immobile fields as a function of
distance from the grain boundary, in the style of the 0-D/1-D march figures:
log y against linear x, one curve per dose.

Every outer face of the domain carries the Dirichlet condition, so the whole
surface is grain boundary and the distance is the minimum over all six faces.
Nodes are binned by that distance and averaged within each bin, which averages
over the faces and over position within a face.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .fields import (load_cd_fields, gb_distance, FAMILIES,
                     B_SI, OMEGA_SI, OMEGA_B3)

__all__ = ["profile", "plot_density_profiles", "plot_content_profiles",
           "plot_size_profiles", "plot_mobile_profiles"]


def profile(P, values, n_bins=60, max_nm=None):
    """Bin `values` by distance from the grain boundary.

    Returns (x [nm], mean value per bin). Bins with no nodes are dropped.
    """
    d_nm = gb_distance(P) * B_SI * 1.0e9
    hi = d_nm.max() if max_nm is None else min(max_nm, d_nm.max())
    edges = np.linspace(0.0, hi, n_bins + 1)
    idx = np.digitize(d_nm, edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    x, y = [], []
    for b in range(n_bins):
        sel = ok & (idx == b)
        if sel.sum() == 0:
            continue
        x.append(0.5 * (edges[b] + edges[b + 1]))
        y.append(float(np.mean(values[sel])))
    return np.asarray(x), np.asarray(y)


def _dose_colors(n):
    cmap = colormaps["coolwarm"]
    return [cmap(t) for t in np.linspace(0.0, 1.0, n)]


def _finish(ax, xlabel, ylabel, title, logy=True):
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)


def _load(evl_dir, steps):
    return {st: load_cd_fields(evl_dir, st) for st in steps}


def plot_density_profiles(evl_dir, steps, doses, out_file=None, max_nm=None,
                          n_bins=60):
    """Loop NUMBER DENSITY [m^-3] against distance from the grain boundary."""
    data = _load(evl_dir, steps)
    colors = _dose_colors(len(steps))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (label, ncol, ccol, bmag, _n, _c) in zip(axes, FAMILIES):
        for (st, dose), col in zip(zip(steps, doses), colors):
            P, F = data[st]
            # n is carried per b^3; per m^3 is n / b_SI^3
            x, y = profile(P, F[:, ncol] / B_SI ** 3, n_bins, max_nm)
            ax.plot(x, y, color=col, lw=1.8, label=f"{dose:g} dpa")
        _finish(ax, "distance from grain boundary, $x$ [nm]",
                "number density [m$^{-3}$]", f"{label} loop density")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_content_profiles(evl_dir, steps, doses, out_file=None, max_nm=None,
                          n_bins=60):
    """Loop CONTENT (defects per m^3) against distance from the grain boundary."""
    data = _load(evl_dir, steps)
    colors = _dose_colors(len(steps))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (label, ncol, ccol, bmag, _n, _c) in zip(axes, FAMILIES):
        for (st, dose), col in zip(zip(steps, doses), colors):
            P, F = data[st]
            # content is a volume fraction; defects per m^3 is c / Omega
            x, y = profile(P, F[:, ccol] / OMEGA_SI, n_bins, max_nm)
            ax.plot(x, y, color=col, lw=1.8, label=f"{dose:g} dpa")
        _finish(ax, "distance from grain boundary, $x$ [nm]",
                "stored defects [m$^{-3}$]", f"{label} loop content")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_size_profiles(evl_dir, steps, doses, out_file=None, max_nm=None,
                       n_bins=60, logy=False):
    """Mean loop DIAMETER [nm] against distance from the grain boundary.

    The diameter is formed from the binned mean content and density rather than
    by averaging nodal diameters: the size is a ratio of two fields, and the mean
    of a ratio is not the ratio of the means. Near the boundary the density rises
    by three orders while the content does not, so averaging nodal ratios there
    would be dominated by the sparsest nodes.
    """
    data = _load(evl_dir, steps)
    colors = _dose_colors(len(steps))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (label, ncol, ccol, bmag, _n, _c) in zip(axes, FAMILIES):
        for (st, dose), col in zip(zip(steps, doses), colors):
            P, F = data[st]
            x, nbar = profile(P, F[:, ncol], n_bins, max_nm)
            _, cbar = profile(P, F[:, ccol], n_bins, max_nm)
            m = cbar / np.maximum(nbar * OMEGA_B3, 1.0e-300)   # defects per loop
            r_b = np.sqrt(np.maximum(m, 0.0) * OMEGA_B3 / (np.pi * bmag))
            d_nm = 2.0 * r_b * B_SI * 1.0e9
            ax.plot(x, d_nm, color=col, lw=1.8, label=f"{dose:g} dpa")
        _finish(ax, "distance from grain boundary, $x$ [nm]",
                "mean loop diameter [nm]", f"{label} loop size", logy=logy)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_mobile_profiles(evl_dir, steps, doses, out_file=None, max_nm=None,
                         n_bins=60):
    """Mobile species concentration [m^-3] against distance from the boundary."""
    data = _load(evl_dir, steps)
    colors = _dose_colors(len(steps))
    names = [("Cv", 0, r"$C_v$"), ("Ci", 1, r"$C_i$"),
             ("C2i", 2, r"$C_{2i}$"), ("C3i", 3, r"$C_{3i}$")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (_k, col_idx, label) in zip(axes, names):
        for (st, dose), col in zip(zip(steps, doses), colors):
            P, F = data[st]
            x, y = profile(P, F[:, col_idx] / OMEGA_SI, n_bins, max_nm)
            ax.plot(x, y, color=col, lw=1.8, label=f"{dose:g} dpa")
        _finish(ax, "distance from grain boundary, $x$ [nm]",
                "concentration [m$^{-3}$]", f"{label} near the GB")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig
