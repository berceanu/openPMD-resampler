"""
Compare particles generated from a PIConGPU calorimeter (CSV written by
start.py --calorimeter_path) with the calorimeter itself.

Writes three figures to --out: compare_dNdE.png (energy spectrum),
compare_energy_yaw.png (d2N/dE dtheta, summed over pitch) and
compare_energy_yaw_polar.png (the same map in polar form). The CSV particles
are re-binned with the PIConGPU kernel's formulas into the calorimeter's own
bins, so every panel uses identical bins. Particle counts per energy band are
printed as ratios to the calorimeter.

    pixi run compare --calorimeter_path H_calorimeter_forward_200000.bp5 \
        --csv "resampled (random)=random.dat" "resampled (per bin)=per_bin.dat"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

from openpmd_resampler.calorimeter import (
    JOULE_PER_KEV,
    CalorimeterSampler,
    calorimeter_frame,
    read_calorimeter,
)
from openpmd_resampler.units import constants

CALORIMETER_LABEL = "PIConGPU's calorimeter"


def calorimeter_counts(cal, edges_kev):
    """Real particles per (energy, yaw) bin, summed over pitch; inner energy bins only."""
    lo, hi = edges_kev[:-1], edges_kev[1:]
    mean_kev = (hi - lo) / np.log(hi / lo) if cal.log_scale else 0.5 * (lo + hi)
    counts = cal.energy_joule[1:-1] / (mean_kev * JOULE_PER_KEV)[:, None, None]
    return counts.sum(axis=1)


def particle_counts(csv_path, cal, mass_mev):
    df = pd.read_csv(csv_path, skipinitialspace=True, dtype=np.float64)
    df.columns = [c.split(" ")[0] for c in df.columns]
    w = df["weights"].to_numpy()
    # Undo the y/z swap -> PIConGPU frame, then rotate into the calorimeter frame.
    p = df[["momentum_x_mev_c", "momentum_z_mev_c", "momentum_y_mev_c"]].to_numpy()
    del df
    norm = np.linalg.norm(p, axis=1)
    d = (p / norm[:, None]) @ calorimeter_frame(cal.pos_yaw_deg, cal.pos_pitch_deg).T
    yaw = np.arctan2(d[:, 0], d[:, 1])
    pitch = np.arcsin(np.clip(d[:, 2], -1.0, 1.0))
    kinetic_kev = 1e3 * (np.sqrt(norm**2 + mass_mev**2) - mass_mev)

    n_energy, n_pitch, n_yaw = cal.energy_joule.shape
    max_yaw, max_pitch = np.deg2rad(cal.max_yaw_deg), np.deg2rad(cal.max_pitch_deg)
    if cal.log_scale:
        x, x0, x1 = np.log10(kinetic_kev), np.log10(cal.min_energy_kev), np.log10(cal.max_energy_kev)
    else:
        x, x0, x1 = kinetic_kev, cal.min_energy_kev, cal.max_energy_kev
    energy_bin = np.floor((x - x0) / (x1 - x0) * (n_energy - 2)).astype(np.int64) + 1
    yaw_bin = np.floor((0.5 + 0.5 * yaw / max_yaw) * n_yaw).astype(np.int64)
    inside = (
        (np.abs(yaw) < max_yaw) & (np.abs(pitch) < max_pitch)
        & (energy_bin >= 1) & (energy_bin <= n_energy - 2)
    )
    yaw_bin = np.clip(yaw_bin, 0, n_yaw - 1)
    flat = (energy_bin[inside] - 1) * n_yaw + yaw_bin[inside]
    counts = np.bincount(flat, w[inside], minlength=(n_energy - 2) * n_yaw)
    return counts.reshape(n_energy - 2, n_yaw), w.size, w.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--calorimeter_path", required=True, help="the calorimeter .bp5 file the CSVs were generated from")
    parser.add_argument(
        "--csv", nargs="+", required=True, metavar="LABEL=PATH",
        help='CSV files from start.py, each with its legend label, e.g. "resampled (random)=random.dat"',
    )
    parser.add_argument("--mass", type=float, default=1836.152, help="relative to the electron mass")
    parser.add_argument("--emax", type=float, default=None, help="energy axis limit in MeV")
    parser.add_argument("--title", default="", help="figure title prefix, e.g. the iteration")
    parser.add_argument("--out", default=".", help="output directory for the figures")
    args = parser.parse_args()

    csv_items = []
    for item in args.csv:
        if "=" not in item:
            parser.error(f"--csv expects LABEL=PATH, got {item!r}")
        csv_items.append(tuple(item.split("=", 1)))

    compare(
        read_calorimeter(args.calorimeter_path),
        csv_items,
        args.mass * constants.electron_mass_mev_c2,
        out=args.out,
        emax=args.emax,
        title=args.title,
    )


def compare(cal, csv_items, mass_mev, out=".", emax=None, title=""):
    """
    Print the per-band count ratios and write the three figures.

    csv_items: (label, path) pairs. A CSV with no particle inside the
    calorimeter's energy and angle range is reported as having zero overlap and
    shown as an empty curve and panel; the reference plots are still written.
    """
    edges_kev = CalorimeterSampler(cal, None, mass_mev).energy_edges_kev()
    e_edges = edges_kev * 1e-3  # MeV
    e_centres = np.sqrt(e_edges[:-1] * e_edges[1:]) if cal.log_scale else 0.5 * (e_edges[:-1] + e_edges[1:])
    de = np.diff(e_edges)
    n_yaw = cal.energy_joule.shape[2]
    yaw_edges = np.linspace(-cal.max_yaw_deg, cal.max_yaw_deg, n_yaw + 1)
    dtheta = np.diff(yaw_edges)

    datasets = [(CALORIMETER_LABEL, calorimeter_counts(cal, edges_kev))]
    for label, path in csv_items:
        counts, number, total = particle_counts(path, cal, mass_mev)
        datasets.append((label, counts))
        print(f"{label}: {number:,} macroparticles, {total:.4e} real particles")

    reference_spectrum = datasets[0][1].sum(axis=1)
    reference_total = reference_spectrum.sum()
    reference_bins = np.nonzero(reference_spectrum)[0]
    if reference_bins.size == 0:
        raise SystemExit(f"{CALORIMETER_LABEL} has no particles in its inner energy bins; nothing to compare.")
    last_bin = reference_bins.max()
    emax = emax or 1.05 * e_edges[last_bin + 1]
    print(f"{CALORIMETER_LABEL}: {reference_total:.4e} real particles, max energy {e_edges[last_bin + 1]:.3f} MeV")
    for index, (label, counts) in enumerate(datasets[1:], start=1):
        spectrum = counts.sum(axis=1)
        filled = np.nonzero(spectrum)[0]
        if filled.size == 0:
            print(
                f"  {label}: zero overlap, no particle inside the calorimeter's range"
                f" ({e_edges[0]:.4g} - {e_edges[-1]:.4g} MeV, |yaw| < {cal.max_yaw_deg:g} deg,"
                f" |pitch| < {cal.max_pitch_deg:g} deg)"
            )
            datasets[index] = (f"{label} (no overlap)", counts)
            continue
        reach = e_edges[filled.max() + 1]
        rows = [f"{label}: max energy {reach:.3f} MeV"]
        for low, high in [(0, 1), (1, 3), (3, 6), (6, 1e9)]:
            band = (e_centres >= low) & (e_centres < high)
            ref = reference_spectrum[band].sum()
            if ref > 0:
                rows.append(f"N[{low}-{high if high < 1e9 else 'max'} MeV] ratio {spectrum[band].sum() / ref:.3f}")
        print("  " + ", ".join(rows))

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{title} " if title else ""

    # 1. dN/dE
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for (label, counts), style in zip(datasets, ["k-", "C1-", "C2--", "C3:"]):
        ax.semilogy(e_centres, counts.sum(axis=1) / de, style, lw=1.0, label=label)
    ax.set(xlim=(0, emax), xlabel="Energy (MeV)", ylabel=r"dN/dE (MeV$^{-1}$)", title=f"{prefix}dN/dE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "compare_dNdE.png", dpi=200)

    # 2. energy-yaw maps, shared log colour scale
    maps = [(label, counts / (de[:, None] * dtheta[None, :])) for label, counts in datasets]
    vmax = maps[0][1].max()
    norm = colors.LogNorm(vmin=vmax * 1e-7, vmax=vmax)
    fig, axs = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 3.4), sharey=True)
    for ax, (label, image) in zip(np.atleast_1d(axs), maps):
        mesh = ax.pcolormesh(yaw_edges, e_edges, np.ma.masked_less_equal(image, 0), norm=norm, cmap="jet")
        ax.set(ylim=(0, emax), xlabel=r"Vertical angle $\theta$ (deg)", title=label)
    np.atleast_1d(axs)[0].set_ylabel("Energy (MeV)")
    fig.colorbar(mesh, ax=axs, label=r"$d^2N/dE\,d\theta$ (MeV$^{-1}$ deg$^{-1}$)")
    fig.suptitle(prefix.strip(), y=1.04)
    fig.savefig(out / "compare_energy_yaw.png", dpi=200, bbox_inches="tight")

    # 3. polar view: 0 deg = laser axis pointing left, radius = energy
    fig, axs = plt.subplots(1, len(maps), figsize=(3.6 * len(maps), 4), subplot_kw={"projection": "polar"})
    theta_edges = np.deg2rad(yaw_edges)
    for ax, (label, image) in zip(np.atleast_1d(axs), maps):
        mesh = ax.pcolormesh(theta_edges, e_edges, np.ma.masked_less_equal(image, 0), norm=norm, cmap="jet")
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-cal.max_yaw_deg)
        ax.set_thetamax(cal.max_yaw_deg)
        ax.set_rlim(0, emax)
        ax.set_title(label)
    fig.colorbar(mesh, ax=axs, shrink=0.6, label=r"$d^2N/dE\,d\theta$ (MeV$^{-1}$ deg$^{-1}$)")
    fig.suptitle(prefix.strip(), y=0.9)
    fig.savefig(out / "compare_energy_yaw_polar.png", dpi=200, bbox_inches="tight")
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
