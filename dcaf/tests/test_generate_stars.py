#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the generate_stars function.

Create a gif animation of the generation of stars in batches starting from a
random positions of stars.

Save the pdf of the closest neighbour of each new star and plot it as a pdf.

This test could be used for other custom generate_stars or PDF used for
sampling.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import cKDTree

from amuse.lab import Particles, units
from dcaf.factory.distance_based import generate_stars
from dcaf.utilities.sampler import lognormal_pdf


def test_generate_stars(
    Nstart=5,
    Nfinal=1000,
    nbatch=5,
    beta = 1,
    box_size=10.0 | units.parsec,
    out_gif="generate_stars.gif",
    pdf_figure = "nn_distance_pdf.png",
    pdf_func=lognormal_pdf,
    mu = -2.5,
    logsigma = 0.9,
    pdf_unit=units.parsec,
    min_separation=0.02 | units.parsec,
    seed=42
    ):
    #mu=-2.15, sigma=0.9):
    test_title = (
        r"Test: lognormal PDF: $\mu= %.2f$ log$_\sigma=%.2f$, "
        r"$\beta=%.2f$" % (mu, logsigma, beta)
    )

    rng = np.random.default_rng(seed)

    # --- Initial stars ---
    stars = Particles(Nstart)
    stars.x = rng.uniform(0, box_size.value_in(units.parsec), Nstart) | units.parsec
    stars.y = rng.uniform(0, box_size.value_in(units.parsec), Nstart) | units.parsec
    stars.z = rng.uniform(0, box_size.value_in(units.parsec), Nstart) | units.parsec
    stars.x -= box_size*0.5
    stars.y -= box_size*0.5
    stars.z -= box_size*0.5
    stars.vx = rng.normal(0, 1, Nstart) | units.kms
    stars.vy = rng.normal(0, 1, Nstart) | units.kms
    stars.vz = rng.normal(0, 1, Nstart) | units.kms

    # Track history of particle sets for animation
    history = [stars.copy()]

    # Store nearest-neighbor distances of new stars
    nn_distances = []

    while len(stars) < Nfinal:
        n_new = min(nbatch, Nfinal - len(stars))
        newstars = generate_stars(
            stars,
            n_new,
            beta = beta,
            box_size=box_size,
            pdf_func=pdf_func,
            pdf_unit=pdf_unit,
            min_separation=min_separation,
            seed=seed + len(history),
        )

        # compute nearest neighbor distance for each new star
        tree = cKDTree(np.column_stack([
            stars.x.value_in(pdf_unit),
            stars.y.value_in(pdf_unit),
            stars.z.value_in(pdf_unit),
        ]))
        for i in range(len(newstars)):
            d, _ = tree.query([
                newstars.x[i].value_in(pdf_unit),
                newstars.y[i].value_in(pdf_unit),
                newstars.z[i].value_in(pdf_unit),
            ], k=1)
            nn_distances.append(d)

        stars.add_particles(newstars)
        history.append(stars.copy())

    # --- Animation ---
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = box_size.value_in(pdf_unit)
    xlims = (-lim,lim)
    ax.set_xlim(xlims)
    ax.set_ylim(xlims)

    def update(frame):
        ax.clear()
        batch = history[frame]
        if frame == 0:
            ax.scatter(batch.x.value_in(pdf_unit), batch.y.value_in(pdf_unit), s=5, c="black")
        else:
            old = history[frame - 1]
            n_new = len(batch) - len(old)
            ax.scatter(old.x.value_in(pdf_unit), old.y.value_in(pdf_unit), s=5, c="black")
            ax.scatter(batch[-n_new:].x.value_in(pdf_unit),
                       batch[-n_new:].y.value_in(pdf_unit),
                       s=10, c="red")
        ax.set_title(f"%s\n Step {frame}: N={len(batch)} stars "%test_title)
        ax.set_xlabel(f"x [{pdf_unit}]"); ax.set_ylabel(f"y [{pdf_unit}]")
        ax.set_aspect("equal", adjustable="box")

    ani = FuncAnimation(fig, update, frames=len(history), interval=300, repeat=False)
    ani.save(out_gif, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"Saved animation to {out_gif}")

    # --- PDF comparison plot ---
    nn_distances = np.array(nn_distances)
    r_vals = np.linspace(nn_distances.min(), nn_distances.max(), 200)

    # normalize the requested PDF over [r_min, r_max]
    r_min = float(min_separation.value_in(pdf_unit))
    r_max = nn_distances.max()
    pdf_vals = pdf_func(r_vals)
    pdf_vals /= np.trapz(pdf_vals[(r_vals >= r_min) & (r_vals <= r_max)],
                         r_vals[(r_vals >= r_min) & (r_vals <= r_max)])

    plt.figure(figsize=(6, 4))
    plt.hist(nn_distances, bins=40, density=True, alpha=0.6, label="Closest \
             neighbour distane")
    plt.plot(r_vals, pdf_vals, 'r-', lw=2, label="Requested PDF")
    plt.xlabel(f"Nearest-neighbor distance [{pdf_unit}]")
    plt.ylabel("PDF")
    plt.legend()
    plt.tight_layout()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(pdf_figure, dpi=150)
    plt.close()
    print("Saved PDF comparison plot as nn_distance_pdf.png")

def get_distribution_from_generate_stars(
    Nstart=50,
    Nfinal=500,
    nbatch=5,
    box_size=1.0 | units.parsec,
    pdf_func=lognormal_pdf,
    pdf_unit=units.parsec,
    min_separation=0.01 | units.parsec,
    seed=42,
    track_nn=True
):
    rng = np.random.default_rng(seed)

    # --- Initial stars ---
    stars = Particles(Nstart)
    stars.x = rng.uniform(0, box_size.value_in(units.parsec), Nstart) | units.parsec
    stars.y = rng.uniform(0, box_size.value_in(units.parsec), Nstart) | units.parsec
    stars.z = rng.uniform(0, box_size.value_in(units.parsec), Nstart) | units.parsec
    stars.vx = rng.normal(0, 1, Nstart) | units.kms
    stars.vy = rng.normal(0, 1, Nstart) | units.kms
    stars.vz = rng.normal(0, 1, Nstart) | units.kms

    # Store NN distances if requested
    nn_distances = []

    while len(stars) < Nfinal:
        n_new = min(nbatch, Nfinal - len(stars))
        newstars = generate_stars(
            stars,
            n_new,
            box_size=box_size,
            pdf_func=pdf_func,
            pdf_unit=pdf_unit,
            min_separation=min_separation,
            seed=seed + len(stars),
        )

        if track_nn:
            tree = cKDTree(np.column_stack([
                stars.x.value_in(pdf_unit),
                stars.y.value_in(pdf_unit),
                stars.z.value_in(pdf_unit),
            ]))
            for i in range(len(newstars)):
                d, _ = tree.query([
                    newstars.x[i].value_in(pdf_unit),
                    newstars.y[i].value_in(pdf_unit),
                    newstars.z[i].value_in(pdf_unit),
                ], k=1)
                nn_distances.append(d)

        stars.add_particles(newstars)

    if track_nn:
        return stars, np.array(nn_distances)
    else:
        return stars

if __name__ == "__main__":
    test_generate_stars(beta=10)
