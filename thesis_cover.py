import os
import os.path
import sys
from glob import glob
from pathlib import Path
from typing import Optional
import random

import adaptive
import matplotlib
import matplotlib.cm
import matplotlib.colors as colors
import matplotlib.font_manager as fm
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import pyplot as plt


def get_cmap(cmap, min_clip=0.0, max_clip=1.0, exp=1.0):
    fcmap = getattr(plt.cm, cmap)
    return colors.LinearSegmentedColormap.from_list(
        "my_colormap", fcmap(np.linspace(min_clip, max_clip, 256) ** exp),
    )


def learner_till(till, learner, data):
    new_learner = adaptive.Learner2D(None, bounds=learner.bounds)
    new_learner.data = {k: v for k, v in data[:till]}
    for x, y in learner._bounds_points:
        # always include the bounds
        new_learner.tell((x, y), learner.data[x, y])
    return new_learner


def plot_tri(learner, ax, xy_size):
    ip = learner.ip()
    tri = ip.tri
    xs, ys = tri.points.T
    x_size, y_size = xy_size
    triang = mtri.Triangulation(x_size * xs, y_size * ys, triangles=tri.vertices)
    return ax.triplot(triang, c="k", lw=0.3, alpha=1, zorder=2), (ip.values, triang)


def to_gradient(data, horizontal, cmap, spread=20, mid=0.5):
    n, m = data.shape if horizontal else data.shape[::-1]
    x = np.linspace(1, 0, n)
    x = 1 / (np.exp((x - mid) * spread) + 1)  # Fermi-Dirac like
    gradient = x.reshape(1, -1).repeat(m, 0)
    if not horizontal:
        gradient = gradient.T
    gradient_rgb = cmap(data)
    gradient_rgb[:, :, -1] = gradient
    return gradient_rgb


def get_new_artists(npoints_tri, learner, data, ax, xy_size, npoints_interp, cmap):
    new_learner = learner_till(npoints_tri, learner, data)
    (line1, line2), (zs, triang) = plot_tri(new_learner, ax, xy_size)
    data = learner.interpolated_on_grid(npoints_interp)[
        -1
    ]  # This uses the original learner!
    x_size, y_size = xy_size
    im = ax.imshow(
        to_gradient(np.rot90(data), horizontal=False, cmap=cmap),
        extent=(-0.5 * x_size, 0.5 * x_size, -0.5 * y_size, 0.5 * y_size),
        zorder=3,
    )
    ax.tripcolor(triang, zs.flatten(), zorder=0, cmap=cmap)
    return im, line1, line2


def generate_cover(
    learner,
    save_fname: Optional[str] = "thesis-cover.pdf",
    with_lines=False,
    npoints_interp=1000,
    dpi=300,
    cmap=None,
    personal_text=None,
    edition=None,
    with_text=True,
):
    data = list(learner.data.items())

    # Measured from proefdruk
    x_total = 34.95  # cm total sides + back
    y_total = 24  # cm top to bottom

    inch_per_cm = 2.54
    margin = 0.5  # add 5 mm margin on each side

    x_size = (x_total + margin) / inch_per_cm
    y_size = (y_total + margin) / inch_per_cm
    xy_size = x_size, y_size

    spine_size = 1.1 / inch_per_cm

    fig, ax = plt.subplots(figsize=(x_size, y_size))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    ax.set_xticks([])
    ax.set_yticks([])

    cmap = cmap or get_cmap("inferno", 0.15, 0.95, 1.15)
    npoints_tri = len(data) // 4
    if len(data) > 4000:
        npoints_tri = max(npoints_tri, 4000)

    im, line1, line2 = get_new_artists(
        npoints_tri, learner, data, ax, xy_size, npoints_interp, cmap
    )

    title = "Towards realistic numerical simulations \n of Majorana devices"
    title2 = "Towards realistic numerical simulations of Majorana devices"
    author = "Bas Nijholt"

    text_color = "white"

    ax.axis("off")
    if with_text:
        font = "proxima_ssv/ProximaNova-Regular.otf"
        text_kwargs = dict(
            path_effects=[
                patheffects.withStroke(
                    linewidth=0.7, foreground="black", capstyle="round", alpha=1
                )
            ],
            zorder=4,
            verticalalignment="center",
            fontproperties=fm.FontProperties(fname=font),
        )
        for pos, text in zip([-0.8, 0.7], [author, title]):
            ax.text(
                x_size / 4,
                pos * (y_size - margin) / 2,
                text.upper(),
                color=text_color,
                weight="bold",
                **text_kwargs,
                horizontalalignment="center",
                fontsize=18,
            )

        ax.text(
            -0.09,
            y_size / 4 - 0.9,
            title2,
            color=text_color,
            weight="bold",
            rotation=-90,
            **text_kwargs,
            fontsize=12,
            horizontalalignment="left",
        )
        ax.text(
            -0.09,
            -y_size / 4 - 1,
            author,
            color=text_color,
            weight="bold",
            rotation=-90,
            **text_kwargs,
            fontsize=12,
            horizontalalignment="left",
        )

        lower_text_back = "Casimir PhD series 2020-11\nISBN 978-90-8593-438-7"
        if edition is not None:
            lower_text_back += f"\nedition {edition} of 120"
        ax.text(
            -x_size / 4,
            -0.8 * (y_size - margin) / 2,
            lower_text_back,
            color=text_color,
            weight="bold",
            horizontalalignment="center",
            **text_kwargs,
            fontsize=11,
        )
        if personal_text is not None:
            ax.text(
                -x_size / 4,
                0.4 * (y_size - margin) / 2,
                personal_text,
                color=text_color,
                weight="bold",
                horizontalalignment="center",
                **text_kwargs,
                fontsize=16,
            )

        if with_lines:
            for i in [-1, +1]:
                line_kwargs = dict(color="cyan", zorder=10, linestyles=":")
                ax.vlines(i * spine_size / 2, -y_size / 2, y_size / 2, **line_kwargs)
                ax.vlines(
                    -i * x_size / 2 + i * margin, -y_size / 2, y_size / 2, **line_kwargs
                )
                ax.hlines(
                    -i * y_size / 2 + i * margin, -x_size / 2, x_size / 2, **line_kwargs
                )

    ax.set_xlim(-x_size / 2, x_size / 2)
    ax.set_ylim(-y_size / 2, y_size / 2)
    print(f"Saving {save_fname}")
    if save_fname is not None:
        fig.savefig(
            save_fname,
            format=save_fname.suffix[1:],
            pad_inches=0,
            dpi=dpi,
        )
    plt.close(fig)


def bounds_from_saved_learner(fname):
    learner = adaptive.Learner2D(None, [(-1, 1), (-1, 1)])
    learner.load(fname)
    xs, ys = np.array(list(learner.data.keys())).T
    bounds = [(xs.min(), xs.max()), (ys.min(), ys.max())]
    return bounds


def load_learner(fname="data/mu-sweep2/data_learner_0246.pickle"):
    learner = adaptive.Learner2D(None, bounds_from_saved_learner(fname))
    learner.load(fname)
    return learner


def save(fname):
    print(f"Opening {fname}")
    f = fname.replace("/", "__")[:-7]
    pdf_fname = f"covers/{f}.pdf"
    print(pdf_fname)
    if os.path.exists(pdf_fname):
        print("exists, exit!")
        sys.exit(0)

    learner = load_learner(fname)
    generate_cover(learner, pdf_fname, with_lines=False, npoints_interp=2000)


def fname_out(folder, fname):
    fname_friendly = str(fname).replace("/", "__")
    return folder / f"{fname_friendly}.pdf"
