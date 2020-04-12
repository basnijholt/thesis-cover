import os
import os.path
import sys
from glob import glob
from typing import Optional

import matplotlib
import matplotlib.cm
import matplotlib.colors as colors
import matplotlib.font_manager as fm
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import pyplot as plt

import adaptive

mymap = colors.LinearSegmentedColormap.from_list(
    "my_colormap", plt.cm.inferno(np.linspace(0.0, 0.9, 256)**0.6)
)


class HistogramNormalize(colors.Normalize):
    def __init__(self, data, vmin=None, vmax=None):
        if vmin is not None:
            data = data[data > vmin]
        if vmax is not None:
            data = data[data < vmax]

        sorted_data = np.sort(data.flatten())
        self.sorted_data = sorted_data[np.isfinite(sorted_data)]
        colors.Normalize.__init__(self, vmin, vmax)

    def __call__(self, value, clip=None):
        return np.ma.masked_array(
            np.searchsorted(self.sorted_data, value) / len(self.sorted_data)
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


def to_gradient(data, horizontal, spread=20, mid=0.5):
    n, m = data.shape if horizontal else data.shape[::-1]
    x = np.linspace(1, 0, n)
    x = 1 / (np.exp((x - mid) * spread) + 1)  # Fermi-Dirac like
    gradient = x.reshape(1, -1).repeat(m, 0)
    if not horizontal:
        gradient = gradient.T
    gradient_rgb = mymap(data)
    gradient_rgb[:, :, -1] = gradient
    return gradient_rgb


def get_new_artists(npoints, learner, data, ax, xy_size):
    new_learner = learner_till(npoints, learner, data)
    (line1, line2), (zs, triang) = plot_tri(new_learner, ax, xy_size)

    data = learner.interpolated_on_grid(1000)[-1]  # This uses the original learner!
    x_size, y_size = xy_size
    normalizer = HistogramNormalize(data)
    im = ax.imshow(
        (to_gradient(np.rot90(data), horizontal=False)),
        extent=(-0.5 * x_size, 0.5 * x_size, -0.5 * y_size, 0.5 * y_size),
        zorder=3,
#         norm=normalizer,
    )

    ax.tripcolor(triang, zs.flatten(), zorder=0, cmap=mymap, norm=normalizer)
    return im, line1, line2


def generate_cover(
    learner, save_fname: Optional[str] = "thesis-cover.pdf", with_lines=False
):
    data = list(learner.data.items())

    # measured from the guides in Tomas's thesis: `thesis_cover.pdf`
    x_right = 14.335
    x_left = 0.591
    y_top = 0.591
    y_bottom = 10.039

    inch_per_cm = 2.54
    margin = 0.5 / inch_per_cm  # add 5 mm margin on each side

    x_size = x_right - x_left + 2 * margin
    y_size = y_bottom - y_top + 2 * margin
    xy_size = x_size, y_size

    spine_size = 0.8 / inch_per_cm

    fig, ax = plt.subplots(figsize=(x_size, y_size))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    ax.set_xticks([])
    ax.set_yticks([])

    im, line1, line2 = get_new_artists(len(data) // 5, learner, data, ax, xy_size)

    title = "Towards realistic numerical simulations \n of Majorana devices"
    title2 = "Towards realistic numerical simulations of Majorana devices"
    author = "Bas Nijholt"

    text_color = "white"

    ax.axis("off")
    #     font = "HKNova-Medium.ttf"  # '/System/Library/Fonts/SFNS.ttf'
    #     font = "/Users/basnijholt/Downloads/bell-gothic-std/BellGothicStd-Light.otf"
    font = "/Users/basnijholt/Downloads/proxima_ssv/ProximaNova-Regular.otf"
    text_kwargs = dict(
        path_effects=[
            patheffects.withStroke(
                linewidth=1, foreground="black", capstyle="round", alpha=0.7
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

    ax.text(
        -x_size / 4,
        -0.9 * (y_size - margin) / 2,
        "Casimir PhD series 2020-11\nISBN 978-90-8593-438-7",
        fontsize=12,
        color=text_color,
        weight="bold",
        horizontalalignment="center",
        **text_kwargs,
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
    ext = save_fname.split(".")[-1]
    if save_fname is not None:
        fig.savefig(
            save_fname, format=ext, bbox_inches="tight", pad_inches=0.001, dpi=500
        )


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
    generate_cover(learner, pdf_fname, with_lines=False)


if __name__ == "__main__":
    chosen = [
        "data__gradient-sweep-alpha2__data_learner_0587",
        "data__gradient-sweep-angle-0-45__data_learner_0334",
        "data__gradient-sweep-alpha__data_learner_0027",
        "data__gradient-sweep-alpha__data_learner_0190",
        "data__gradient-sweep-alpha__data_learner_0184",
        "data__mu-sweep__data_learner_0051",
        "data__gradient-sweep-angle-0-45__data_learner_0057",
        "data__gradient-sweep-alpha2__data_learner_0584",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0233",
        "data__gradient-sweep-alpha__data_learner_0030",
        "data__gradient-sweep-alpha__data_learner_0187",
        "data__gradient-sweep-angle-0-45__data_learner_0069",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0811",
        "data__gradient-sweep-rotation-0-45__data_learner_0016",
        "data__mu-sweep2__data_learner_0199",
        "data__gradient-sweep-angle-0-45__data_learner_0068",
        "data__gradient-sweep-angle-0-45__data_learner_0332",
        "data__mu-sweep2__data_learner_0200",
        "data__gradient-sweep-alpha2__data_learner_0379",
        "data__mu-sweep2__data_learner_0201",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0035",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0021",
        "data__gradient-sweep-angle-0-45__data_learner_0325",
        "data__gradient-sweep-alpha__data_learner_0220",
        "data__gradient-sweep-alpha2__data_learner_0178",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0723",
        "data__gradient-sweep-alpha2__data_learner_0080",
        "data__mu-sweep2__data_learner_0259",
        "data__gradient-sweep-alpha__data_learner_0118",
        "data__mu-sweep2__data_learner_0113",
        "data__gradient-sweep-alpha__data_learner_0119",
        "data__mu-sweep2__data_learner_0258",
        "data__gradient-sweep-alpha2__data_learner_0444",
        "data__gradient-sweep-alpha__data_learner_0086",
        "data__gradient-sweep-alpha__data_learner_0284",
        "data__gradient-sweep-alpha__data_learner_0286",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0052",
        "data__gradient-sweep-alpha2__data_learner_0083",
        "data__gradient-sweep-alpha__data_learner_0087",
        "data__mu-sweep2__data_learner_0111",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0457",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0709",
        "data__gradient-sweep-angle-0-45__data_learner_0345",
        "data__mu-sweep2__data_learner_0049",
        "data__gradient-sweep-alpha__data_learner_0254",
        "data__gradient-sweep-alpha2__data_learner_0092",
        "data__mu-sweep__data_learner_0183",
        "data__gradient-sweep-alpha2__data_learner_0079",
        "data__mu-sweep2__data_learner_0115",
        "data__mu-sweep2__data_learner_0101",
        "data__mu-sweep2__data_learner_0114",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0693",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0485",
        "data__gradient-sweep-alpha__data_learner_0094",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0726",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0685",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0691",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0684",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0492",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0479",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0727",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0055",
        "data__gradient-sweep-alpha__data_learner_0059",
        "data__gradient-sweep-alpha2__data_learner_0076",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0879",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0461",
        "data__mu-sweep2__data_learner_0245",
        "data__mu-sweep2__data_learner_0047",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0703",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0715",
        "data__gradient-sweep-rotation-0-45__data_learner_0082",
        "data__gradient-sweep-rotation-0-45__data_learner_0083",
        "data__gradient-sweep-alpha2__data_learner_0921",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0710",
        "data__gradient-sweep-alpha2__data_learner_0272",
        "data__gradient-sweep-alpha__data_learner_0088",
        "data__mu-sweep2__data_learner_0256",
        "data__gradient-sweep-alpha__data_learner_0116",
        "data__mu-sweep2__data_learner_0257",
        "data__gradient-sweep-alpha__data_learner_0062",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0711",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0705",
        "data__gradient-sweep-rotation-0-45-move-SO-and-potential__data_learner_0013",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0713",
        "data__gradient-sweep-alpha__data_learner_0276",
        "data__gradient-sweep-alpha2__data_learner_0073",
        "data__gradient-sweep-alpha__data_learner_0060",
        "data__mu-sweep2__data_learner_0255",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0459",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0471",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0699",
        "data__gradient-sweep-alpha__data_learner_0061",
        "data__gradient-sweep-angle-0-45__data_learner_0372",
        "data__mu-sweep2__data_learner_0233",
        "data__gradient-sweep-alpha2__data_learner_0175",
        "data__mu-sweep2__data_learner_0192",
        "data__mu-sweep2__data_learner_0179",
        "data__gradient-sweep-angle-0-45__data_learner_0076",
        "data__mu-sweep2__data_learner_0232",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0748",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0006",
        "data__gradient-sweep-alpha2__data_learner_0566",
        "data__gradient-sweep-alpha2__data_learner_0823",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0830",
        "data__gradient-sweep-angle-0-45__data_learner_0049",
        "data__gradient-sweep-alpha2__data_learner_0188",
        "data__mu-sweep2__data_learner_0037",
        "data__gradient-sweep-alpha__data_learner_0028",
        "data__mu-sweep2__data_learner_0235",
        "data__mu-sweep2__data_learner_0180",
        "data__gradient-sweep-alpha2__data_learner_0370",
        "data__mu-sweep2__data_learner_0181",
        "data__gradient-sweep-rotation-0-45-move-SO-and-potential__data_learner_0489",
        "data__gradient-sweep-alpha__data_learner_0029",
        "data__mu-sweep2__data_learner_0036",
        "data__mu-sweep2__data_learner_0034",
        "data__gradient-sweep-alpha__data_learner_0215",
        "data__gradient-sweep-rotation-0-90-move-SO-slowly__data_learner_0228",
        "data__mu-sweep2__data_learner_0236",
        "data__mu-sweep2__data_learner_0222",
        "data__gradient-sweep-alpha__data_learner_0188",
        "data__mu-sweep2__data_learner_0035",
    ]

    def f_pickle(fname):
        return fname.replace("__", "/") + ".pickle"

    fnames = glob("data/*/*pickle")
    i = int(sys.argv[1])
    fname = fnames[i]
    save(fname)
