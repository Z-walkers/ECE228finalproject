import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


METHODS = {
    "pinn": {"label": "PINN", "color": "red", "linestyle": "-.", "marker": "+"},
    "stpinn": {"label": "ST-PINN", "color": "blue", "linestyle": "--", "marker": "o"},
    "ffu-stpinn": {
        "label": "FFUSTPINN",
        "color": "purple",
        "linestyle": "-",
        "marker": "d",
    },
    "dynamic-ffu-stpinn": {"label": "Dynamic FFUSTPINN", "color": "green", "linestyle": ":", "marker": "s"},
}


DEFAULT_MIN_ITER = 10000
PAPER_ORDER = ["stpinn", "pinn", "ffu-stpinn", "dynamic-ffu-stpinn"]


TASK_PREFIX = {
    "burgers": "burgers1d",
    "diffreact": "diffreact1d",
    "diffsorb": "diffsorb1D",
}


INPUT_PATH = {
    "burgers": "input/burgers1D.npy",
    "diffreact": "input/diffreact1D.npy",
    "diffsorb": "input/diffsorb1D.npy",
}


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ordered_methods(methods):
    return [method for method in PAPER_ORDER if method in methods]


def load_field(path):
    data = np.load(path, allow_pickle=True).item()
    return data["x"].reshape(-1), data["t"].reshape(-1), data["u"].reshape(-1)


def load_prediction(path):
    return np.load(path, allow_pickle=True).item()["u"].reshape(-1)


def timestamp_from_result(path, method):
    name = os.path.basename(path)
    marker = f"-{method}-"
    if marker not in name:
        return None
    return os.path.splitext(name.split(marker, 1)[1])[0]


def result_prefixes(task):
    # File names on disk are inconsistently cased (e.g. diffreact1D vs diffreact1d).
    # Linux is case-sensitive, so try both the '...1d' and '...1D' variants.
    prefix = TASK_PREFIX[task]
    prefixes = []
    for candidate in (prefix, prefix[:-1] + "d", prefix[:-1] + "D"):
        if candidate not in prefixes:
            prefixes.append(candidate)
    return prefixes


def matching_log(task, method, prediction_path):
    stamp = timestamp_from_result(prediction_path, method)
    if not stamp:
        return None
    for prefix in result_prefixes(task):
        path = os.path.join("output", "log", f"{prefix}-{method}-{stamp}.txt")
        if os.path.exists(path):
            return path
        matches = glob.glob(os.path.join("output", "log", f"{prefix}-{method}-{stamp}*"))
        if matches:
            return max(matches, key=os.path.getmtime)
    return None


def max_logged_iteration(path):
    if not path or not os.path.exists(path):
        return -1
    max_it = -1
    pattern = re.compile(r"\bIt:?\s*(\d+)")
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                max_it = max(max_it, int(match.group(1)))
    return max_it


def candidate_predictions(task, method):
    files = []
    for prefix in result_prefixes(task):
        pattern = os.path.join("output", "prediction", f"{prefix}-{method}-*.npy")
        files.extend(glob.glob(pattern))
    files = list(dict.fromkeys(files))
    return sorted(files, key=os.path.getmtime, reverse=True)


def latest_prediction(task, method, min_iter=DEFAULT_MIN_ITER, allow_incomplete=False):
    files = candidate_predictions(task, method)
    if not files:
        return None

    scored = []
    for path in files:
        log_path = matching_log(task, method, path)
        max_it = max_logged_iteration(log_path)
        scored.append((path, log_path, max_it))
        if allow_incomplete or max_it >= min_iter:
            return path

    fallback = scored[0]
    print(
        "[warn] no complete prediction found for "
        f"{task}/{method} with min_iter={min_iter}; using latest "
        f"{os.path.basename(fallback[0])}, max_logged_it={fallback[2]}."
    )
    return fallback[0]


def latest_log(task, method, min_iter=DEFAULT_MIN_ITER, allow_incomplete=False):
    files = []
    for prefix in result_prefixes(task):
        files.extend(glob.glob(os.path.join("output", "log", f"{prefix}-{method}-*")))
    files = list(dict.fromkeys(files))
    if not files:
        return None

    files = sorted(files, key=os.path.getmtime, reverse=True)
    for path in files:
        if allow_incomplete or max_logged_iteration(path) >= min_iter:
            return path

    fallback = files[0]
    print(
        "[warn] no complete log found for "
        f"{task}/{method} with min_iter={min_iter}; using latest "
        f"{os.path.basename(fallback)}, max_logged_it={max_logged_iteration(fallback)}."
    )
    return fallback


def gridify(x, t, values):
    x_vals = np.unique(x)
    t_vals = np.unique(t)
    order = np.lexsort((x, t))
    grid = values[order].reshape(len(t_vals), len(x_vals)).T
    return x_vals, t_vals, grid


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def plot_burgers(methods, out_dir, min_iter, allow_incomplete):
    x, t, u = load_field(INPUT_PATH["burgers"])
    x_vals, t_vals, u_true = gridify(x, t, u)
    predictions = {}

    for method in ordered_methods(methods):
        path = latest_prediction("burgers", method, min_iter, allow_incomplete)
        if path:
            predictions[method] = gridify(x, t, load_prediction(path))[2]
        else:
            print(f"[warn] missing prediction for burgers/{method}; skipping this curve.")

    times = [0.3, 0.6, 0.9]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.7), constrained_layout=True)
    for ax, target_t in zip(axes, times):
        j = nearest_index(t_vals, target_t)
        ax.plot(x_vals, u_true[:, j], color="0.7", linewidth=4.5, label="Solver")
        for method, pred in predictions.items():
            style = METHODS[method]
            ax.plot(
                x_vals,
                pred[:, j],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(1, len(x_vals) // 22),
                markersize=3.5,
                linewidth=1.6,
            )
        ax.set_title(f"Time={t_vals[j]:.1f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(x)$")
        ax.set_xlim(x_vals.min(), x_vals.max())
        ax.grid(False)
    axes[-1].legend()

    path = os.path.join(out_dir, "Burgers1D.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_loss_log(path):
    points = []
    if not path or not os.path.exists(path):
        return points
    pattern = re.compile(r"It:?\s*(\d+).*?(?:Loss[:=]\s*|Total=)([0-9.eE+-]+)")
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                points.append((int(match.group(1)), float(match.group(2))))
    return points


def plot_diffreact_loss(methods, out_dir, min_iter, allow_incomplete):
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plotted = False

    for method in ordered_methods(methods):
        points = parse_loss_log(latest_log("diffreact", method, min_iter, allow_incomplete))
        if not points:
            continue
        style = METHODS[method]
        steps, losses = zip(*points)
        ax.semilogy(
            steps,
            losses,
            label=style["label"],
            color=style["color"],
            linestyle="-" if method in {"pinn", "stpinn"} else style["linestyle"],
            linewidth=1.6,
        )
        plotted = True

    ax.set_xlabel("Iteration times")
    ax.set_ylabel("Loss")
    ax.set_xlim(left=0)
    ax.grid(False)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if value == 0 else f"{int(value / 1000)}k")
    )
    if plotted:
        ax.legend()
    else:
        print("[warn] no diffreact loss logs found; generated an empty loss axes.")

    path = os.path.join(out_dir, "Diff-React-1D.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def add_field_image(fig, ax, field, extent, title, vmin=0.0, vmax=1.0):
    im = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, pad=4)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_diffsorb_fields(methods, out_dir, min_iter, allow_incomplete):
    x, t, u = load_field(INPUT_PATH["diffsorb"])
    x_vals, t_vals, u_true = gridify(x, t, u)
    selected_methods = []
    prediction_paths = {}

    for method in ordered_methods(methods):
        path = latest_prediction("diffsorb", method, min_iter, allow_incomplete)
        if path:
            selected_methods.append(method)
            prediction_paths[method] = path
        else:
            print(f"[warn] missing prediction for diffsorb/{method}; skipping this field.")

    predictions = {
        method: gridify(x, t, load_prediction(prediction_paths[method]))[2]
        for method in selected_methods
    }

    ncols = max(1, len(selected_methods))
    fig = plt.figure(figsize=(4.4 * ncols, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(3, ncols)

    t_plot = (t_vals - t_vals.min()) / (t_vals.max() - t_vals.min())
    extent = [t_plot.min(), t_plot.max(), x_vals.min(), x_vals.max()]
    field_vmax = max(1.0, float(np.nanmax(u_true)))

    solver_col = ncols // 2
    solver_ax = fig.add_subplot(grid[0, solver_col])
    add_field_image(fig, solver_ax, u_true, extent, r"$u$(Solver)", vmax=field_vmax)
    for col in range(ncols):
        if col != solver_col:
            blank_ax = fig.add_subplot(grid[0, col])
            blank_ax.axis("off")

    for col, method in enumerate(selected_methods):
        style = METHODS[method]
        pred = predictions[method]
        pred_ax = fig.add_subplot(grid[1, col])
        add_field_image(fig, pred_ax, pred, extent, rf"$u$({style['label']})", vmax=field_vmax)

        err_ax = fig.add_subplot(grid[2, col])
        err = np.abs(pred - u_true)
        add_field_image(
            fig,
            err_ax,
            err,
            extent,
            f"Point-wise Error ({style['label']})",
            vmin=0.0,
            vmax=0.02,
        )

    path = os.path.join(out_dir, "Diff-Sorb-1D.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot paper-style PINN/ST-PINN figures.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pinn", "stpinn", "ffu-stpinn"],
        choices=list(METHODS.keys()),
    )
    parser.add_argument("--out-dir", default="output/figures")
    parser.add_argument(
        "--min-iter",
        type=int,
        default=DEFAULT_MIN_ITER,
        help="Prefer results whose matching log reached at least this iteration.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Use the latest files even if their logs look like smoke/incomplete runs.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    ensure_dir(args.out_dir)
    paths = [
        plot_burgers(args.methods, args.out_dir, args.min_iter, args.allow_incomplete),
        plot_diffreact_loss(args.methods, args.out_dir, args.min_iter, args.allow_incomplete),
        plot_diffsorb_fields(args.methods, args.out_dir, args.min_iter, args.allow_incomplete),
    ]
    print("Generated figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
