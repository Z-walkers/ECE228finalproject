"""Regenerate the input/*.npy datasets from the original PDEBench HDF5 files
at full (paper) resolution.

The training scripts use *every* grid point of the loaded npy as the sample /
collocation set, so the spatiotemporal resolution is decided entirely here.
Paper resolutions (Yan et al., ST-PINN, IJCNN 2023):

    burgers    : Nx x Nt = 1024 x 256,  domain [0,1] x [0,2]
    diffreact  : Nx x Nt = 1024 x 256,  domain [0,1] x [0,1]
    diffsorb   : Nx x Nt = 1024 x 101,  domain [0,1] x (0,500]

The npy is a dict {'x','t','u'} where each value has shape (Nx*Nt, 1) and the
flattening is x-major / t-minor, i.e. point index = i*Nt + j for (x_i, t_j).
That layout is required by the training scripts, which rely on:
    np.where(t == 0.0)      -> the initial condition (one row per x)
    np.where(x == x[0, 0])  -> all time steps at the first spatial node

PDEBench stores 1D data in one of two HDF5 layouts; this script handles both:
  (A) flat   : top-level datasets 'tensor' (S, Nt, Nx), 'x-coordinate', 't-coordinate'
  (B) grouped: per-sample groups '0000'.. each with 'data' (Nt, Nx[, 1]) and a
               'grid' group holding 'x' and 't'

Usage:
    # 1) inspect the file structure first (recommended)
    python source/build_dataset.py --h5 /path/to/file.h5 --inspect

    # 2) build the npy (sample 0 by default)
    python source/build_dataset.py --pde burgers   --h5 1D_Burgers_Sols_Nu0.01.hdf5 --out ./input/burgers1D.npy
    python source/build_dataset.py --pde diffreact --h5 1D_ReacDiff...h5            --out ./input/diffreact1D.npy
    python source/build_dataset.py --pde diffsorb  --h5 1D_diff-sorp_NA_NA.h5       --out ./input/diffsorb1D.npy
"""
import argparse
import os

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "h5py is required: pip install h5py  (or: uv pip install h5py)"
    ) from exc


def inspect(path):
    """Print the full structure (groups, datasets, shapes) of an HDF5 file."""
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  [dataset] {name:40s} shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"  [group]   {name}")

    with h5py.File(path, "r") as f:
        print(f"Top-level keys: {list(f.keys())}")
        f.visititems(visit)


def _load_sample(path, sample):
    """Return (u, x_grid, t_grid) for one sample.

    u has shape (Nt, Nx); x_grid/t_grid are 1D coordinate arrays.
    """
    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        # Layout (A): flat 'tensor' format -------------------------------------
        if "tensor" in keys:
            tensor = f["tensor"]
            u = np.asarray(tensor[sample])  # (Nt, Nx)
            x_grid = np.asarray(f["x-coordinate"]).reshape(-1)
            t_grid = np.asarray(f["t-coordinate"]).reshape(-1)
            return u, x_grid, t_grid

        # Layout (B): grouped per-sample format --------------------------------
        sample_keys = sorted(k for k in keys if k not in ("grid",) and isinstance(f[k], h5py.Group))
        if not sample_keys:
            raise SystemExit(
                f"Could not recognise the HDF5 layout. Top-level keys = {keys}.\n"
                "Run with --inspect and share the output so the loader can be adjusted."
            )
        g = f[sample_keys[sample]]
        u = np.asarray(g["data"])           # (Nt, Nx) or (Nt, Nx, 1)
        u = np.squeeze(u)
        # grid may live in a shared top-level 'grid' group or inside the sample
        grid = f["grid"] if "grid" in keys else g["grid"]
        x_grid = np.asarray(grid["x"]).reshape(-1)
        t_grid = np.asarray(grid["t"]).reshape(-1)
        return u, x_grid, t_grid


def _resize_axis(arr, axis, target, coord):
    """Subsample (stride) a tensor + its coordinate to `target` points along axis.

    Only down-sampling is supported (you cannot create resolution that is not in
    the source file). Picks `target` evenly-spaced indices including both ends.
    """
    n = arr.shape[axis]
    if target is None or target >= n:
        return arr, coord
    idx = np.linspace(0, n - 1, target).round().astype(int)
    return np.take(arr, idx, axis=axis), coord[idx]


def build(pde, h5_path, out_path, sample, nx, nt, rescale_x):
    u, x_grid, t_grid = _load_sample(h5_path, sample)  # u: (Nt, Nx)

    if u.shape != (t_grid.size, x_grid.size):
        # some files store (Nx, Nt); transpose to (Nt, Nx)
        if u.shape == (x_grid.size, t_grid.size):
            u = u.T
        else:
            raise SystemExit(
                f"u shape {u.shape} does not match (Nt={t_grid.size}, Nx={x_grid.size})."
            )

    # optional downsample to the requested resolution
    u, t_grid = _resize_axis(u, 0, nt, t_grid)
    u, x_grid = _resize_axis(u, 1, nx, x_grid)

    if rescale_x:  # map x onto [0,1] (training code assumes periodic BC at 0 and 1)
        x_grid = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min())

    Nx, Nt = x_grid.size, t_grid.size
    # x-major / t-minor flatten via meshgrid(indexing='ij')
    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")  # (Nx, Nt)
    U = u.T                                             # (Nx, Nt)

    data = {
        "x": X.reshape(-1, 1).astype(np.float64),
        "t": T.reshape(-1, 1).astype(np.float64),
        "u": U.reshape(-1, 1).astype(np.float64),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, data)

    print(f"[{pde}] saved {out_path}")
    print(f"  Nx x Nt = {Nx} x {Nt}  (total {Nx * Nt} points)")
    print(f"  x in [{x_grid.min():.4g}, {x_grid.max():.4g}], "
          f"t in [{t_grid.min():.4g}, {t_grid.max():.4g}]")
    print(f"  u in [{data['u'].min():.4g}, {data['u'].max():.4g}]")
    print(f"  init points (t==0): {(data['t'].reshape(-1) == 0.0).sum()}; "
          f"boundary points (x==x[0]): {(data['x'].reshape(-1) == data['x'][0, 0]).sum()}")


DEFAULT_RES = {
    "burgers":   (1024, 256),
    "diffreact": (1024, 256),
    "diffsorb":  (1024, 101),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, help="path to the PDEBench HDF5 file")
    p.add_argument("--pde", choices=list(DEFAULT_RES.keys()), help="which PDE (sets default resolution)")
    p.add_argument("--out", help="output npy path (e.g. ./input/burgers1D.npy)")
    p.add_argument("--sample", type=int, default=0, help="which trajectory index to extract")
    p.add_argument("--nx", type=int, default=None, help="override target Nx (downsample only)")
    p.add_argument("--nt", type=int, default=None, help="override target Nt (downsample only)")
    p.add_argument("--rescale-x", action="store_true", help="rescale x onto [0,1]")
    p.add_argument("--inspect", action="store_true", help="print HDF5 structure and exit")
    args = p.parse_args()

    if args.inspect:
        inspect(args.h5)
        return

    if not args.pde or not args.out:
        raise SystemExit("--pde and --out are required unless --inspect is used.")

    nx = args.nx if args.nx is not None else DEFAULT_RES[args.pde][0]
    nt = args.nt if args.nt is not None else DEFAULT_RES[args.pde][1]
    build(args.pde, args.h5, args.out, args.sample, nx, nt, args.rescale_x)


if __name__ == "__main__":
    main()
