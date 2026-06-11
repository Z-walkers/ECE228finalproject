"""Stream one trajectory from a remote PDEBench HDF5 (DaRUS) and write the npy
that the ST-PINN training scripts expect, WITHOUT downloading the whole file.

We open the remote file via fsspec HTTP (range requests) so h5py only reads the
metadata + the single sample slice it needs.
"""
import argparse
import os
import time

import numpy as np
import h5py
import requests

DARUS = "https://darus.uni-stuttgart.de/api/access/datafile/{id}"


class HTTPRangeFile:
    """Minimal seekable, read-only file object backed by HTTP range requests.

    Lets h5py read only the bytes it needs from a remote HDF5 file. A 1 MiB
    block cache keeps the number of HTTP round-trips reasonable.
    """

    BLOCK = 1 << 20

    def __init__(self, url, session=None):
        self.url = url
        self.session = session or requests.Session()
        self.pos = 0
        self.cache = {}
        r = self.session.get(url, headers={"Range": "bytes=0-0"}, stream=True)
        r.raise_for_status()
        cr = r.headers.get("Content-Range")
        if cr:
            self.size = int(cr.split("/")[-1])
        else:
            self.size = int(r.headers.get("Content-Length", 0))
        r.close()

    def _block(self, idx):
        if idx not in self.cache:
            start = idx * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            last = None
            for attempt in range(8):
                try:
                    r = self.session.get(
                        self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60
                    )
                    r.raise_for_status()
                    self.cache[idx] = r.content
                    break
                except Exception as exc:  # transient network errors -> retry w/ backoff
                    last = exc
                    time.sleep(min(2 ** attempt, 20))
            else:
                raise last
        return self.cache[idx]

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, length=-1):
        if length is None or length < 0:
            length = self.size - self.pos
        end = min(self.pos + length, self.size)
        out = bytearray()
        p = self.pos
        while p < end:
            idx = p // self.BLOCK
            blk = self._block(idx)
            off = p - idx * self.BLOCK
            take = min(len(blk) - off, end - p)
            out += blk[off:off + take]
            p += take
        self.pos = end
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def close(self):
        self.session.close()

DEFAULT_RES = {
    "burgers": (1024, 256),
    "diffreact": (1024, 256),
    "diffsorb": (1024, 101),
}


def list_files(doi="doi:10.18419/darus-2986", contains=None):
    url = "https://darus.uni-stuttgart.de/api/datasets/:persistentId/"
    r = requests.get(url, params={"persistentId": doi}, timeout=120)
    r.raise_for_status()
    files = r.json()["data"]["latestVersion"]["files"]
    for f in files:
        df = f["dataFile"]
        name = df.get("filename", "")
        if contains and contains.lower() not in name.lower():
            continue
        print(f"{df['id']:>8}  {name}  ({df.get('filesize', 0)/1e6:.1f} MB)")


def _open_remote(file_id):
    url = DARUS.format(id=file_id)
    return h5py.File(HTTPRangeFile(url), "r")


def inspect(file_id):
    with _open_remote(file_id) as f:
        print(f"Top-level keys: {list(f.keys())}")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [dataset] {name:40s} shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"  [group]   {name}")

        f.visititems(visit)


def _load_sample(f, sample):
    # Membership tests (`in`) are single link lookups; avoid list(f.keys()) which
    # would force h5py to walk the entire group tree (slow over HTTP for files
    # with thousands of per-sample groups, e.g. diffusion-sorption).
    if "tensor" in f:  # flat layout (Burgers / ReacDiff)
        u = np.asarray(f["tensor"][sample])  # (Nt, Nx)
        x_grid = np.asarray(f["x-coordinate"]).reshape(-1)
        t_grid = np.asarray(f["t-coordinate"]).reshape(-1)
        return np.squeeze(u), x_grid, t_grid

    # grouped layout: sample groups named '0000', '0001', ... accessed directly
    g = None
    for name in (f"{sample:04d}", str(sample), f"{sample:05d}"):
        if name in f:
            g = f[name]
            break
    if g is None:
        raise SystemExit(
            "Could not find a 'tensor' dataset nor a sample group like '0000'. "
            "Run with --inspect on a local copy to check the layout."
        )
    u = np.squeeze(np.asarray(g["data"]))
    grid = f["grid"] if "grid" in f else g["grid"]
    x_grid = np.asarray(grid["x"]).reshape(-1)
    t_grid = np.asarray(grid["t"]).reshape(-1)
    return u, x_grid, t_grid


def _resize_axis(arr, axis, target, coord):
    n = arr.shape[axis]
    if target is None or target >= n:
        return arr, coord
    idx = np.linspace(0, n - 1, target).round().astype(int)
    return np.take(arr, idx, axis=axis), coord[idx]


def build(pde, file_id, out_path, sample, nx, nt, rescale_x):
    with _open_remote(file_id) as f:
        u, x_grid, t_grid = _load_sample(f, sample)

    # orient u to (Nt, Nx); x_grid reliably identifies the spatial axis
    nx0 = x_grid.size
    if u.shape[1] != nx0 and u.shape[0] == nx0:
        u = u.T
    # PDEBench sometimes stores one extra coordinate point; align coords to data
    t_grid = t_grid[:u.shape[0]]
    x_grid = x_grid[:u.shape[1]]
    if u.shape != (t_grid.size, x_grid.size):
        raise SystemExit(f"u shape {u.shape} != (Nt={t_grid.size}, Nx={x_grid.size})")

    u, t_grid = _resize_axis(u, 0, nt, t_grid)
    u, x_grid = _resize_axis(u, 1, nx, x_grid)
    if rescale_x:
        x_grid = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min())

    Nx, Nt = x_grid.size, t_grid.size
    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")
    U = u.T
    data = {
        "x": X.reshape(-1, 1).astype(np.float64),
        "t": T.reshape(-1, 1).astype(np.float64),
        "u": U.reshape(-1, 1).astype(np.float64),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, data)
    print(f"[{pde}] saved {out_path}: Nx x Nt = {Nx} x {Nt} ({Nx*Nt} pts)")
    print(f"  x in [{x_grid.min():.4g},{x_grid.max():.4g}], t in [{t_grid.min():.4g},{t_grid.max():.4g}]")
    print(f"  u in [{data['u'].min():.4g},{data['u'].max():.4g}]; "
          f"init(t==0)={(data['t'].reshape(-1)==0.0).sum()}, "
          f"bound(x==x0)={(data['x'].reshape(-1)==data['x'][0,0]).sum()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", help="list dataset files containing this substring, then exit")
    p.add_argument("--inspect", type=int, help="DaRUS file id to inspect, then exit")
    p.add_argument("--pde", choices=list(DEFAULT_RES.keys()))
    p.add_argument("--id", type=int, help="DaRUS file id")
    p.add_argument("--out")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--nx", type=int)
    p.add_argument("--nt", type=int)
    p.add_argument("--rescale-x", action="store_true")
    args = p.parse_args()

    if args.list is not None:
        list_files(contains=args.list)
        return
    if args.inspect is not None:
        inspect(args.inspect)
        return

    nx = args.nx if args.nx is not None else DEFAULT_RES[args.pde][0]
    nt = args.nt if args.nt is not None else DEFAULT_RES[args.pde][1]
    build(args.pde, args.id, args.out, args.sample, nx, nt, args.rescale_x)


if __name__ == "__main__":
    main()
