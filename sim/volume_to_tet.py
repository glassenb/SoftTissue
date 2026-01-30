import argparse
import os
import sys

import numpy as np
import pyvista as pv
import tetgen


def load_volume(npz_path):
    data = np.load(npz_path)
    volume = data["volume"]
    origin = data["origin"].astype(float)
    pitch = float(data["pitch"])
    return volume, origin, pitch


def make_grid(volume, origin, pitch):
    dims = np.array(volume.shape, dtype=int) + 1
    grid = pv.ImageData(dimensions=dims, spacing=(pitch, pitch, pitch), origin=origin)
    grid.cell_data["label"] = volume.flatten(order="F")
    return grid


def volume_surface(grid):
    filled = grid.threshold(value=0.5, scalars="label")
    surface = filled.extract_surface().triangulate().clean()
    return surface


def tetrahedralize(surface, max_volume=None, mindihedral=10, minratio=1.5):
    tg = tetgen.TetGen(surface)
    kwargs = {"order": 1, "mindihedral": mindihedral, "minratio": minratio}
    if max_volume is not None:
        kwargs["maxvolume"] = max_volume
    tg.tetrahedralize(**kwargs)
    return tg.grid


def label_tets(nodes, tets, volume, origin, pitch):
    centers = nodes[tets].mean(axis=1)
    idx = np.floor((centers - origin) / pitch).astype(int)
    valid = (
        (idx[:, 0] >= 0)
        & (idx[:, 1] >= 0)
        & (idx[:, 2] >= 0)
        & (idx[:, 0] < volume.shape[0])
        & (idx[:, 1] < volume.shape[1])
        & (idx[:, 2] < volume.shape[2])
    )
    labels = np.zeros(len(tets), dtype=np.uint16)
    labels[valid] = volume[idx[valid, 0], idx[valid, 1], idx[valid, 2]]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate a tet mesh from a labeled voxel volume.")
    parser.add_argument("--npz", required=True, help="Input volume NPZ.")
    parser.add_argument("--out", required=True, help="Output tet mesh VTK.")
    parser.add_argument("--npz-out", help="Optional tet mesh NPZ (nodes, tets, labels).")
    parser.add_argument("--max-volume", type=float, default=None, help="Tet max volume.")
    parser.add_argument("--decimate", type=float, default=0.0, help="Surface decimation 0-1.")
    parser.add_argument("--smooth", type=int, default=0, help="Surface smoothing iterations.")
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"Missing volume: {args.npz}")
        return 1

    volume, origin, pitch = load_volume(args.npz)
    grid = make_grid(volume, origin, pitch)
    surface = volume_surface(grid)

    if args.decimate > 0:
        try:
            surface = surface.decimate_pro(target_reduction=args.decimate)
        except TypeError:
            surface = surface.decimate_pro(reduction=args.decimate)
    if args.smooth > 0:
        surface = surface.smooth(n_iter=args.smooth)

    grid_tet = tetrahedralize(surface, max_volume=args.max_volume)

    cells = grid_tet.cells.reshape(-1, 5)
    tets = cells[:, 1:5].astype(int)
    nodes = grid_tet.points.astype(float)

    labels = label_tets(nodes, tets, volume, origin, pitch)
    grid_tet.cell_data["label"] = labels

    grid_tet.save(args.out)
    print(f"Saved tet mesh: {args.out}")

    if args.npz_out:
        np.savez_compressed(args.npz_out, nodes=nodes, tets=tets, labels=labels, origin=origin, pitch=pitch)
        print(f"Saved tet NPZ: {args.npz_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
