import argparse
import os
import sys

import numpy as np
import pyvista as pv


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
    try:
        import tetgen
    except Exception as exc:
        raise RuntimeError("tetgen is required for surface tetrahedralization.") from exc
    tg = tetgen.TetGen(surface)
    kwargs = {"order": 1, "mindihedral": mindihedral, "minratio": minratio}
    if max_volume is not None:
        kwargs["maxvolume"] = max_volume
    tg.tetrahedralize(**kwargs)
    return tg.grid


def voxel_tet_mesh(volume, origin, pitch):
    nx, ny, nz = volume.shape
    gx = np.arange(nx + 1) * pitch + origin[0]
    gy = np.arange(ny + 1) * pitch + origin[1]
    gz = np.arange(nz + 1) * pitch + origin[2]
    grid_x, grid_y, grid_z = np.meshgrid(gx, gy, gz, indexing="ij")
    nodes = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    def vid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    filled = np.argwhere(volume > 0)
    tets = []
    labels = []
    for i, j, k in filled:
        v000 = vid(i, j, k)
        v100 = vid(i + 1, j, k)
        v010 = vid(i, j + 1, k)
        v110 = vid(i + 1, j + 1, k)
        v001 = vid(i, j, k + 1)
        v101 = vid(i + 1, j, k + 1)
        v011 = vid(i, j + 1, k + 1)
        v111 = vid(i + 1, j + 1, k + 1)
        # 6-tet split along body diagonal (v000-v111)
        tets.extend(
            [
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            ]
        )
        labels.extend([int(volume[i, j, k])] * 6)

    tets = np.array(tets, dtype=np.int64)
    labels = np.array(labels, dtype=np.uint16)
    return nodes, tets, labels


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
    parser.add_argument("--voxel-tets", action="store_true", help="Generate tets directly from voxels.")
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"Missing volume: {args.npz}")
        return 1

    volume, origin, pitch = load_volume(args.npz)
    grid = make_grid(volume, origin, pitch)
    if args.voxel_tets:
        nodes, tets, labels = voxel_tet_mesh(volume, origin, pitch)
        cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64), tets]).reshape(-1)
        celltypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
        grid_tet = pv.UnstructuredGrid(cells, celltypes, nodes)
        grid_tet.cell_data["label"] = labels
    else:
        surface = volume_surface(grid)

        if args.decimate > 0:
            try:
                surface = surface.decimate_pro(target_reduction=args.decimate)
            except TypeError:
                surface = surface.decimate_pro(reduction=args.decimate)
        if args.smooth > 0:
            surface = surface.smooth(n_iter=args.smooth)

        try:
            grid_tet = tetrahedralize(surface, max_volume=args.max_volume)
        except Exception as exc:
            print(f"Tetgen failed ({exc}); falling back to voxel tets.")
            nodes, tets, labels = voxel_tet_mesh(volume, origin, pitch)
            cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64), tets]).reshape(-1)
            celltypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
            grid_tet = pv.UnstructuredGrid(cells, celltypes, nodes)
            grid_tet.cell_data["label"] = labels
        else:
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
