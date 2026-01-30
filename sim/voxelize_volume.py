import argparse
import json
import os
import sys

import numpy as np
import trimesh as tm


def load_mesh(path, keep_largest=True):
    mesh = tm.load(path, force="mesh")
    if isinstance(mesh, tm.Scene):
        mesh = tm.util.concatenate(mesh.dump())

    # Clean using non-deprecated trimesh APIs.
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass

    if keep_largest:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            parts = sorted(parts, key=lambda p: p.volume if p.volume > 0 else p.area, reverse=True)
            print(f"{os.path.basename(path)}: {len(parts)} components, keeping largest")
            mesh = parts[0]

    return mesh


def voxelize_mesh(mesh, pitch, fill=True):
    vg = mesh.voxelized(pitch)
    if fill:
        try:
            vg = vg.fill()
        except Exception:
            pass
    return vg


def voxel_origin(vg):
    if hasattr(vg, "translation") and vg.translation is not None:
        return np.asarray(vg.translation, dtype=float)
    if hasattr(vg, "transform") and vg.transform is not None:
        return np.asarray(vg.transform[:3, 3], dtype=float)
    raise AttributeError("VoxelGrid has no translation/transform for origin.")


def paste_voxels(volume, vg, label, origin, pitch, resolve="overwrite"):
    # Compute offset from global origin to voxel grid origin.
    vg_origin = voxel_origin(vg)
    offset = np.rint((vg_origin - origin) / pitch).astype(int)

    sx, sy, sz = vg.matrix.shape
    x0, y0, z0 = offset
    x1, y1, z1 = x0 + sx, y0 + sy, z0 + sz

    gx0 = max(0, x0)
    gy0 = max(0, y0)
    gz0 = max(0, z0)
    gx1 = min(volume.shape[0], x1)
    gy1 = min(volume.shape[1], y1)
    gz1 = min(volume.shape[2], z1)

    if gx0 >= gx1 or gy0 >= gy1 or gz0 >= gz1:
        return 0

    lx0 = gx0 - x0
    ly0 = gy0 - y0
    lz0 = gz0 - z0
    lx1 = lx0 + (gx1 - gx0)
    ly1 = ly0 + (gy1 - gy0)
    lz1 = lz0 + (gz1 - gz0)

    mask = vg.matrix[lx0:lx1, ly0:ly1, lz0:lz1]
    sub = volume[gx0:gx1, gy0:gy1, gz0:gz1]

    if resolve == "overwrite":
        sub[mask] = label
    else:
        sub[np.logical_and(mask, sub == 0)] = label

    volume[gx0:gx1, gy0:gy1, gz0:gz1] = sub
    return int(mask.sum())


def parse_pairs(stls, labels):
    if stls is None or labels is None or len(stls) != len(labels):
        raise ValueError("Provide equal number of --stl and --label arguments.")
    return list(zip(labels, stls))


def main():
    parser = argparse.ArgumentParser(description="Voxelize STL meshes into a labeled volume.")
    parser.add_argument("--stl", action="append", help="STL path (repeatable).")
    parser.add_argument("--label", action="append", type=int, help="Label for each STL (repeatable).")
    parser.add_argument("--pitch", type=float, default=2.0, help="Voxel spacing in world units.")
    parser.add_argument("--padding", type=float, default=0.0, help="Padding (world units).")
    parser.add_argument("--fill", action="store_true", help="Fill interior voxels.")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only largest component per STL.")
    parser.add_argument("--resolve", choices=["overwrite", "keep"], default="overwrite")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--meta", default=None, help="Optional JSON metadata output.")
    args = parser.parse_args()

    pairs = parse_pairs(args.stl, args.label)

    meshes = []
    bounds_min = None
    bounds_max = None

    for label, path in pairs:
        if not os.path.isfile(path):
            print(f"Missing STL: {path}")
            return 1
        mesh = load_mesh(path, keep_largest=args.keep_largest)
        meshes.append((label, path, mesh))
        mn, mx = mesh.bounds
        bounds_min = mn if bounds_min is None else np.minimum(bounds_min, mn)
        bounds_max = mx if bounds_max is None else np.maximum(bounds_max, mx)

    if bounds_min is None:
        print("No meshes loaded.")
        return 1

    bounds_min = bounds_min - args.padding
    bounds_max = bounds_max + args.padding

    dims = np.ceil((bounds_max - bounds_min) / args.pitch).astype(int) + 1
    if np.any(dims <= 0):
        print(f"Invalid volume size: {dims}")
        return 1

    voxel_count = int(np.prod(dims))
    est_mb = voxel_count * 2 / (1024 * 1024)
    print(f"Volume dims: {dims.tolist()} voxels (~{est_mb:.1f} MB uint16)")

    volume = np.zeros(dims, dtype=np.uint16)

    for label, path, mesh in meshes:
        vg = voxelize_mesh(mesh, args.pitch, fill=args.fill)
        count = paste_voxels(volume, vg, label, bounds_min, args.pitch, resolve=args.resolve)
        print(f"Label {label}: {os.path.basename(path)} -> {count} voxels")

    np.savez_compressed(args.out, volume=volume, origin=bounds_min, pitch=args.pitch)
    print(f"Saved volume: {args.out}")

    if args.meta:
        meta = {
            "origin": bounds_min.tolist(),
            "pitch": float(args.pitch),
            "dims": dims.tolist(),
            "labels": {str(label): path for label, path, _ in meshes},
            "resolve": args.resolve,
            "filled": bool(args.fill),
        }
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta: {args.meta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
