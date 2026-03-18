import argparse
import json
import os
import struct
import sys

import numpy as np


DTYPE_MAP = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def load_nifti(path):
    with open(path, "rb") as f:
        header = f.read(348)
        if len(header) < 348:
            raise ValueError(f"Short NIfTI header: {path}")
        sizeof_hdr_le = struct.unpack("<i", header[0:4])[0]
        if sizeof_hdr_le == 348:
            endian = "<"
        else:
            sizeof_hdr_be = struct.unpack(">i", header[0:4])[0]
            if sizeof_hdr_be != 348:
                raise ValueError(f"Invalid NIfTI header: {path}")
            endian = ">"

        dim = struct.unpack(endian + "8h", header[40:56])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
        sform_code = struct.unpack(endian + "h", header[254:256])[0]
        pixdim = np.asarray(struct.unpack(endian + "8f", header[76:108])[1:4], dtype=float)
        shape = tuple(int(v) for v in dim[1:4])
        dtype = DTYPE_MAP.get(datatype)
        if dtype is None:
            raise ValueError(f"Unsupported NIfTI datatype {datatype}: {path}")

        affine = np.eye(4, dtype=float)
        if sform_code > 0:
            affine[0, :] = struct.unpack(endian + "4f", header[280:296])
            affine[1, :] = struct.unpack(endian + "4f", header[296:312])
            affine[2, :] = struct.unpack(endian + "4f", header[312:328])
        else:
            affine[0, 0] = pixdim[0]
            affine[1, 1] = pixdim[1]
            affine[2, 2] = pixdim[2]

    with open(path, "rb") as f:
        f.seek(vox_offset)
        data = np.fromfile(f, dtype=dtype, count=int(np.prod(shape)))
    if data.size != int(np.prod(shape)):
        raise ValueError(f"Unexpected voxel count in {path}")
    data = data.reshape(shape, order="F")
    return data, affine


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def parse_mask_entries(args):
    if args.config:
        config_path = os.path.abspath(args.config)
        config = load_config(config_path)
        base_dir = os.path.dirname(config_path)
        entries = config.get("masks", [])
        if not entries:
            raise ValueError("Config must contain a non-empty 'masks' list.")
        out_path = resolve_path(base_dir, config["out"]) if config.get("out") else args.out
        meta_path = resolve_path(base_dir, config["meta"]) if config.get("meta") else args.meta
        pitch = float(config.get("pitch", args.pitch))
        resolve = config.get("resolve", args.resolve)
        threshold = float(config.get("threshold", args.threshold))
        parsed = []
        for entry in entries:
            parsed.append(
                {
                    "path": resolve_path(base_dir, entry["path"]),
                    "label": int(entry["label"]),
                    "name": entry.get("name") or os.path.basename(entry["path"]),
                }
            )
        return parsed, out_path, meta_path, pitch, resolve, threshold

    if not args.mask or not args.label or len(args.mask) != len(args.label):
        raise ValueError("Provide equal numbers of --mask and --label arguments, or use --config.")
    parsed = []
    for label, path in zip(args.label, args.mask):
        parsed.append({"path": path, "label": int(label), "name": os.path.basename(path)})
    return parsed, args.out, args.meta, float(args.pitch), args.resolve, float(args.threshold)


def volume_bounds(shape, affine):
    corners = []
    for i in (-0.5, shape[0] - 0.5):
        for j in (-0.5, shape[1] - 0.5):
            for k in (-0.5, shape[2] - 0.5):
                pt = affine @ np.array([i, j, k, 1.0], dtype=float)
                corners.append(pt[:3])
    corners = np.asarray(corners, dtype=float)
    return corners.min(axis=0), corners.max(axis=0)


def output_centers(origin, dims, pitch):
    grid_x = origin[0] + (np.arange(dims[0], dtype=float) + 0.5) * pitch
    grid_y = origin[1] + (np.arange(dims[1], dtype=float) + 0.5) * pitch
    grid_z = origin[2] + (np.arange(dims[2], dtype=float) + 0.5) * pitch
    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def sample_mask(mask, affine_inv, world_points):
    pts_h = np.c_[world_points, np.ones((world_points.shape[0], 1), dtype=float)]
    ijk = (affine_inv @ pts_h.T).T[:, :3]
    idx = np.rint(ijk).astype(int)
    valid = (
        (idx[:, 0] >= 0)
        & (idx[:, 1] >= 0)
        & (idx[:, 2] >= 0)
        & (idx[:, 0] < mask.shape[0])
        & (idx[:, 1] < mask.shape[1])
        & (idx[:, 2] < mask.shape[2])
    )
    values = np.zeros(world_points.shape[0], dtype=bool)
    valid_idx = idx[valid]
    values[valid] = mask[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
    return values


def main():
    parser = argparse.ArgumentParser(description="Rasterize labeled NIfTI masks onto a coarse voxel grid.")
    parser.add_argument("--config", help="JSON config file describing masks, labels, and output paths.")
    parser.add_argument("--mask", action="append", help="NIfTI mask path (repeatable).")
    parser.add_argument("--label", action="append", type=int, help="Output label for each mask (repeatable).")
    parser.add_argument("--pitch", type=float, default=6.0, help="Output isotropic voxel pitch in world units.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Mask threshold; values above this are treated as inside.")
    parser.add_argument("--resolve", choices=["overwrite", "keep"], default="overwrite")
    parser.add_argument("--out", help="Output .npz path.")
    parser.add_argument("--meta", help="Optional JSON metadata output.")
    args = parser.parse_args()

    try:
        entries, out_path, meta_path, pitch, resolve_mode, threshold = parse_mask_entries(args)
    except Exception as exc:
        print(exc)
        return 1

    if not out_path:
        print("No output path provided.")
        return 1

    reference_shape = None
    reference_affine = None
    loaded = []
    for entry in entries:
        path = entry["path"]
        if not os.path.isfile(path):
            print(f"Missing mask: {path}")
            return 1
        data, affine = load_nifti(path)
        mask = data > threshold
        if reference_shape is None:
            reference_shape = mask.shape
            reference_affine = affine
        else:
            if mask.shape != reference_shape:
                print(f"Shape mismatch: {path} -> {mask.shape}, expected {reference_shape}")
                return 1
            if not np.allclose(affine, reference_affine, atol=1e-4):
                print(f"Affine mismatch: {path}")
                return 1
        loaded.append((entry, mask))

    bounds_min, bounds_max = volume_bounds(reference_shape, reference_affine)
    dims = np.ceil((bounds_max - bounds_min) / pitch).astype(int)
    dims = np.maximum(dims, 1)
    centers = output_centers(bounds_min, dims, pitch)
    affine_inv = np.linalg.inv(reference_affine)

    voxel_count = int(np.prod(dims))
    est_mb = voxel_count * 2 / (1024 * 1024)
    print(f"Output dims: {dims.tolist()} voxels (~{est_mb:.1f} MB uint16)")

    volume = np.zeros(tuple(int(v) for v in dims), dtype=np.uint16)
    flat = volume.reshape(-1)

    for entry, mask in loaded:
        sampled = sample_mask(mask, affine_inv, centers)
        count = int(np.count_nonzero(sampled))
        if resolve_mode == "overwrite":
            flat[sampled] = entry["label"]
        else:
            flat[np.logical_and(sampled, flat == 0)] = entry["label"]
        print(f"Label {entry['label']}: {entry['name']} -> {count} voxels")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, volume=volume, origin=bounds_min, pitch=pitch, source_affine=reference_affine)
    print(f"Saved volume: {out_path}")

    if meta_path:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        meta = {
            "origin": bounds_min.tolist(),
            "pitch": float(pitch),
            "dims": [int(v) for v in dims.tolist()],
            "resolve": resolve_mode,
            "threshold": float(threshold),
            "source_shape": [int(v) for v in reference_shape],
            "source_affine": reference_affine.tolist(),
            "masks": entries,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
