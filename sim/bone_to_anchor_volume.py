import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage


def load_volume(npz_path):
    data = np.load(npz_path)
    volume = data["volume"]
    origin = data["origin"].astype(float)
    pitch = float(data["pitch"])
    return volume, origin, pitch


def validate_compatible(reference_shape, reference_origin, reference_pitch, bone_shape, bone_origin, bone_pitch):
    if tuple(reference_shape) != tuple(bone_shape):
        raise ValueError(f"Shape mismatch: {reference_shape} vs {bone_shape}")
    if abs(float(reference_pitch) - float(bone_pitch)) > 1e-9:
        raise ValueError(f"Pitch mismatch: {reference_pitch} vs {bone_pitch}")
    if not np.allclose(reference_origin, bone_origin):
        raise ValueError(f"Origin mismatch: {reference_origin.tolist()} vs {bone_origin.tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Relabel tissue voxels near a bone mask into an anchor region.")
    parser.add_argument("--volume-npz", required=True, help="Input labeled tissue volume NPZ.")
    parser.add_argument("--bone-npz", required=True, help="Input bone mask NPZ.")
    parser.add_argument(
        "--distance-voxels",
        type=float,
        default=None,
        help="Distance threshold in voxel units on the coarse grid.",
    )
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=None,
        help="Distance threshold in mm on the coarse grid.",
    )
    parser.add_argument(
        "--body-label",
        action="append",
        type=int,
        default=None,
        help="Restrict anchor candidates to these labels. Repeatable; default is all nonzero labels.",
    )
    parser.add_argument("--anchor-label", type=int, default=8, help="Relabel selected voxels to this label.")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest connected anchor component.")
    parser.add_argument("--out", required=True, help="Output merged labeled volume NPZ.")
    parser.add_argument("--meta", default=None, help="Optional JSON metadata output.")
    args = parser.parse_args()

    if (args.distance_voxels is None) == (args.distance_mm is None):
        raise ValueError("Provide exactly one of --distance-voxels or --distance-mm.")

    volume, origin, pitch = load_volume(args.volume_npz)
    bone_volume, bone_origin, bone_pitch = load_volume(args.bone_npz)
    validate_compatible(volume.shape, origin, pitch, bone_volume.shape, bone_origin, bone_pitch)

    if args.distance_mm is not None:
        distance_voxels = float(args.distance_mm) / float(pitch)
    else:
        distance_voxels = float(args.distance_voxels)

    body_labels = set(int(v) for v in (args.body_label or []))
    if body_labels:
        body_mask = np.isin(volume, np.asarray(sorted(body_labels), dtype=volume.dtype))
    else:
        body_mask = volume > 0

    bone_mask = bone_volume > 0
    if not np.any(bone_mask):
        raise ValueError("Bone volume is empty.")

    distance_to_bone = ndimage.distance_transform_edt(~bone_mask)
    anchor_mask = body_mask & (distance_to_bone <= distance_voxels)

    if args.keep_largest and np.any(anchor_mask):
        labels, num = ndimage.label(anchor_mask)
        if num > 1:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            anchor_mask = labels == counts.argmax()

    out_volume = volume.copy()
    out_volume[anchor_mask] = int(args.anchor_label)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, volume=out_volume, origin=origin, pitch=pitch)
    print(f"Saved anchor volume: {args.out}")
    print(f"Anchor voxels: {int(anchor_mask.sum())}")

    if args.meta:
        labels, counts = np.unique(volume[anchor_mask], return_counts=True) if np.any(anchor_mask) else (np.array([], dtype=int), np.array([], dtype=int))
        meta = {
            "volume_npz": args.volume_npz,
            "bone_npz": args.bone_npz,
            "distance_voxels": float(distance_voxels),
            "distance_mm": float(distance_voxels * pitch),
            "body_labels": sorted(body_labels) if body_labels else "all_nonzero",
            "anchor_label": int(args.anchor_label),
            "keep_largest": bool(args.keep_largest),
            "anchor_voxels": int(anchor_mask.sum()),
            "source_label_counts": {str(int(k)): int(v) for k, v in zip(labels, counts)},
            "origin": origin.tolist(),
            "pitch": float(pitch),
            "shape": [int(v) for v in volume.shape],
        }
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta: {args.meta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
