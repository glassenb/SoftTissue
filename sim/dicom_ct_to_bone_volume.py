import argparse
import json
import os
import sys

import numpy as np
import pydicom


def load_reference_volume(npz_path):
    data = np.load(npz_path)
    volume = data["volume"]
    origin = data["origin"].astype(float)
    pitch = float(data["pitch"])
    return volume.shape, origin, pitch


def load_ct_series(dicom_dir):
    files = sorted(
        [
            os.path.join(dicom_dir, name)
            for name in os.listdir(dicom_dir)
            if os.path.isfile(os.path.join(dicom_dir, name))
        ]
    )
    if not files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    headers = []
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        headers.append((path, ds))

    first = headers[0][1]
    image_orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    col_dir = image_orientation[:3]
    row_dir = image_orientation[3:]
    normal = np.cross(col_dir, row_dir)
    normal /= np.linalg.norm(normal)
    spacing_row = float(first.PixelSpacing[0])
    spacing_col = float(first.PixelSpacing[1])

    positions = []
    for path, ds in headers:
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        positions.append((float(np.dot(ipp, normal)), path))
    positions.sort(key=lambda item: item[0])

    if len(positions) > 1:
        slice_spacing = float(np.median(np.diff([item[0] for item in positions])))
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))

    volume = np.zeros((int(first.Rows), int(first.Columns), len(positions)), dtype=np.float32)
    for k, (_, path) in enumerate(positions):
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume[:, :, k] = arr * slope + intercept

    origin = np.asarray(pydicom.dcmread(positions[0][1], stop_before_pixels=True, force=True).ImagePositionPatient, dtype=float)
    affine = np.eye(4, dtype=float)
    affine[:3, 0] = row_dir * spacing_row
    affine[:3, 1] = col_dir * spacing_col
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = origin
    return volume, affine


def output_centers(origin, dims, pitch):
    grid_x = origin[0] + (np.arange(dims[0], dtype=float) + 0.5) * pitch
    grid_y = origin[1] + (np.arange(dims[1], dtype=float) + 0.5) * pitch
    grid_z = origin[2] + (np.arange(dims[2], dtype=float) + 0.5) * pitch
    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def sample_volume(volume, affine_inv, world_points):
    pts_h = np.c_[world_points, np.ones((world_points.shape[0], 1), dtype=float)]
    ijk = (affine_inv @ pts_h.T).T[:, :3]
    idx = np.rint(ijk).astype(int)
    valid = (
        (idx[:, 0] >= 0)
        & (idx[:, 1] >= 0)
        & (idx[:, 2] >= 0)
        & (idx[:, 0] < volume.shape[0])
        & (idx[:, 1] < volume.shape[1])
        & (idx[:, 2] < volume.shape[2])
    )
    values = np.full(world_points.shape[0], -1024.0, dtype=np.float32)
    valid_idx = idx[valid]
    values[valid] = volume[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
    return values


def main():
    parser = argparse.ArgumentParser(description="Extract a coarse CT bone mask volume from a DICOM series.")
    parser.add_argument("--dicom-dir", required=True, help="Directory containing CT DICOM slices.")
    parser.add_argument("--reference-npz", required=True, help="Reference coarse volume NPZ for origin/pitch/dims.")
    parser.add_argument("--threshold-hu", type=float, default=300.0, help="Bone threshold in HU.")
    parser.add_argument("--label", type=int, default=8, help="Label value to assign to bone voxels.")
    parser.add_argument("--out", required=True, help="Output NPZ path.")
    parser.add_argument("--meta", default=None, help="Optional JSON metadata output.")
    args = parser.parse_args()

    dims, origin, pitch = load_reference_volume(args.reference_npz)
    volume_ct, affine = load_ct_series(args.dicom_dir)
    centers = output_centers(origin, np.asarray(dims, dtype=int), pitch)
    affine_inv = np.linalg.inv(affine)
    hu = sample_volume(volume_ct, affine_inv, centers)
    mask = hu >= float(args.threshold_hu)

    out_volume = np.zeros(dims, dtype=np.uint16)
    out_volume.reshape(-1)[mask] = int(args.label)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, volume=out_volume, origin=origin, pitch=pitch, source_affine=affine)
    print(f"Saved bone volume: {args.out}")
    print(f"Bone voxels: {int(mask.sum())}")

    if args.meta:
        meta = {
            "dicom_dir": args.dicom_dir,
            "reference_npz": args.reference_npz,
            "threshold_hu": float(args.threshold_hu),
            "label": int(args.label),
            "origin": origin.tolist(),
            "pitch": float(pitch),
            "dims": [int(v) for v in dims],
            "bone_voxels": int(mask.sum()),
            "source_affine": affine.tolist(),
        }
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta: {args.meta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
