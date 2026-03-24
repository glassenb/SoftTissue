import argparse
import json
from pathlib import Path

import numpy as np

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def load_fixed_nodes(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.ndarray):
        arr = data
    else:
        if "fixed_nodes" in data.files:
            arr = data["fixed_nodes"]
        elif data.files:
            arr = data[data.files[0]]
        else:
            raise ValueError(f"no arrays in fixed-node file: {path}")
    return np.asarray(arr, dtype=np.uint32).reshape(-1)


def slab_mask(nodes: np.ndarray, axis: str, side: str, frac: float) -> np.ndarray:
    idx = AXIS_TO_INDEX[axis]
    values = nodes[:, idx]
    lo = float(values.min())
    hi = float(values.max())
    threshold = lo + frac * (hi - lo) if side == "min" else hi - frac * (hi - lo)
    if side == "min":
        return values <= threshold
    return values >= threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--unit-scale", type=float, default=1.0)
    parser.add_argument("--fixed-nodes-npz")
    parser.add_argument("--fix-axis", choices=["x", "y", "z"])
    parser.add_argument("--fix-side", choices=["min", "max"])
    parser.add_argument("--fix-frac", type=float)
    parser.add_argument("--fix-extend-axis", choices=["x", "y", "z"])
    parser.add_argument("--fix-extend-side", choices=["min", "max"])
    parser.add_argument("--fix-extend-frac", type=float)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = np.load(npz_path)
    nodes = np.asarray(mesh["nodes"], dtype=np.float32) * np.float32(args.unit_scale)
    tets = np.asarray(mesh["tets"], dtype=np.uint32)
    labels = np.asarray(mesh["labels"], dtype=np.uint16) if "labels" in mesh.files else None

    fixed_nodes = np.empty((0,), dtype=np.uint32)
    if args.fixed_nodes_npz:
        fixed_nodes = load_fixed_nodes(Path(args.fixed_nodes_npz))
    elif args.fix_axis and args.fix_side and args.fix_frac is not None:
        mask = slab_mask(nodes, args.fix_axis, args.fix_side, args.fix_frac)
        if args.fix_extend_axis and args.fix_extend_side and args.fix_extend_frac is not None:
            mask |= slab_mask(nodes, args.fix_extend_axis, args.fix_extend_side, args.fix_extend_frac)
        fixed_nodes = np.flatnonzero(mask).astype(np.uint32)

    (out_dir / "nodes_f32.bin").write_bytes(nodes.astype(np.float32).tobytes())
    (out_dir / "tets_u32.bin").write_bytes(tets.astype(np.uint32).tobytes())
    if labels is not None:
        (out_dir / "labels_u16.bin").write_bytes(labels.astype(np.uint16).tobytes())
    if fixed_nodes.size:
        (out_dir / "fixed_nodes_u32.bin").write_bytes(fixed_nodes.astype(np.uint32).tobytes())

    meta = {
        "source_npz": str(npz_path),
        "unit_scale": args.unit_scale,
        "node_count": int(nodes.shape[0]),
        "tet_count": int(tets.shape[0]),
        "fixed_node_count": int(fixed_nodes.shape[0]),
        "bbox_min": nodes.min(axis=0).astype(float).tolist(),
        "bbox_max": nodes.max(axis=0).astype(float).tolist(),
        "files": {
            "nodes": "nodes_f32.bin",
            "tets": "tets_u32.bin",
            "labels": "labels_u16.bin" if labels is not None else None,
            "fixed_nodes": "fixed_nodes_u32.bin" if fixed_nodes.size else None,
        },
    }
    (out_dir / "bundle_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
