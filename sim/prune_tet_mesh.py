import argparse
import json
import os
import sys

import numpy as np
import pyvista as pv


def load_tet_mesh(tet_path=None, tet_npz_path=None):
    origin = None
    pitch = None
    if tet_path:
        if not os.path.isfile(tet_path):
            raise FileNotFoundError(f"Tet mesh not found: {tet_path}")
        grid = pv.read(tet_path)
        cells = grid.cells.reshape(-1, 5)
        tets = cells[:, 1:5].astype(np.int64)
        nodes = grid.points.astype(float)
        if "label" in grid.cell_data:
            labels = np.asarray(grid.cell_data["label"]).astype(np.int32)
        elif "Label" in grid.cell_data:
            labels = np.asarray(grid.cell_data["Label"]).astype(np.int32)
        else:
            raise ValueError("Tet mesh is missing cell_data['label'].")
        return nodes, tets, labels, origin, pitch

    if not tet_npz_path or not os.path.isfile(tet_npz_path):
        raise FileNotFoundError(f"Tet NPZ not found: {tet_npz_path}")
    data = np.load(tet_npz_path)
    nodes = data["nodes"].astype(float)
    tets = data["tets"].astype(np.int64)
    labels = data["labels"].astype(np.int32)
    if "origin" in data:
        origin = data["origin"].astype(float)
    if "pitch" in data:
        pitch = float(data["pitch"])
    return nodes, tets, labels, origin, pitch


def save_tet_mesh(path, nodes, tets, labels):
    cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64), tets]).reshape(-1)
    celltypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, nodes)
    grid.cell_data["label"] = labels.astype(np.int32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid.save(path)


def prune_by_muscle_back(nodes, tets, labels, muscle_label=6, axis="y", tol=1e-8):
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError(f"Invalid axis: {axis}")
    idx = axis_map[axis]

    muscle_mask = labels == int(muscle_label)
    if not np.any(muscle_mask):
        raise ValueError(f"No tets found with muscle label {muscle_label}.")

    muscle_nodes = np.unique(tets[muscle_mask].reshape(-1))
    muscle_min = float(nodes[muscle_nodes, idx].min())
    tet_axis_max = nodes[tets][:, :, idx].max(axis=1)

    remove_mask = muscle_mask | (tet_axis_max <= muscle_min + float(tol))
    keep_mask = ~remove_mask

    removed_tets = tets[remove_mask]
    kept_tets = tets[keep_mask]
    if len(kept_tets) == 0:
        raise ValueError("Prune rule removed all tets.")

    removed_nodes_old = np.unique(removed_tets.reshape(-1))
    kept_nodes_old = np.unique(kept_tets.reshape(-1))
    interface_nodes_old = np.intersect1d(removed_nodes_old, kept_nodes_old)

    node_map = np.full(len(nodes), -1, dtype=np.int64)
    node_map[kept_nodes_old] = np.arange(len(kept_nodes_old), dtype=np.int64)

    new_nodes = nodes[kept_nodes_old]
    new_tets = node_map[kept_tets]
    new_labels = labels[keep_mask]
    interface_nodes_new = node_map[interface_nodes_old]

    meta = {
        "axis": axis,
        "muscle_label": int(muscle_label),
        "muscle_min": muscle_min,
        "remove_rule": f"remove label=={int(muscle_label)} or tet max({axis}) <= muscle_min",
        "input": {"nodes": int(len(nodes)), "tets": int(len(tets))},
        "output": {
            "nodes": int(len(new_nodes)),
            "tets": int(len(new_tets)),
            "fixed_nodes": int(len(interface_nodes_new)),
        },
        "removed": {
            "tets": int(remove_mask.sum()),
            "nodes": int(len(removed_nodes_old)),
            "tet_labels": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(labels[remove_mask], return_counts=True))
            },
        },
        "kept": {
            "tet_labels": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(new_labels, return_counts=True))
            },
        },
    }
    return new_nodes, new_tets, new_labels, interface_nodes_new, meta


def main():
    parser = argparse.ArgumentParser(description="Prune a tet mesh by removing muscle and anything posterior to the muscle back face.")
    parser.add_argument("--tet", help="Input tet mesh VTK.")
    parser.add_argument("--tet-npz", help="Input tet mesh NPZ.")
    parser.add_argument("--out", required=True, help="Output pruned tet mesh VTK.")
    parser.add_argument("--npz-out", help="Optional output tet mesh NPZ.")
    parser.add_argument("--fixed-nodes-out", required=True, help="Output NPZ containing fixed_nodes indices for the pruned mesh.")
    parser.add_argument("--meta-out", required=True, help="Output JSON metadata path.")
    parser.add_argument("--muscle-label", type=int, default=6, help="Tet label treated as muscle.")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="y", help="Posterior axis to prune along.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Tolerance on the posterior cutoff.")
    args = parser.parse_args()

    if bool(args.tet) == bool(args.tet_npz):
        print("Provide exactly one of --tet or --tet-npz.")
        return 1

    try:
        nodes, tets, labels, origin, pitch = load_tet_mesh(args.tet, args.tet_npz)
        new_nodes, new_tets, new_labels, fixed_nodes, meta = prune_by_muscle_back(
            nodes,
            tets,
            labels,
            muscle_label=args.muscle_label,
            axis=args.axis,
            tol=args.tol,
        )
    except Exception as exc:
        print(exc)
        return 1

    save_tet_mesh(args.out, new_nodes, new_tets, new_labels)
    print(f"Saved pruned tet mesh: {args.out}")

    os.makedirs(os.path.dirname(args.fixed_nodes_out), exist_ok=True)
    np.savez_compressed(args.fixed_nodes_out, fixed_nodes=fixed_nodes.astype(np.int64))
    print(f"Saved fixed nodes: {args.fixed_nodes_out}")

    if args.npz_out:
        os.makedirs(os.path.dirname(args.npz_out), exist_ok=True)
        payload = {"nodes": new_nodes, "tets": new_tets, "labels": new_labels}
        if origin is not None:
            payload["origin"] = origin
        if pitch is not None:
            payload["pitch"] = pitch
        np.savez_compressed(args.npz_out, **payload)
        print(f"Saved pruned tet NPZ: {args.npz_out}")

    meta.update(
        {
            "source_tet": args.tet,
            "source_tet_npz": args.tet_npz,
            "out": args.out,
            "npz_out": args.npz_out,
            "fixed_nodes_out": args.fixed_nodes_out,
        }
    )
    os.makedirs(os.path.dirname(args.meta_out), exist_ok=True)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {args.meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
