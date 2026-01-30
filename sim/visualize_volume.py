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


def main():
    parser = argparse.ArgumentParser(description="Visualize a labeled voxel volume.")
    parser.add_argument("--npz", required=True, help="Input volume NPZ.")
    parser.add_argument("--out", help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show interactive plot.")
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"Missing volume: {args.npz}")
        return 1

    volume, origin, pitch = load_volume(args.npz)
    grid = make_grid(volume, origin, pitch)

    labels = np.unique(volume)
    labels = labels[labels > 0]
    if labels.size == 0:
        print("No labels found in volume.")
        return 1

    plotter = pv.Plotter(off_screen=bool(args.out) and not args.show)
    plotter.add_axes()
    plotter.add_mesh(grid.outline(), color="black")

    rng = np.random.default_rng(42)
    for label in labels:
        block = grid.threshold([label - 0.5, label + 0.5], scalars="label")
        surface = block.extract_surface().triangulate()
        color = rng.random(3)
        plotter.add_mesh(surface, color=tuple(color.tolist()), opacity=0.65, show_edges=False)

    if args.out:
        plotter.show(screenshot=args.out)
        print(f"Saved: {args.out}")
    elif args.show:
        plotter.show()
    else:
        print("No output requested. Use --out or --show.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
