import argparse
import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyvista as pv


def read_nodes(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 3)


def read_tets(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32).reshape(-1, 4)


def boundary_triangles(tets: np.ndarray) -> np.ndarray:
    faces = {}
    for tet in tets:
        a, b, c, d = map(int, tet)
        tris = [(a, b, c), (a, d, b), (a, c, d), (b, d, c)]
        for tri in tris:
            key = tuple(sorted(tri))
            if key in faces:
                del faces[key]
            else:
                faces[key] = tri
    return np.asarray(list(faces.values()), dtype=np.int32)


def camera_for_mode(points: np.ndarray, mode: str):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = np.maximum(maxs - mins, 1e-6)
    dist = float(np.linalg.norm(extent)) * 2.4
    if mode == 'side_x':
        pos = center + np.array([dist, 0.0, 0.0])
        up = (0.0, 0.0, 1.0)
    elif mode == 'side_y':
        pos = center + np.array([0.0, dist, 0.0])
        up = (0.0, 0.0, 1.0)
    else:
        pos = center + np.array([0.9 * dist, -1.2 * dist, 0.7 * dist])
        up = (0.0, 0.0, 1.0)
    return [tuple(pos), tuple(center), up]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--dump-dir', required=True)
    parser.add_argument('--out-video', required=True)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--camera', choices=['oblique', 'side_x', 'side_y'], default='oblique')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--show-ghost', action='store_true')
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    dump_dir = Path(args.dump_dir)
    out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    nodes = read_nodes(bundle_dir / 'nodes_f32.bin')
    tets = read_tets(bundle_dir / 'tets_u32.bin')
    tris = boundary_triangles(tets)
    faces = np.hstack([np.full((tris.shape[0], 1), 3, dtype=np.int32), tris]).ravel()

    frame_files = sorted(dump_dir.glob('frame_*.bin'))
    if not frame_files:
        raise SystemExit(f'no dumped frames in {dump_dir}')

    plotter = pv.Plotter(off_screen=True, window_size=(args.width, args.height))
    plotter.set_background('white')
    moving = pv.PolyData(nodes.copy(), faces)
    if args.show_ghost:
        ghost = pv.PolyData(nodes.copy(), faces)
        plotter.add_mesh(ghost, color='#c9c9c9', opacity=0.15, show_edges=False)
    plotter.add_mesh(moving, color='#d44b4b', opacity=0.9, show_edges=False)
    plotter.add_axes()
    plotter.camera_position = camera_for_mode(nodes, args.camera)

    with imageio.get_writer(out_video, fps=args.fps, codec='libx264', quality=8) as writer:
        for frame_path in frame_files:
            pts = read_nodes(frame_path)
            moving.points = pts
            img = plotter.screenshot(return_img=True)
            writer.append_data(img)

    meta = {
        'bundle_dir': str(bundle_dir),
        'dump_dir': str(dump_dir),
        'out_video': str(out_video),
        'fps': args.fps,
        'camera': args.camera,
        'frame_count': len(frame_files),
    }
    (out_video.with_suffix('.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
