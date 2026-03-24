import argparse
import time
from pathlib import Path

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
    parser.add_argument('--camera', choices=['oblique', 'side_x', 'side_y'], default='oblique')
    parser.add_argument('--show-ghost', action='store_true')
    parser.add_argument('--playback-fps', type=float, default=30.0)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    dump_dir = Path(args.dump_dir)

    nodes = read_nodes(bundle_dir / 'nodes_f32.bin')
    tets = read_tets(bundle_dir / 'tets_u32.bin')
    tris = boundary_triangles(tets)
    faces = np.hstack([np.full((tris.shape[0], 1), 3, dtype=np.int32), tris]).ravel()
    frame_files = sorted(dump_dir.glob('frame_*.bin'))
    if not frame_files:
        raise SystemExit(f'no dumped frames in {dump_dir}')
    frames = [read_nodes(p) for p in frame_files]

    plotter = pv.Plotter(window_size=(1400, 1000))
    plotter.set_background('white')
    moving = pv.PolyData(frames[0].copy(), faces)
    if args.show_ghost:
        ghost = pv.PolyData(nodes.copy(), faces)
        plotter.add_mesh(ghost, color='#c9c9c9', opacity=0.12, show_edges=False)
    plotter.add_mesh(moving, color='#d44b4b', opacity=0.92, show_edges=False)
    label = plotter.add_text('', position='upper_left', font_size=14)
    plotter.add_axes()
    plotter.camera_position = camera_for_mode(nodes, args.camera)

    state = {'frame': 0, 'playing': False, 'syncing_slider': False}
    slider_widget = None

    def update_label():
        status = 'playing' if state['playing'] else 'paused'
        label.SetText(2, f'frame {state["frame"]} / {len(frames)-1} [{status}]  space=play/pause  left/right=step')

    def apply_frame(idx: int, sync_slider: bool = True):
        idx = max(0, min(len(frames) - 1, idx))
        state['frame'] = idx
        moving.points = frames[idx]
        update_label()
        if slider_widget is not None and sync_slider and not state['syncing_slider']:
            state['syncing_slider'] = True
            slider_widget.GetRepresentation().SetValue(float(idx))
            state['syncing_slider'] = False
        plotter.render()

    def slider_callback(value):
        if state['syncing_slider']:
            return
        apply_frame(int(round(value)), sync_slider=False)

    def toggle_playback():
        state['playing'] = not state['playing']
        update_label()
        plotter.render()

    slider_widget = plotter.add_slider_widget(slider_callback, [0, len(frames)-1], value=0, title='Frame', pointa=(0.2, 0.08), pointb=(0.8, 0.08))
    plotter.add_key_event('Right', lambda: apply_frame(state['frame'] + 1))
    plotter.add_key_event('Left', lambda: apply_frame(state['frame'] - 1))
    plotter.add_key_event('space', toggle_playback)
    apply_frame(0, sync_slider=False)

    frame_interval = 1.0 / max(args.playback_fps, 1.0)
    last_advance = time.perf_counter()
    plotter.show(auto_close=False, interactive_update=True)
    while not getattr(plotter, '_closed', False):
        now = time.perf_counter()
        if state['playing'] and (now - last_advance) >= frame_interval:
            last_advance = now
            next_frame = state['frame'] + 1
            if next_frame >= len(frames):
                state['playing'] = False
                update_label()
                plotter.render()
            else:
                apply_frame(next_frame)
        plotter.update(stime=1, force_redraw=False)
        time.sleep(0.005)


if __name__ == '__main__':
    main()
