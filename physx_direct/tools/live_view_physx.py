import argparse
import math
import shutil
import subprocess
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


def write_gravity_control(path: Path, gravity: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(f'{gravity[0]} {gravity[1]} {gravity[2]}\n', encoding='utf-8')
    tmp.replace(path)


def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return vec
    axis = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def parse_initial_gravity(sim_cmd) -> np.ndarray:
    if '--gravity' in sim_cmd:
        idx = sim_cmd.index('--gravity')
        if idx + 3 < len(sim_cmd):
            try:
                return np.array([float(sim_cmd[idx + 1]), float(sim_cmd[idx + 2]), float(sim_cmd[idx + 3])], dtype=float)
            except ValueError:
                pass
    return np.array([0.0, 0.0, -9.81], dtype=float)


def safe_read_latest(dump_dir: Path, expected_count: int):
    latest = dump_dir / 'latest.bin'
    step_file = dump_dir / 'latest_step.txt'
    try:
        raw = latest.read_bytes()
        step_text = step_file.read_text(encoding='utf-8').strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None, None
    if len(raw) != expected_count * 3 * 4:
        return None, None
    try:
        step = int(step_text)
    except ValueError:
        return None, None
    pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).copy()
    return step, pts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--dump-dir', required=True)
    parser.add_argument('--camera', choices=['oblique', 'side_x', 'side_y'], default='oblique')
    parser.add_argument('--show-ghost', action='store_true')
    parser.add_argument('--render-fps', type=float, default=30.0)
    parser.add_argument('--keep-dump', action='store_true')
    parser.add_argument('sim_command', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    sim_cmd = list(args.sim_command)
    while sim_cmd and sim_cmd[0] == '--':
        sim_cmd.pop(0)
    if not sim_cmd:
        raise SystemExit('pass the sim command after --')

    bundle_dir = Path(args.bundle_dir)
    dump_dir = Path(args.dump_dir)
    gravity_control = dump_dir / 'control_gravity.txt'

    if dump_dir.exists() and not args.keep_dump:
        for child in dump_dir.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    nodes = read_nodes(bundle_dir / 'nodes_f32.bin')
    tets = read_tets(bundle_dir / 'tets_u32.bin')
    tris = boundary_triangles(tets)
    faces = np.hstack([np.full((tris.shape[0], 1), 3, dtype=np.int32), tris]).ravel()

    gravity_vec = parse_initial_gravity(sim_cmd)
    write_gravity_control(gravity_control, gravity_vec)
    proc_state = {'proc': subprocess.Popen(sim_cmd)}

    try:
        plotter = pv.Plotter(window_size=(1400, 1000))
        plotter.set_background('white')
        moving = pv.PolyData(nodes.copy(), faces)
        if args.show_ghost:
            ghost = pv.PolyData(nodes.copy(), faces)
            plotter.add_mesh(ghost, color='#c9c9c9', opacity=0.12, show_edges=False)
        plotter.add_mesh(moving, color='#d44b4b', opacity=0.92, show_edges=False)
        label = plotter.add_text('', position='upper_left', font_size=14)
        plotter.add_axes()
        plotter.camera_position = camera_for_mode(nodes, args.camera)

        state = {
            'paused': False,
            'last_step': None,
            'gravity_mode': False,
            'dragging_gravity': False,
            'drag_pos': None,
            'frozen_camera': None,
        }

        def proc_status_text() -> str:
            proc = proc_state['proc']
            return 'running' if proc.poll() is None else f'exit({proc.poll()})'

        def update_label() -> None:
            status = 'paused' if state['paused'] else 'live'
            step_text = '--' if state['last_step'] is None else str(state['last_step'])
            mode = 'gravity-drag' if state['gravity_mode'] else 'camera'
            g = gravity_vec
            label.SetText(2, f'step {step_text}  [{status}]  sim={proc_status_text()}  mode={mode}  g=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f})  space=pause display  r=rerun  g=toggle gravity drag  i/k/j/l=rotate gravity  0=reset g')

        def reset_scene_state() -> None:
            state['paused'] = False
            state['last_step'] = None
            state['dragging_gravity'] = False
            state['drag_pos'] = None
            moving.points = nodes.copy()
            write_gravity_control(gravity_control, gravity_vec)
            update_label()
            plotter.render()

        def stop_proc() -> None:
            proc = proc_state['proc']
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        def rerun_sim() -> None:
            stop_proc()
            if dump_dir.exists() and not args.keep_dump:
                for child in dump_dir.iterdir():
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
            dump_dir.mkdir(parents=True, exist_ok=True)
            write_gravity_control(gravity_control, gravity_vec)
            proc_state['proc'] = subprocess.Popen(sim_cmd)
            reset_scene_state()

        def toggle_pause() -> None:
            state['paused'] = not state['paused']
            update_label()
            plotter.render()

        def toggle_gravity_mode() -> None:
            state['gravity_mode'] = not state['gravity_mode']
            state['dragging_gravity'] = False
            state['drag_pos'] = None
            state['frozen_camera'] = None
            update_label()
            plotter.render()

        def on_right_press(*_args) -> None:
            if not state['gravity_mode']:
                return
            state['dragging_gravity'] = True
            state['drag_pos'] = np.array(plotter.iren.get_event_position(), dtype=float)
            state['frozen_camera'] = plotter.camera_position

        def on_right_release(*_args) -> None:
            if not state['gravity_mode']:
                return
            state['dragging_gravity'] = False
            state['drag_pos'] = None
            if state['frozen_camera'] is not None:
                plotter.camera_position = state['frozen_camera']
                plotter.render()

        def apply_gravity_rotation(delta_yaw: float, delta_pitch: float) -> None:
            cam_pos, focal, up = plotter.camera_position
            cam_pos = np.array(cam_pos, dtype=float)
            focal = np.array(focal, dtype=float)
            up_vec = np.array(up, dtype=float)
            forward = focal - cam_pos
            fnorm = np.linalg.norm(forward)
            if fnorm < 1e-8:
                return
            forward /= fnorm
            upn = up_vec / max(np.linalg.norm(up_vec), 1e-8)
            right = np.cross(forward, upn)
            rnorm = np.linalg.norm(right)
            if rnorm < 1e-8:
                return
            right /= rnorm
            scale = np.linalg.norm(gravity_vec)
            if scale < 1e-8:
                scale = 9.81
            rotated = rotate_vector(gravity_vec, upn, delta_yaw)
            rotated = rotate_vector(rotated, right, delta_pitch)
            gravity_vec[:] = rotated / max(np.linalg.norm(rotated), 1e-8) * scale
            write_gravity_control(gravity_control, gravity_vec)
            if state['frozen_camera'] is not None:
                plotter.camera_position = state['frozen_camera']
            update_label()
            plotter.render()

        def on_mouse_move(*_args) -> None:
            if not state['gravity_mode'] or not state['dragging_gravity'] or state['drag_pos'] is None:
                return
            pos = np.array(plotter.iren.get_event_position(), dtype=float)
            delta = pos - state['drag_pos']
            state['drag_pos'] = pos
            if np.allclose(delta, 0.0):
                return
            sensitivity = 0.0045
            apply_gravity_rotation(-delta[0] * sensitivity, delta[1] * sensitivity)

        plotter.add_key_event('space', toggle_pause)
        plotter.add_key_event('r', rerun_sim)
        plotter.add_key_event('g', toggle_gravity_mode)
        plotter.add_key_event('Up', lambda: apply_gravity_rotation(0.0, 0.08))
        plotter.add_key_event('Down', lambda: apply_gravity_rotation(0.0, -0.08))
        plotter.add_key_event('Left', lambda: apply_gravity_rotation(0.08, 0.0))
        plotter.add_key_event('Right', lambda: apply_gravity_rotation(-0.08, 0.0))
        plotter.add_key_event('i', lambda: apply_gravity_rotation(0.0, 0.08))
        plotter.add_key_event('k', lambda: apply_gravity_rotation(0.0, -0.08))
        plotter.add_key_event('j', lambda: apply_gravity_rotation(0.08, 0.0))
        plotter.add_key_event('l', lambda: apply_gravity_rotation(-0.08, 0.0))
        plotter.add_key_event('0', lambda: (gravity_vec.__setitem__(slice(None), parse_initial_gravity(sim_cmd)), write_gravity_control(gravity_control, gravity_vec), update_label(), plotter.render()))
        plotter.iren.add_observer('RightButtonPressEvent', on_right_press, interactor_style_fallback=False)
        plotter.iren.add_observer('RightButtonReleaseEvent', on_right_release)
        plotter.iren.add_observer('MouseMoveEvent', on_mouse_move, interactor_style_fallback=False)
        update_label()
        plotter.show(auto_close=False, interactive_update=True)

        frame_interval = 1.0 / max(args.render_fps, 1.0)
        last_tick = time.perf_counter()
        while not getattr(plotter, '_closed', False):
            now = time.perf_counter()
            if (now - last_tick) >= frame_interval:
                last_tick = now
                if not state['paused']:
                    step, pts = safe_read_latest(dump_dir, nodes.shape[0])
                    if step is not None and step != state['last_step']:
                        state['last_step'] = step
                        moving.points = pts
                update_label()
                plotter.render()
            plotter.update(stime=1, force_redraw=False)
            time.sleep(0.005)
    finally:
        stop_proc()


if __name__ == '__main__':
    main()
