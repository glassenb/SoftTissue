import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv
import pypbd as pbd


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
GRAVITY_TO_VECTOR = {
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+x": np.array([1.0, 0.0, 0.0]),
    "x": np.array([1.0, 0.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*?$", "", text, flags=re.M)
    return text


def load_config(path: Path) -> dict:
    return json.loads(strip_json_comments(path.read_text(encoding="utf-8-sig")))


def gravity_transform(gravity_axis: str) -> np.ndarray:
    g = GRAVITY_TO_VECTOR[gravity_axis].astype(np.float64)
    y_int_in_orig = -g
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, y_int_in_orig)) > 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    x_int_in_orig = np.cross(helper, y_int_in_orig)
    x_int_in_orig /= np.linalg.norm(x_int_in_orig)
    z_int_in_orig = np.cross(x_int_in_orig, y_int_in_orig)
    z_int_in_orig /= np.linalg.norm(z_int_in_orig)
    return np.stack([x_int_in_orig, y_int_in_orig, z_int_in_orig], axis=0)


def select_fixed_nodes(points_orig: np.ndarray, cfg: dict) -> np.ndarray:
    fixed_nodes_file = cfg.get("fix_nodes_file")
    if fixed_nodes_file:
        data = np.load(fixed_nodes_file)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "fixed_nodes" in data.files:
                return np.asarray(data["fixed_nodes"], dtype=np.int32)
            first = data.files[0]
            return np.asarray(data[first], dtype=np.int32)
        return np.asarray(data, dtype=np.int32)

    fix_axis = cfg.get("fix_axis")
    if not fix_axis:
        return np.empty((0,), dtype=np.int32)
    axis = AXIS_TO_INDEX[fix_axis]
    coords = points_orig[:, axis]
    side = cfg.get("fix_side", "max")
    frac = float(cfg.get("fix_frac", 0.0))
    lo = float(coords.min())
    hi = float(coords.max())
    if side == "max":
        threshold = hi - frac * (hi - lo)
        fixed = coords >= threshold
    elif side == "min":
        threshold = lo + frac * (hi - lo)
        fixed = coords <= threshold
    else:
        raise ValueError(f"Unsupported fix_side: {side}")

    extend_axis = cfg.get("fix_extend_axis")
    extend_frac = cfg.get("fix_extend_frac")
    if extend_axis and extend_frac:
        ext_axis = AXIS_TO_INDEX[extend_axis]
        ext_coords = points_orig[:, ext_axis]
        ext_side = cfg.get("fix_extend_side", side)
        ext_frac = float(extend_frac)
        ext_lo = float(ext_coords.min())
        ext_hi = float(ext_coords.max())
        if ext_side == "max":
            ext_threshold = ext_hi - ext_frac * (ext_hi - ext_lo)
            extra = ext_coords >= ext_threshold
        elif ext_side == "min":
            ext_threshold = ext_lo + ext_frac * (ext_hi - ext_lo)
            extra = ext_coords <= ext_threshold
        else:
            raise ValueError(f"Unsupported fix_extend_side: {ext_side}")
        fixed |= extra

    return np.flatnonzero(fixed).astype(np.int32)


def build_faces(surface_mesh) -> np.ndarray:
    faces = np.array(surface_mesh.getFaces(), dtype=np.int32).reshape(-1, 3)
    counts = np.full((faces.shape[0], 1), 3, dtype=np.int32)
    return np.hstack([counts, faces]).ravel()


def camera_from_bounds(bounds, y_side: str = "max"):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5])
    extent = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    radius = max(float(np.linalg.norm(extent)), 1e-3)
    y = ymax + 0.9 * radius if y_side == "max" else ymin - 0.9 * radius
    x = xmin - 0.45 * radius
    z = zmax + 0.35 * radius
    pos = np.array([x, y, z])
    return pos, center, np.array([0.0, 0.0, 1.0])


def make_plotter(initial_points: np.ndarray, surface_faces: np.ndarray, cfg: dict):
    current = pv.PolyData(initial_points.copy(), surface_faces)
    initial = pv.PolyData(initial_points.copy(), surface_faces)
    plotter = pv.Plotter(off_screen=True, window_size=tuple(cfg.get("window_size", [1280, 960])))
    plotter.set_background(cfg.get("background", "white"))
    plotter.add_mesh(initial, color=cfg.get("initial_color", "lightgray"), opacity=float(cfg.get("initial_opacity", 0.18)), smooth_shading=False)
    plotter.add_mesh(current, color=cfg.get("current_color", "salmon"), opacity=float(cfg.get("current_opacity", 0.75)), smooth_shading=False)
    plotter.show_axes()
    cpos = camera_from_bounds(current.bounds, cfg.get("camera_y_side", "max"))
    plotter.camera_position = cpos
    return plotter, current


def ensure_empty_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def encode_video(frames_dir: Path, output_mp4: Path, fps: int):
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)


def run(cfg: dict) -> dict:
    tet_npz = Path(cfg["tet_npz"])
    mesh = np.load(tet_npz)
    points_orig = np.asarray(mesh["nodes"], dtype=np.float64) * float(cfg.get("unit_scale", 1.0))
    tets = np.asarray(mesh["tets"], dtype=np.int32)

    transform = gravity_transform(cfg.get("gravity_axis", "-y"))
    points_int = points_orig @ transform.T

    sim = pbd.Simulation.getCurrent()
    sim.initDefault()
    model = sim.getModel()

    tet_model = model.addTetModel(points_int.tolist(), tets.reshape(-1).tolist(), testMesh=bool(cfg.get("test_mesh", False)))
    particles = model.getParticles()

    fixed_nodes = select_fixed_nodes(points_orig, cfg)
    for idx in fixed_nodes:
        particles.setMass(int(idx), 0.0)

    solid_method = int(cfg.get("solid_method", 3))
    stiffness = float(cfg.get("stiffness", 1e5))
    poisson_ratio = float(cfg.get("poisson_ratio", 0.3))
    volume_stiffness = float(cfg.get("volume_stiffness", stiffness))
    model.addSolidConstraints(tet_model, solid_method, stiffness, poisson_ratio, volume_stiffness, bool(cfg.get("normalize_stretch", False)), bool(cfg.get("normalize_shear", False)))

    timestep = sim.getTimeStep()
    timestep.setValueUInt(pbd.TimeStepController.NUM_SUB_STEPS, int(cfg.get("substeps", 5)))
    timestep.setValueUInt(pbd.TimeStepController.MAX_ITERATIONS, int(cfg.get("max_iterations", 1)))
    timestep.setValueUInt(pbd.TimeStepController.MAX_ITERATIONS_V, int(cfg.get("max_velocity_iterations", 0)))
    timestep.setValueInt(pbd.TimeStepController.VELOCITY_UPDATE_METHOD, int(cfg.get("velocity_update_method", pbd.TimeStepController.ENUM_VUPDATE_FIRST_ORDER)))
    pbd.TimeManager.getCurrent().setTimeStepSize(float(cfg.get("dt", 0.002)))

    render_cfg = cfg.get("render", {})
    record_dir = Path(render_cfg["record_dir"]) if render_cfg.get("record_dir") else None
    output_video = Path(render_cfg["record_video"]) if render_cfg.get("record_video") else None
    render_every = max(1, int(render_cfg.get("record_every", 1)))
    record_fps = int(render_cfg.get("record_fps", 30))

    surface_faces = build_faces(tet_model.getSurfaceMesh())
    plotter = None
    current_poly = None
    if record_dir:
        ensure_empty_dir(record_dir)
        plotter, current_poly = make_plotter(points_orig, surface_faces, render_cfg)

    steps = int(cfg.get("steps", 150))
    vel_damp = float(cfg.get("velocity_damping", 1.0))
    report_axis = cfg.get("report_axis", "z")
    report_index = AXIS_TO_INDEX[report_axis]

    max_disp = 0.0
    for step in range(steps):
        timestep.step(model)
        if vel_damp < 1.0:
            for i in range(particles.size()):
                if particles.getMass(i) != 0.0:
                    particles.setVelocity(i, np.asarray(particles.getVelocity(i), dtype=np.float64) * vel_damp)

        if record_dir and (step % render_every == 0 or step == steps - 1):
            points_step = np.asarray(particles.getVertices(), dtype=np.float64) @ transform
            current_poly.points = points_step
            plotter.camera_position = camera_from_bounds(current_poly.bounds, render_cfg.get("camera_y_side", "max"))
            plotter.add_text(f"XPBD step {step + 1}/{steps}", position="upper_left", font_size=10, color="black", name="status_text")
            plotter.show(screenshot=str(record_dir / f"frame_{step // render_every:05d}.png"), auto_close=False)

    points_final = np.asarray(particles.getVertices(), dtype=np.float64) @ transform
    disp = points_final - points_orig
    max_disp = float(np.linalg.norm(disp, axis=1).max())
    axis_disp = disp[:, report_index]

    if plotter is not None:
        plotter.close()

    if record_dir and output_video:
        encode_video(record_dir, output_video, record_fps)

    result = {
        "tet_npz": str(tet_npz),
        "steps": steps,
        "dt": float(cfg.get("dt", 0.002)),
        "fixed_nodes": int(fixed_nodes.size),
        "max_disp": max_disp,
        f"mean_{report_axis}_disp": float(axis_disp.mean()),
        f"min_{report_axis}_disp": float(axis_disp.min()),
        f"max_{report_axis}_disp": float(axis_disp.max()),
        "record_dir": str(record_dir) if record_dir else None,
        "record_video": str(output_video) if output_video else None,
    }

    metadata_path = cfg.get("metadata_json")
    if metadata_path:
        meta_path = Path(metadata_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    result = run(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
