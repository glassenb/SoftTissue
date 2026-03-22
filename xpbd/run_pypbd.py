import argparse
import json
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


def make_plotter(
    initial_points: np.ndarray,
    surface_faces: np.ndarray,
    cfg: dict,
    plate_render=None,
    plate_render_start=None,
):
    current = pv.PolyData(initial_points.copy(), surface_faces)
    initial = pv.PolyData(initial_points.copy(), surface_faces)
    plotter = pv.Plotter(off_screen=True, window_size=tuple(cfg.get("window_size", [1280, 960])))
    plotter.set_background(cfg.get("background", "white"))
    plotter.add_mesh(initial, color=cfg.get("initial_color", "lightgray"), opacity=float(cfg.get("initial_opacity", 0.18)), smooth_shading=False)
    plotter.add_mesh(current, color=cfg.get("current_color", "salmon"), opacity=float(cfg.get("current_opacity", 0.75)), smooth_shading=False)
    plate_start_poly = None
    if plate_render_start is not None and cfg.get("plate_show_start_ghost", True):
        plate_start_poly = pv.PolyData(plate_render_start["points"].copy(), plate_render_start["faces"])
        plotter.add_mesh(
            plate_start_poly,
            color=cfg.get("plate_start_ghost_color", "royalblue"),
            opacity=float(cfg.get("plate_start_ghost_opacity", 0.22)),
            smooth_shading=False,
            style=cfg.get("plate_start_ghost_style", "wireframe"),
            show_edges=bool(cfg.get("plate_start_ghost_show_edges", True)),
            edge_color=cfg.get("plate_start_ghost_edge_color", "navy"),
            line_width=float(cfg.get("plate_start_ghost_line_width", 2.0)),
        )
    plate_poly = None
    if plate_render is not None:
        plate_poly = pv.PolyData(plate_render["points"].copy(), plate_render["faces"])
        plotter.add_mesh(
            plate_poly,
            color=cfg.get("plate_color", "deepskyblue"),
            opacity=float(cfg.get("plate_opacity", 0.55)),
            smooth_shading=False,
            show_edges=bool(cfg.get("plate_show_edges", True)),
            edge_color=cfg.get("plate_edge_color", "midnightblue"),
            line_width=float(cfg.get("plate_line_width", 1.0)),
        )
    plotter.show_axes()
    cpos = camera_from_bounds(current.bounds, cfg.get("camera_y_side", "max"))
    plotter.camera_position = cpos
    return plotter, current, plate_poly, plate_start_poly, cpos


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


def local_asset_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "assets" / name


def map_original_axis_side_to_internal(transform: np.ndarray, axis_name: str, side: str):
    axis_orig = AXIS_TO_INDEX[axis_name]
    column = transform[:, axis_orig]
    axis_internal = int(np.argmax(np.abs(column)))
    sign = float(column[axis_internal])
    if sign >= 0.0:
        side_internal = side
    else:
        side_internal = "max" if side == "min" else "min"
    inv_axis = {v: k for k, v in AXIS_TO_INDEX.items()}
    return inv_axis[axis_internal], side_internal



def create_plate(model, collision_detection, points_int: np.ndarray, plate_cfg: dict, transform: np.ndarray):
    if not plate_cfg.get("enabled", False):
        return None

    plate_obj = Path(plate_cfg.get("obj", str(local_asset_path("plate.obj"))))
    if not plate_obj.exists():
        raise FileNotFoundError(f"Plate OBJ not found: {plate_obj}")

    (vd, mesh) = pbd.OBJLoader.loadObjToMesh(str(plate_obj), [1, 1, 1])
    mins = points_int.min(axis=0)
    maxs = points_int.max(axis=0)
    extents = np.maximum(maxs - mins, 1e-6)

    if plate_cfg.get("axis_original"):
        axis_name, side = map_original_axis_side_to_internal(
            transform,
            plate_cfg["axis_original"],
            plate_cfg.get("side_original", plate_cfg.get("side", "min")),
        )
    else:
        side = plate_cfg.get("side", "min")
        axis_name = plate_cfg.get("axis", "y")

    axis = AXIS_TO_INDEX[axis_name]
    start_offset = float(plate_cfg.get("start_offset", 0.05))
    end_offset = float(plate_cfg.get("end_offset", start_offset))
    steps = max(0, int(plate_cfg.get("move_steps", 0)))
    move_start_step = max(0, int(plate_cfg.get("move_start_step", 0)))

    raw_scale = plate_cfg.get("scale_fraction", 1.25)
    if isinstance(raw_scale, (list, tuple)):
        tangent_fraction = max(float(v) for v in raw_scale)
    else:
        tangent_fraction = float(raw_scale)
    scale = extents * tangent_fraction
    thickness_fraction = float(plate_cfg.get("thickness_fraction", 0.04))
    scale[axis] = max(extents[axis] * thickness_fraction, float(plate_cfg.get("min_thickness", 0.005)))

    center = 0.5 * (mins + maxs)
    start_pos = center.copy()
    end_pos = center.copy()
    coord_min = mins[axis]
    coord_max = maxs[axis]
    sign = -1.0 if side == "min" else 1.0
    start_pos[axis] = (coord_min if side == "min" else coord_max) + sign * (start_offset + 0.5 * scale[axis])
    end_pos[axis] = (coord_min if side == "min" else coord_max) + sign * (end_offset + 0.5 * scale[axis])

    rb = model.addRigidBody(
        float(plate_cfg.get("density", 1.0)),
        vd,
        mesh,
        start_pos,
        np.identity(3),
        scale,
        testMesh=False,
        generateCollisionObject=False,
        resolution=np.asarray(plate_cfg.get("resolution", [20, 10, 20]), dtype=np.uint32),
    )
    rb.setMass(0.0)
    rb.setFrictionCoeff(float(plate_cfg.get("friction", 0.2)))
    rb.setRestitutionCoeff(float(plate_cfg.get("restitution", 0.0)))
    rb.setVelocity(np.zeros(3, dtype=np.float64))
    rb.setVelocity0(np.zeros(3, dtype=np.float64))
    rb.setAcceleration(np.zeros(3, dtype=np.float64))
    rb.setLastPosition(start_pos)
    rb.setOldPosition(start_pos)
    rb.setPosition0(start_pos)
    rb.updateInverseTransformation()
    rb.getGeometry().updateMeshTransformation(rb.getPosition(), rb.getRotationMatrix())
    base_points_int = np.array(rb.getGeometry().getVertexData().getVertices(), dtype=np.float64, copy=True)
    base_faces = build_faces(rb.getGeometry().getMesh())
    if collision_detection is not None:
        collision_box_scale = scale * float(plate_cfg.get("collision_box_scale", 1.0))
        vdl = rb.getGeometry().getVertexDataLocal()
        collision_detection.addCollisionBox(
            len(model.getRigidBodies()) - 1,
            pbd.CollisionObject.RigidBodyCollisionObjectType,
            vdl,
            0,
            vdl.size(),
            collision_box_scale,
            bool(plate_cfg.get("collision_test_mesh", True)),
            bool(plate_cfg.get("collision_invert_sdf", False)),
        )

    return {
        "rb": rb,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "move_steps": steps,
        "move_start_step": move_start_step,
        "axis": axis_name,
        "side": side,
        "scale": scale,
        "obj": plate_obj,
        "axis_original": plate_cfg.get("axis_original"),
        "side_original": plate_cfg.get("side_original"),
        "base_points_int": base_points_int,
        "base_faces": base_faces,
    }


def update_plate_state(plate_state, step: int):
    if plate_state is None:
        return
    rb = plate_state["rb"]
    move_steps = plate_state["move_steps"]
    move_start_step = plate_state["move_start_step"]
    if step < move_start_step:
        alpha = 0.0
    elif move_steps <= 0:
        alpha = 1.0
    else:
        alpha = min(1.0, max(0.0, (step - move_start_step) / float(move_steps)))
    pos = (1.0 - alpha) * plate_state["start_pos"] + alpha * plate_state["end_pos"]
    rb.setPosition(pos)
    rb.setPosition0(pos)
    rb.setLastPosition(pos)
    rb.setOldPosition(pos)
    rb.setVelocity(np.zeros(3, dtype=np.float64))
    rb.setVelocity0(np.zeros(3, dtype=np.float64))
    rb.setAcceleration(np.zeros(3, dtype=np.float64))
    rb.updateInverseTransformation()
    rb.getGeometry().updateMeshTransformation(rb.getPosition(), rb.getRotationMatrix())
    return pos


def get_plate_render_state(plate_state, transform: np.ndarray):
    if plate_state is None:
        return None
    rb = plate_state["rb"]
    pos = np.asarray(rb.getPosition(), dtype=np.float64)
    rot = np.asarray(rb.getRotationMatrix(), dtype=np.float64)
    local = plate_state["base_points_int"] - plate_state["start_pos"]
    points_int = local @ rot.T + pos
    points = points_int @ transform
    return {"points": points, "faces": plate_state["base_faces"]}


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
    timestep = sim.getTimeStep()

    plate_cfg = cfg.get("plate", {})
    use_analytic_box_contact = bool(plate_cfg.get("enabled", False) and plate_cfg.get("collider_type", "analytic_box") == "analytic_box")
    collision_detection = None
    if use_analytic_box_contact:
        collision_detection = pbd.DistanceFieldCollisionDetection()
        timestep.setCollisionDetection(model, collision_detection)
        collision_detection.setTolerance(float(cfg.get("collision_tolerance", 0.005)))

    body_test_mesh = bool(cfg.get("test_mesh", False) and not use_analytic_box_contact)
    body_generate_collision = bool(cfg.get("generate_collision_object", False) and not use_analytic_box_contact)
    collision_resolution = np.asarray(cfg.get("collision_resolution", [25, 25, 25]), dtype=np.uint32)

    tet_model = model.addTetModel(
        points_int.tolist(),
        tets.reshape(-1).tolist(),
        testMesh=body_test_mesh,
        generateCollisionObject=body_generate_collision,
        resolution=collision_resolution,
    )
    particles = model.getParticles()

    fixed_nodes = select_fixed_nodes(points_orig, cfg)
    for idx in fixed_nodes:
        particles.setMass(int(idx), 0.0)

    solid_method = int(cfg.get("solid_method", 3))
    stiffness = float(cfg.get("stiffness", 1e5))
    poisson_ratio = float(cfg.get("poisson_ratio", 0.3))
    volume_stiffness = float(cfg.get("volume_stiffness", stiffness))
    model.addSolidConstraints(
        tet_model,
        solid_method,
        stiffness,
        poisson_ratio,
        volume_stiffness,
        bool(cfg.get("normalize_stretch", False)),
        bool(cfg.get("normalize_shear", False)),
    )

    if collision_detection is not None:
        collision_detection.addCollisionObjectWithoutGeometry(
            0,
            pbd.CollisionObject.TetModelCollisionObjectType,
            particles,
            tet_model.getIndexOffset(),
            tet_model.getParticleMesh().numVertices(),
            bool(cfg.get("body_collision_test_mesh", True)),
        )

    max_velocity_iterations = int(cfg.get("max_velocity_iterations", 0))
    if plate_cfg.get("enabled", False) and max_velocity_iterations < 1:
        max_velocity_iterations = 5
    timestep.setValueUInt(pbd.TimeStepController.NUM_SUB_STEPS, int(cfg.get("substeps", 5)))
    timestep.setValueUInt(pbd.TimeStepController.MAX_ITERATIONS, int(cfg.get("max_iterations", 1)))
    timestep.setValueUInt(pbd.TimeStepController.MAX_ITERATIONS_V, max_velocity_iterations)
    timestep.setValueInt(
        pbd.TimeStepController.VELOCITY_UPDATE_METHOD,
        int(cfg.get("velocity_update_method", pbd.TimeStepController.ENUM_VUPDATE_FIRST_ORDER)),
    )
    pbd.TimeManager.getCurrent().setTimeStepSize(float(cfg.get("dt", 0.002)))

    plate_state = create_plate(model, collision_detection, points_int, plate_cfg, transform)
    if plate_state is not None:
        model.setContactStiffnessParticleRigidBody(float(cfg.get("contact_stiffness_particle_rigid", 1.0)))
        model.setContactStiffnessRigidBody(float(cfg.get("contact_stiffness_rigid", 1.0)))
        if timestep.getCollisionDetection() is not None:
            timestep.getCollisionDetection().setTolerance(float(cfg.get("collision_tolerance", 0.005)))

    render_cfg = cfg.get("render", {})
    record_dir = Path(render_cfg["record_dir"]) if render_cfg.get("record_dir") else None
    output_video = Path(render_cfg["record_video"]) if render_cfg.get("record_video") else None
    render_every = max(1, int(render_cfg.get("record_every", 1)))
    record_fps = int(render_cfg.get("record_fps", 30))

    surface_faces = build_faces(tet_model.getSurfaceMesh())
    plotter = None
    current_poly = None
    plate_poly = None
    camera_position = None
    if record_dir:
        ensure_empty_dir(record_dir)
        plotter, current_poly, plate_poly, _plate_start_poly, camera_position = make_plotter(
            points_orig,
            surface_faces,
            render_cfg,
            get_plate_render_state(plate_state, transform),
            get_plate_render_state(plate_state, transform),
        )

    steps = int(cfg.get("steps", 150))
    vel_damp = float(cfg.get("velocity_damping", 1.0))
    report_axis = cfg.get("report_axis", "z")
    report_index = AXIS_TO_INDEX[report_axis]
    fail_max_disp = cfg.get("fail_max_disp")
    fail_abs_coord = cfg.get("fail_abs_coord")

    failed = False
    fail_reason = None
    completed_steps = 0
    max_particle_rigid_contacts = 0
    max_rigid_body_contacts = 0
    max_particle_solid_contacts = 0
    for step in range(steps):
        update_plate_state(plate_state, step)
        timestep.step(model)
        if vel_damp < 1.0:
            for i in range(particles.size()):
                if particles.getMass(i) != 0.0:
                    particles.setVelocity(i, np.asarray(particles.getVelocity(i), dtype=np.float64) * vel_damp)

        points_step = np.asarray(particles.getVertices(), dtype=np.float64) @ transform
        disp_step = points_step - points_orig
        completed_steps = step + 1
        particle_rigid_contacts = len(model.getParticleRigidBodyContactConstraints())
        rigid_body_contacts = len(model.getRigidBodyContactConstraints())
        particle_solid_contacts = len(model.getParticleSolidContactConstraints())
        max_particle_rigid_contacts = max(max_particle_rigid_contacts, particle_rigid_contacts)
        max_rigid_body_contacts = max(max_rigid_body_contacts, rigid_body_contacts)
        max_particle_solid_contacts = max(max_particle_solid_contacts, particle_solid_contacts)
        if not np.isfinite(points_step).all():
            failed = True
            fail_reason = "non_finite_positions"
        elif fail_abs_coord is not None and float(np.abs(points_step).max()) > float(fail_abs_coord):
            failed = True
            fail_reason = f"abs_coord_exceeded:{float(np.abs(points_step).max())}"
        elif fail_max_disp is not None and float(np.linalg.norm(disp_step, axis=1).max()) > float(fail_max_disp):
            failed = True
            fail_reason = f"max_disp_exceeded:{float(np.linalg.norm(disp_step, axis=1).max())}"

        if record_dir and (step % render_every == 0 or step == steps - 1 or failed):
            current_poly.points = points_step
            if plate_poly is not None:
                plate_poly.points = get_plate_render_state(plate_state, transform)["points"]
            if render_cfg.get("lock_camera", True):
                plotter.camera_position = camera_position
            else:
                plotter.camera_position = camera_from_bounds(current_poly.bounds, render_cfg.get("camera_y_side", "max"))
            status = f"XPBD step {step + 1}/{steps}"
            if plate_state is not None:
                plate_pos = np.asarray(plate_state["rb"].getPosition(), dtype=np.float64)
                if plate_state.get("axis_original"):
                    plate_pos_orig = plate_pos @ transform
                    status += f" | plate_{plate_state['axis_original']}_orig={plate_pos_orig[AXIS_TO_INDEX[plate_state['axis_original']]]:.4f}"
                else:
                    status += f" | plate_{plate_state['axis']}={plate_pos[AXIS_TO_INDEX[plate_state['axis']]]:.4f}"
            if plate_state is not None:
                status += f" | pr_contacts={particle_rigid_contacts}"
            if failed:
                status += f" | FAILED: {fail_reason}"
            plotter.add_text(status, position="upper_left", font_size=10, color="black", name="status_text")
            plotter.show(screenshot=str(record_dir / f"frame_{step // render_every:05d}.png"), auto_close=False)

        if failed:
            break

    points_final = np.asarray(particles.getVertices(), dtype=np.float64) @ transform
    disp = points_final - points_orig
    axis_disp = disp[:, report_index]

    if plotter is not None:
        plotter.close()

    if record_dir and output_video:
        encode_video(record_dir, output_video, record_fps)

    result = {
        "tet_npz": str(tet_npz),
        "steps_requested": steps,
        "steps_completed": completed_steps,
        "dt": float(cfg.get("dt", 0.002)),
        "effective_max_velocity_iterations": int(max_velocity_iterations),
        "fixed_nodes": int(fixed_nodes.size),
        "max_disp": float(np.linalg.norm(disp, axis=1).max()),
        f"mean_{report_axis}_disp": float(axis_disp.mean()),
        f"min_{report_axis}_disp": float(axis_disp.min()),
        f"max_{report_axis}_disp": float(axis_disp.max()),
        "failed": failed,
        "fail_reason": fail_reason,
        "record_dir": str(record_dir) if record_dir else None,
        "record_video": str(output_video) if output_video else None,
        "max_particle_rigid_contacts": int(max_particle_rigid_contacts),
        "max_rigid_body_contacts": int(max_rigid_body_contacts),
        "max_particle_solid_contacts": int(max_particle_solid_contacts),
    }
    if plate_state is not None:
        plate_start_internal = plate_state["start_pos"]
        plate_end_internal = plate_state["end_pos"]
        plate_final_internal = np.asarray(plate_state["rb"].getPosition(), dtype=np.float64)
        result["plate_start_pos_internal"] = plate_start_internal.tolist()
        result["plate_end_pos_internal"] = plate_end_internal.tolist()
        result["plate_final_pos_internal"] = plate_final_internal.tolist()
        result["plate_start_pos_original"] = (plate_start_internal @ transform).tolist()
        result["plate_end_pos_original"] = (plate_end_internal @ transform).tolist()
        result["plate_final_pos_original"] = (plate_final_internal @ transform).tolist()

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
