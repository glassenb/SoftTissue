import argparse
import json
import os
import sys
import time
import subprocess

import numpy as np
import pyvista as pv
import trimesh as tm

try:
    import pymeshfix
except Exception:
    pymeshfix = None
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def load_surface(path, hole_size=1000.0, decimate=0.0, use_meshfix=True, keep_largest=True, min_comp_ratio=0.0):
    mesh = None
    try:
        tmesh = tm.load(path, force="mesh")
        if isinstance(tmesh, tm.Scene):
            tmesh = tm.util.concatenate(tmesh.dump())
        tmesh.remove_degenerate_faces()
        tmesh.remove_duplicate_faces()
        tmesh.remove_unreferenced_vertices()
        tmesh.merge_vertices()
        parts = tmesh.split(only_watertight=False)
        if len(parts) > 1:
            parts = sorted(parts, key=lambda p: p.volume if p.volume > 0 else p.area, reverse=True)
            if keep_largest:
                print(f"Found {len(parts)} components; keeping largest only.")
                tmesh = parts[0]
            elif min_comp_ratio > 0:
                maxv = parts[0].volume if parts[0].volume > 0 else parts[0].area
                keep = [p for p in parts if (p.volume if p.volume > 0 else p.area) >= maxv * min_comp_ratio]
                print(f"Found {len(parts)} components; keeping {len(keep)} by ratio >= {min_comp_ratio}.")
                tmesh = tm.util.concatenate(keep) if len(keep) > 1 else keep[0]
        faces = np.hstack(
            [
                np.full((len(tmesh.faces), 1), 3, dtype=np.int64),
                tmesh.faces.astype(np.int64),
            ]
        )
        mesh = pv.PolyData(tmesh.vertices, faces)
    except Exception:
        mesh = None

    if mesh is None:
        mesh = pv.read(path)
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()

    mesh = mesh.triangulate().clean()
    if use_meshfix and pymeshfix is not None:
        try:
            faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
            mf = pymeshfix.MeshFix(mesh.points, faces)
            mf.repair(verbose=False, joincomp=True, remove_smallest_components=True)
            fixed_faces = np.hstack(
                [
                    np.full((len(mf.f), 1), 3, dtype=np.int64),
                    mf.f.astype(np.int64),
                ]
            )
            mesh = pv.PolyData(mf.v, fixed_faces).triangulate().clean()
        except Exception:
            pass
    if decimate > 0:
        try:
            mesh = mesh.decimate_pro(target_reduction=decimate)
        except TypeError:
            try:
                mesh = mesh.decimate_pro(reduction=decimate)
            except Exception:
                pass
        except Exception:
            pass
    # Try to keep the largest connected component only.
    try:
        mesh = mesh.connectivity(extraction_mode="largest")
    except Exception:
        pass
    # Attempt basic repair steps for tetgen.
    try:
        mesh = mesh.fill_holes(hole_size)
    except Exception:
        pass
    try:
        mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    except Exception:
        pass
    return mesh


def tetrahedralize(surface, max_volume=None, mindihedral=10, minratio=1.5):
    try:
        import tetgen
    except Exception as exc:
        raise RuntimeError("tetgen is required for STL-based tetrahedralization.") from exc
    tg = tetgen.TetGen(surface)
    kwargs = {"order": 1, "mindihedral": mindihedral, "minratio": minratio}
    if max_volume is not None:
        kwargs["maxvolume"] = max_volume
    tg.tetrahedralize(**kwargs)
    return tg.grid


def elasticity_matrix(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.array(
        [
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ],
        dtype=float,
    )
    return D


def tet_shape_gradients(x):
    # x is (4, 3); rows are node coords
    M = np.ones((4, 4), dtype=float)
    M[:, 1:] = x
    invM = np.linalg.inv(M)
    grads = invM[1:, :]  # (3, 4); columns correspond to nodes
    return grads


def tet_volume(x):
    return abs(np.linalg.det(x[1:] - x[0])) / 6.0


def assemble_linear_fem(nodes, tets, E, nu, rho):
    n_nodes = nodes.shape[0]
    ndof = n_nodes * 3

    D = elasticity_matrix(E, nu)
    rows = []
    cols = []
    data = []

    masses = np.zeros(n_nodes, dtype=float)

    for tet in tets:
        x = nodes[tet]
        grads = tet_shape_gradients(x)
        V = tet_volume(x)
        if V <= 0:
            continue

        B = np.zeros((6, 12), dtype=float)
        for i in range(4):
            bx, by, bz = grads[:, i]
            j = i * 3
            B[0, j] = bx
            B[1, j + 1] = by
            B[2, j + 2] = bz
            B[3, j] = by
            B[3, j + 1] = bx
            B[4, j + 1] = bz
            B[4, j + 2] = by
            B[5, j] = bz
            B[5, j + 2] = bx

        Ke = (B.T @ D @ B) * V

        dofs = np.empty(12, dtype=int)
        dofs[0::3] = tet * 3
        dofs[1::3] = tet * 3 + 1
        dofs[2::3] = tet * 3 + 2

        for a in range(12):
            for b in range(12):
                rows.append(dofs[a])
                cols.append(dofs[b])
                data.append(Ke[a, b])

        masses[tet] += (rho * V) / 4.0

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsc()
    return K, masses


def assemble_linear_fem_materials(nodes, tets, labels, materials, default_props):
    n_nodes = nodes.shape[0]
    ndof = n_nodes * 3

    rows = []
    cols = []
    data = []
    masses = np.zeros(n_nodes, dtype=float)

    D_cache = {}

    for i, tet in enumerate(tets):
        x = nodes[tet]
        grads = tet_shape_gradients(x)
        V = tet_volume(x)
        if V <= 0:
            continue

        label = int(labels[i]) if labels is not None else 0
        props = materials.get(label, default_props)
        E = float(props["E"])
        nu = float(props["nu"])
        rho = float(props["rho"])

        key = (E, nu)
        D = D_cache.get(key)
        if D is None:
            D = elasticity_matrix(E, nu)
            D_cache[key] = D

        B = np.zeros((6, 12), dtype=float)
        for j in range(4):
            bx, by, bz = grads[:, j]
            k = j * 3
            B[0, k] = bx
            B[1, k + 1] = by
            B[2, k + 2] = bz
            B[3, k] = by
            B[3, k + 1] = bx
            B[4, k + 1] = bz
            B[4, k + 2] = by
            B[5, k] = bz
            B[5, k + 2] = bx

        Ke = (B.T @ D @ B) * V

        dofs = np.empty(12, dtype=int)
        dofs[0::3] = tet * 3
        dofs[1::3] = tet * 3 + 1
        dofs[2::3] = tet * 3 + 2

        for a in range(12):
            for b in range(12):
                rows.append(dofs[a])
                cols.append(dofs[b])
                data.append(Ke[a, b])

        masses[tet] += (rho * V) / 4.0

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsc()
    return K, masses


def load_materials(path, default_E, default_nu, default_rho):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    default_props = {
        "E": float(default_E),
        "nu": float(default_nu),
        "rho": float(default_rho),
    }

    if "default" in data:
        d = data["default"]
        default_props["E"] = float(d.get("E", default_props["E"]))
        default_props["nu"] = float(d.get("nu", default_props["nu"]))
        default_props["rho"] = float(d.get("rho", default_props["rho"]))

    mats = data.get("materials", data)
    materials = {}
    for key, props in mats.items():
        if key == "default":
            continue
        try:
            label = int(key)
        except Exception:
            continue
        materials[label] = {
            "E": float(props.get("E", default_props["E"])),
            "nu": float(props.get("nu", default_props["nu"])),
            "rho": float(props.get("rho", default_props["rho"])),
        }
    return materials, default_props


def build_gravity_forces(masses, gravity_axis="-z"):
    axis_map = {"x": 0, "y": 1, "z": 2}
    sign = -1.0 if gravity_axis.startswith("-") else 1.0
    axis = gravity_axis.lstrip("-")
    if axis not in axis_map:
        raise ValueError(f"Invalid gravity axis: {gravity_axis}")
    idx = axis_map[axis]

    g = 9.81
    ndof = masses.shape[0] * 3
    f = np.zeros(ndof, dtype=float)
    f[idx::3] = masses * sign * g
    return f


def axis_stats(points, axis):
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    vals = points[:, idx]
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
    }


def axis_from_gravity(gravity_axis):
    axis = gravity_axis.lstrip("-")
    return axis if axis in {"x", "y", "z"} else "z"


def setup_plotter(show, surface, grid, record_dir=None):
    if not show and not record_dir:
        return None
    offscreen = bool(record_dir) and not show
    plotter = pv.Plotter(off_screen=offscreen)
    plotter.add_mesh(surface, color="lightgray", opacity=0.25)
    plotter.add_mesh(grid, color="tomato", opacity=0.6, show_edges=False)
    plotter.add_axes()
    # Default camera: from +Y looking back at the object.
    try:
        bounds = grid.bounds
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])
        size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        pos = (cx, bounds[3] + 1.5 * size, cz)
        plotter.camera_position = [pos, (cx, cy, cz), (0, 0, 1)]
    except Exception:
        pass
    if show:
        plotter.show(auto_close=False, interactive_update=True)
    return plotter


def record_frame(plotter, record_dir, prefix, frame_idx, fmt="png"):
    if plotter is None or record_dir is None:
        return
    fname = f"{prefix}_{frame_idx:05d}.{fmt}"
    path = os.path.join(record_dir, fname)
    try:
        plotter.render()
    except Exception:
        pass
    try:
        plotter.screenshot(path)
    except Exception:
        plotter.show(screenshot=path)


def encode_video(record_dir, prefix, fmt, fps, output_path):
    if record_dir is None or output_path is None:
        return
    input_pattern = os.path.join(record_dir, f"{prefix}_%05d.{fmt}")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=False)


def load_materials_text(path):
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def write_run_metadata(path_base, args, start_stats, end_stats):
    if not path_base:
        return
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "start_stats": start_stats,
        "end_stats": end_stats,
        "materials_text": load_materials_text(args.materials),
    }
    json_path = f"{path_base}.json"
    md_path = f"{path_base}.md"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Simulation Run\n\n")
            f.write(f"Timestamp: {meta['timestamp']}\n\n")
            f.write("## Command\n\n")
            f.write(f"`{meta['command']}`\n\n")
            f.write("## Params\n\n")
            for k, v in sorted(meta["args"].items()):
                f.write(f"- {k}: {v}\n")
            f.write("\n## Stats\n\n")
            f.write(f"- start: {start_stats}\n")
            f.write(f"- end: {end_stats}\n")
            if meta["materials_text"]:
                f.write("\n## Materials\n\n")
                f.write("```json\n")
                f.write(meta["materials_text"])
                f.write("\n```\n")
    except Exception:
        pass


def select_fixed_dofs(
    nodes,
    fix_axis="x",
    fix_frac=0.02,
    fix_side="min",
    fix_extend_axis=None,
    fix_extend_frac=0.0,
    fix_extend_side="max",
):
    axis_map = {"x": 0, "y": 1, "z": 2}
    if fix_axis not in axis_map:
        raise ValueError(f"Invalid fix axis: {fix_axis}")
    if fix_side not in {"min", "max"}:
        raise ValueError(f"Invalid fix side: {fix_side}")
    idx = axis_map[fix_axis]

    mins = nodes[:, idx].min()
    maxs = nodes[:, idx].max()
    tol = (maxs - mins) * fix_frac
    if fix_side == "min":
        fixed_nodes = np.where(nodes[:, idx] <= mins + tol)[0]
    else:
        fixed_nodes = np.where(nodes[:, idx] >= maxs - tol)[0]

    if fix_extend_axis:
        if fix_extend_axis not in axis_map:
            raise ValueError(f"Invalid fix extend axis: {fix_extend_axis}")
        if fix_extend_side not in {"min", "max"}:
            raise ValueError(f"Invalid fix extend side: {fix_extend_side}")
        eidx = axis_map[fix_extend_axis]
        emins = nodes[:, eidx].min()
        emaxs = nodes[:, eidx].max()
        etol = (emaxs - emins) * fix_extend_frac
        if etol > 0:
            if fix_extend_side == "min":
                extra = np.where(nodes[:, eidx] <= emins + etol)[0]
            else:
                extra = np.where(nodes[:, eidx] >= emaxs - etol)[0]
            fixed_nodes = np.unique(np.concatenate([fixed_nodes, extra]))

    fixed_dofs = np.empty(len(fixed_nodes) * 3, dtype=int)
    fixed_dofs[0::3] = fixed_nodes * 3
    fixed_dofs[1::3] = fixed_nodes * 3 + 1
    fixed_dofs[2::3] = fixed_nodes * 3 + 2
    return fixed_dofs


def solve_static(K, f, fixed_dofs):
    ndof = K.shape[0]
    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, fixed_dofs)

    K_ff = K[free, :][:, free]
    f_f = f[free]

    u = np.zeros(ndof, dtype=float)
    u_f = spla.spsolve(K_ff, f_f)
    u[free] = u_f
    return u


def make_static_solver(K, fixed_dofs):
    ndof = K.shape[0]
    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, fixed_dofs)
    K_ff = K[free, :][:, free]
    solver = spla.factorized(K_ff)
    return solver, free


def main():
    parser = argparse.ArgumentParser(description="Preview a linear FEM deformation.")
    parser.add_argument("--stl", help="Path to outer surface STL.")
    parser.add_argument("--tet", help="Path to tet mesh VTK.")
    parser.add_argument("--tet-npz", help="Path to tet mesh NPZ (nodes, tets).")
    parser.add_argument("--out", default="preview_deformed.vtk", help="Output VTK path.")
    parser.add_argument("--max-volume", type=float, default=None, help="Tet max volume.")
    parser.add_argument("--E", type=float, default=2.0e4, help="Young's modulus.")
    parser.add_argument("--nu", type=float, default=0.45, help="Poisson ratio.")
    parser.add_argument("--rho", type=float, default=1000.0, help="Density.")
    parser.add_argument("--materials", default=None, help="JSON materials file (per label).")
    parser.add_argument("--default-E", type=float, default=2.0e4, help="Default Young's modulus.")
    parser.add_argument("--default-nu", type=float, default=0.45, help="Default Poisson ratio.")
    parser.add_argument("--default-rho", type=float, default=1000.0, help="Default density.")
    parser.add_argument("--fix-axis", default="x", choices=["x", "y", "z"])
    parser.add_argument("--fix-frac", type=float, default=0.02)
    parser.add_argument("--fix-side", default="min", choices=["min", "max"], help="Clamp min or max side.")
    parser.add_argument("--fix-extend-axis", default=None, choices=["x", "y", "z"])
    parser.add_argument("--fix-extend-frac", type=float, default=0.0, help="Extra clamp band size.")
    parser.add_argument(
        "--fix-extend-side",
        default="max",
        choices=["min", "max"],
        help="Clamp extra band on min or max side.",
    )
    parser.add_argument("--gravity-axis", default="-z", help="e.g. -z, +y")
    parser.add_argument("--scale", type=float, default=1.0, help="Displacement scale.")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale coords to meters (e.g. 0.001 for mm).")
    parser.add_argument("--steps", type=int, default=0, help="Dynamic steps (0 = static solve).")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for dynamics.")
    parser.add_argument("--damping", type=float, default=0.02, help="Velocity damping coefficient.")
    parser.add_argument("--stiffness-damping", type=float, default=0.0, help="Stiffness-proportional damping.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS for realtime preview.")
    parser.add_argument("--quasi-static", action="store_true", help="Solve static K u = f each frame.")
    parser.add_argument("--implicit", action="store_true", help="Use implicit (backward Euler) dynamics.")
    parser.add_argument("--substeps", type=int, default=1, help="Substeps per frame.")
    parser.add_argument("--gravity-scale", type=float, default=1.0, help="Gravity multiplier.")
    parser.add_argument("--gravity-ramp", type=int, default=0, help="Ramp gravity over N steps.")
    parser.add_argument("--max-disp", type=float, default=0.0, help="Clamp per-step displacement (units).")
    parser.add_argument("--report-axis", default=None, choices=["x", "y", "z"], help="Axis for stats.")
    parser.add_argument("--keep-largest-component", action="store_true", help="Keep only largest connected mesh.")
    parser.add_argument("--record-dir", default=None, help="Directory to save frames (PNG).")
    parser.add_argument("--record-every", type=int, default=1, help="Record every N steps.")
    parser.add_argument("--record-prefix", default="frame", help="Frame file prefix.")
    parser.add_argument("--record-format", default="png", choices=["png", "jpg"], help="Frame image format.")
    parser.add_argument("--record-video", default=None, help="Output video path (mp4).")
    parser.add_argument("--record-fps", type=int, default=30, help="FPS for encoded video.")
    parser.add_argument("--record-real-time", action="store_true", help="Match recorded frame rate to dt.")
    parser.add_argument("--decimate", type=float, default=0.0, help="Target reduction 0-1.")
    parser.add_argument("--no-meshfix", action="store_true", help="Skip meshfix repair.")
    parser.add_argument("--keep-all", action="store_true", help="Keep all connected components.")
    parser.add_argument("--min-comp-ratio", type=float, default=0.0, help="Keep components >= ratio of largest.")
    parser.add_argument("--screenshot", default=None, help="PNG path for preview.")
    parser.add_argument("--show", action="store_true", help="Show interactive plot.")
    args = parser.parse_args()

    labels = None
    if args.tet or args.tet_npz:
        if args.tet and not os.path.isfile(args.tet):
            print(f"Tet mesh not found: {args.tet}")
            return 1
        if args.tet_npz and not os.path.isfile(args.tet_npz):
            print(f"Tet NPZ not found: {args.tet_npz}")
            return 1

        if args.tet:
            grid = pv.read(args.tet)
        else:
            data = np.load(args.tet_npz)
            nodes = data["nodes"].astype(float)
            tets = data["tets"].astype(int)
            labels = data["labels"].astype(int) if "labels" in data else None
            cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64), tets]).reshape(-1)
            celltypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
            grid = pv.UnstructuredGrid(cells, celltypes, nodes)
        if args.keep_largest_component:
            try:
                grid = grid.connectivity(extraction_mode="largest")
            except Exception:
                pass
        cells = grid.cells.reshape(-1, 5)
        tets = cells[:, 1:5].astype(int)
        nodes = grid.points.astype(float)
        if labels is None:
            if "label" in grid.cell_data:
                labels = np.asarray(grid.cell_data["label"]).astype(int)
            elif "Label" in grid.cell_data:
                labels = np.asarray(grid.cell_data["Label"]).astype(int)
        surface = grid.extract_surface().triangulate()
    else:
        if not args.stl or not os.path.isfile(args.stl):
            print("Provide --stl or --tet/--tet-npz.")
            return 1
        surface = load_surface(
            args.stl,
            decimate=args.decimate,
            use_meshfix=not args.no_meshfix,
            keep_largest=not args.keep_all,
            min_comp_ratio=args.min_comp_ratio,
        )
        grid = tetrahedralize(surface, max_volume=args.max_volume)

        cells = grid.cells.reshape(-1, 5)
        tets = cells[:, 1:5].astype(int)
        nodes = grid.points.astype(float)

    print(f"Nodes: {nodes.shape[0]}, Tets: {tets.shape[0]}")

    unit_scale = float(args.unit_scale)
    nodes_phys = nodes * unit_scale
    if args.materials:
        if not os.path.isfile(args.materials):
            print(f"Materials file not found: {args.materials}")
            return 1
        materials, default_props = load_materials(args.materials, args.default_E, args.default_nu, args.default_rho)
        if labels is None:
            print("Warning: no per-tet labels found; using default material.")
            labels = np.zeros(len(tets), dtype=int)
        K, masses = assemble_linear_fem_materials(nodes_phys, tets, labels, materials, default_props)
    else:
        K, masses = assemble_linear_fem(nodes_phys, tets, args.E, args.nu, args.rho)
    f = build_gravity_forces(masses, args.gravity_axis) * float(args.gravity_scale)
    fixed_dofs = select_fixed_dofs(
        nodes,
        args.fix_axis,
        args.fix_frac,
        args.fix_side,
        args.fix_extend_axis,
        args.fix_extend_frac,
        args.fix_extend_side,
    )
    report_axis = args.report_axis or axis_from_gravity(args.gravity_axis)
    start_stats = axis_stats(nodes, report_axis)
    print(f"Start {report_axis}-stats: {start_stats}")

    if args.steps > 0:
        ndof = nodes.shape[0] * 3
        if args.record_dir:
            os.makedirs(args.record_dir, exist_ok=True)
        plotter = setup_plotter(args.show, surface, grid, args.record_dir)

        last_time = time.time()
        if args.quasi_static:
            solver, free = make_static_solver(K, fixed_dofs)
            u = np.zeros(ndof, dtype=float)
            frame_idx = 0
            for step in range(args.steps):
                if args.gravity_ramp and args.gravity_ramp > 0:
                    ramp = min(1.0, (step + 1) / float(args.gravity_ramp))
                else:
                    ramp = 1.0
                f_step = f * ramp
                u[:] = 0.0
                u[free] = solver(f_step[free])

                if plotter is not None:
                    u_world = u.reshape(-1, 3) / unit_scale
                    grid.points = nodes + u_world * args.scale
                    plotter.update_coordinates(grid.points, mesh=grid, render=False)
                    if args.show:
                        plotter.update()
                    if args.record_dir and (step % max(1, args.record_every) == 0):
                        record_frame(plotter, args.record_dir, args.record_prefix, frame_idx, args.record_format)
                        frame_idx += 1
                    if args.fps > 0:
                        now = time.time()
                        delay = max(0.0, (1.0 / args.fps) - (now - last_time))
                        if delay > 0:
                            time.sleep(delay)
                        last_time = time.time()
        elif args.implicit:
            u = np.zeros(ndof, dtype=float)
            v = np.zeros(ndof, dtype=float)
            m_dof = np.repeat(masses, 3)
            m_dof[m_dof == 0] = 1.0
            dt = float(args.dt)
            record_every = args.record_every
            if args.record_real_time:
                record_every = max(1, int(round(1.0 / (max(1e-8, dt) * float(args.record_fps)))))
            # Rayleigh damping: C = alpha*M + beta*K
            alpha = float(args.damping)
            beta = float(args.stiffness_damping)
            c_dof = alpha * m_dof
            free = np.setdiff1d(np.arange(ndof), fixed_dofs)
            # Build and factor implicit system once (constant dt and damping)
            A_diag = m_dof + dt * c_dof
            K_scaled = (dt * dt + dt * beta) * K
            A = sp.diags(A_diag, format="csc") + K_scaled
            A_ff = A[free, :][:, free]
            solver = spla.factorized(A_ff)
            frame_idx = 0

            for step in range(args.steps):
                if args.gravity_ramp and args.gravity_ramp > 0:
                    ramp = min(1.0, (step + 1) / float(args.gravity_ramp))
                else:
                    ramp = 1.0
                f_step = f * ramp

                rhs = m_dof * v + dt * (f_step - K @ u)
                v_new = np.zeros_like(v)
                v_new[free] = solver(rhs[free])
                v = v_new
                u += dt * v
                u[fixed_dofs] = 0.0
                v[fixed_dofs] = 0.0

                if plotter is not None:
                    u_world = u.reshape(-1, 3) / unit_scale
                    grid.points = nodes + u_world * args.scale
                    plotter.update_coordinates(grid.points, mesh=grid, render=False)
                    if args.show:
                        plotter.update()
                    if args.record_dir and (step % max(1, record_every) == 0):
                        record_frame(plotter, args.record_dir, args.record_prefix, frame_idx, args.record_format)
                        frame_idx += 1
                    if args.fps > 0:
                        now = time.time()
                        delay = max(0.0, (1.0 / args.fps) - (now - last_time))
                        if delay > 0:
                            time.sleep(delay)
                        last_time = time.time()
        else:
            u = np.zeros(ndof, dtype=float)
            v = np.zeros(ndof, dtype=float)
            m_dof = np.repeat(masses, 3)
            m_dof[m_dof == 0] = 1.0
            dt = float(args.dt)
            substeps = max(1, int(args.substeps))
            max_disp = float(args.max_disp) * unit_scale
            record_every = args.record_every
            if args.record_real_time:
                record_every = max(1, int(round(1.0 / (max(1e-8, dt) * float(args.record_fps)))))
            frame_idx = 0
            for step in range(args.steps):
                if args.gravity_ramp and args.gravity_ramp > 0:
                    ramp = min(1.0, (step + 1) / float(args.gravity_ramp))
                else:
                    ramp = 1.0
                f_step = f * ramp

                for _ in range(substeps):
                    Ku = K @ u
                    a = (f_step - Ku - args.damping * v) / m_dof
                    v += dt * a
                    if max_disp > 0:
                        v3 = v.reshape(-1, 3)
                        speed = np.linalg.norm(v3, axis=1)
                        max_vel = max_disp / max(dt, 1e-8)
                        scale = np.minimum(1.0, max_vel / (speed + 1e-12))
                        v3 *= scale[:, None]
                        v = v3.reshape(-1)
                    u += dt * v
                    u[fixed_dofs] = 0.0
                    v[fixed_dofs] = 0.0

                if plotter is not None:
                    u_world = u.reshape(-1, 3) / unit_scale
                    grid.points = nodes + u_world * args.scale
                    plotter.update_coordinates(grid.points, mesh=grid, render=False)
                    if args.show:
                        plotter.update()
                    if args.record_dir and (step % max(1, record_every) == 0):
                        record_frame(plotter, args.record_dir, args.record_prefix, frame_idx, args.record_format)
                        frame_idx += 1
                    if args.fps > 0:
                        now = time.time()
                        delay = max(0.0, (1.0 / args.fps) - (now - last_time))
                        if delay > 0:
                            time.sleep(delay)
                        last_time = time.time()

        u_world = u.reshape(-1, 3) / unit_scale
        deformed = nodes + u_world * args.scale
        grid.points = deformed
        surface_def = grid.extract_surface().triangulate()
        end_stats = axis_stats(deformed, report_axis)
        print(f"End {report_axis}-stats: {end_stats}")
        print(f"Delta {report_axis}-mean: {end_stats['mean'] - start_stats['mean']:.3f}")
        surface_def.save(args.out)
        print(f"Saved: {args.out}")

        if args.screenshot:
            plotter = plotter or pv.Plotter(off_screen=True)
            if plotter is None or plotter.renderer is None:
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(surface, color="lightgray", opacity=0.25)
                plotter.add_mesh(surface_def, color="tomato", opacity=0.75)
            plotter.show(screenshot=args.screenshot)
            print(f"Screenshot: {args.screenshot}")

        if args.show and plotter is not None:
            print("Simulation complete. Close the window to exit.")
            plotter.show()
        meta_base = None
        if args.record_video:
            meta_base = os.path.splitext(args.record_video)[0]
            encode_video(
                args.record_dir,
                args.record_prefix,
                args.record_format,
                args.record_fps,
                args.record_video,
            )
        elif args.record_dir:
            meta_base = os.path.join(args.record_dir, args.record_prefix)
        if meta_base:
            write_run_metadata(meta_base, args, start_stats, end_stats)
    else:
        u = solve_static(K, f, fixed_dofs).reshape(-1, 3)
        u_world = u / unit_scale
        deformed = nodes + u_world * args.scale
        end_stats = axis_stats(deformed, report_axis)
        print(f"End {report_axis}-stats: {end_stats}")
        print(f"Delta {report_axis}-mean: {end_stats['mean'] - start_stats['mean']:.3f}")

        grid_def = grid.copy()
        grid_def.points = deformed
        surface_def = grid_def.extract_surface().triangulate()

        surface_def.save(args.out)
        print(f"Saved: {args.out}")

        if args.screenshot or args.show:
            plotter = pv.Plotter(off_screen=bool(args.screenshot) and not args.show)
            plotter.add_mesh(surface, color="lightgray", opacity=0.35)
            plotter.add_mesh(surface_def, color="tomato", opacity=0.75)
            plotter.add_axes()
            if args.screenshot:
                plotter.show(screenshot=args.screenshot)
                print(f"Screenshot: {args.screenshot}")
            elif args.show:
                plotter.show()
        if args.record_video:
            write_run_metadata(os.path.splitext(args.record_video)[0], args, start_stats, end_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
