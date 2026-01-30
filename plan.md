Project Spec: Interactive Soft Tissue Deformation from MRI Meshes (Breast MRI)

Objective
Build a real-time (or near real-time with optimization) simulation pipeline that
transforms segmented breast MRI surface meshes into a tissue-aware, interactive
deformation system using scripted physics and/or learned physics (MeshGraphNet).

Constraints and Assumptions
- Target: real-time or near real-time interaction (aim for 30-60 FPS on a
  moderate mesh; provide fallback for offline accuracy runs).
- Inputs are professional-grade STLs, aligned and watertight. We will still
  verify alignment, units, and mesh health as a required validation step.
- Tissues are nested (e.g., tumor inside gland, implant inside fat).

Inputs
- STL meshes of segmented regions (breast MRI).
  Examples: MRI_skin.stl, MRI_fat.stl, MRI_gland.stl, MRI_implant.stl,
  MRI_tumor.stl
- All meshes are expected to be closed surfaces.

Key Decision: Scripted Physics vs MeshGraphNet
- Scripted physics: explicit physical model (mass-spring or FEM). Pros: direct
  control, interpretable parameters, no training data required. Cons: slower
  for large meshes without optimization.
- MeshGraphNet: learned surrogate trained on simulation data. Pros: very fast
  inference after training. Cons: requires a training dataset and can fail to
  generalize outside the training distribution.
Plan supports both; start with scripted physics to generate training data.

Pipeline Overview (Detailed)
0) Data Validation and Normalization (required)
   - Confirm units (mm vs m) and coordinate convention.
   - Check alignment between STLs (shared bounding box, overlap sanity checks).
   - Verify watertightness, remove self-intersections, and fix normals if needed.
   - Optional remesh/decimate for uniform triangle size (target 1-2 mm).
   - Output: cleaned STLs in a normalized coordinate frame.

1) Voxelization and Label Fusion (nested tissues)
   - Choose voxel spacing (1-2 mm typical for breast MRI).
   - Voxelize each STL into binary volumes.
   - Resolve nesting by explicit priority order or signed distance:
       Priority example: tumor > implant > gland > fat > skin.
   - Output: labeled_volume.npy or labeled_volume.nii.gz, label_map.json.

2) Tetrahedral Mesh Generation
   - Use skin (or union of all STLs) as outer boundary.
   - Generate tet mesh with TetWild, Gmsh, or iso2mesh.
   - Quality constraints: min dihedral, max aspect ratio; log and repair if
     elements are poor.
   - Assign material label per tet via voxel lookup at centroid or majority vote.
   - Output: tet_mesh.vtk or .npz with nodes, tets, and per-tet label.

3) Material Property Mapping
   - Define per-tissue material parameters (E, nu, density) in consistent units.
   - Compute per-tet mass from density and volume.
   - Output: materials.py or materials.json.

4) Graph Construction (MeshGraphNet-ready)
   - Nodes = vertices; edges = shared tet adjacency or 1-ring neighbors.
   - Node features = position, tissue ID, mass, optional stiffness.
   - Edge features = rest length, tissue transition flag.
   - Output: meshgraphnet_input.npz (or PyTorch Geometric dataset).

5) Simulation Modes
   A) Scripted Physics (baseline, data generation)
      - Choose solver: mass-spring or corotated linear FEM.
      - Use semi-implicit Euler with damping.
      - Boundary conditions: fixed chest wall region or floor constraint.
      - Gravity and user drag constraints supported.
      - Output: deformed node positions and surface mesh.

   B) Interactive Deformation (real-time UI)
      - User drags a surface node; apply displacement constraint.
      - Update deformation using solver or MeshGraphNet inference.
      - Real-time target: 30-60 FPS for a reduced mesh; 10-30 FPS for full mesh.

6) Performance Optimization
   - Precompute adjacency and factorization (if using linear FEM).
   - Use sparse solvers and GPU acceleration where possible.
   - Multi-resolution meshes: coarse for interaction, refined for export.

7) Visualization and Export
   - Show initial vs deformed mesh with tissue-based colors.
   - Toggle tumor visibility.
   - Export deformed surface as .ply or .vtk; optionally export tet mesh.

Validation and QA
- Verify no inverted tets (positive Jacobian).
- Check mass conservation and plausible displacement magnitude.
- Visual sanity checks on nested regions (tumor stays inside gland).

Folder Structure
project/
  data/
    stl_raw/
    stl_clean/
    voxelized_volume.npy
    tet_mesh.vtk
  graphs/
    meshgraphnet_input.npz
  sim/
    run_gravity_sim.py
    run_interactive_sim.py
  vis/
    viewer.py
  materials.py
  README.md

Milestones
1) Validation + voxelization + label fusion (nested tissues).
2) Tet meshing + material labeling + basic gravity sim.
3) Interactive drag UI with scripted physics (real-time target).
4) MeshGraphNet dataset generation and optional training.
5) MeshGraphNet inference integration for fast interaction.

Open Questions
- Final performance target and hardware (CPU vs GPU)?
- Preferred solver (mass-spring vs FEM)?
- Desired output mesh resolution for real-time mode?
