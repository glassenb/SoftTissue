# Direct PhysX Spike

This workspace is isolated from the existing `pyPBD` and `SOFA` paths.

Purpose:
- validate direct PhysX 5 deformable-volume setup on this machine
- keep the vendor SDK and any local build patches contained under `C:\dev\softtissue\physx_direct`
- build a minimal headless spike against the old GM MRI reference mesh before touching the newer SA MRI case

Current layout:
- `third_party/PhysX` - vendored PhysX source
- `build/physx_sdk_vs` - generated PhysX SDK build tree
- `tools/export_physx_bundle.py` - converts the project's tet `.npz` files into a simple binary bundle for the C++ spike
- `app/main.cpp` - minimal direct PhysX deformable-volume runner

Build the app:

```powershell
$cmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
& $cmake -S C:\dev\softtissue\physx_direct -B C:\dev\softtissue\physx_direct\build\app_vs -G "Visual Studio 17 2022" -A x64
& $cmake --build C:\dev\softtissue\physx_direct\build\app_vs --config Release --target st_physx_gm_reference -j 4
```

Export the GM MRI reference bundle:

```powershell
python C:\dev\softtissue\physx_direct\tools\export_physx_bundle.py --npz C:\dev\softtissue\Data\GM_STL\MRI\tet_mesh_gm_mri_p6.npz --out-dir C:\dev\softtissue\physx_direct\data\gm_mri_p6_reference --unit-scale 0.001 --fix-axis y --fix-side max --fix-frac 0.3
```

Run the GM MRI reference spike:

```powershell
C:\dev\softtissue\physx_direct\build\app_vs\Release\st_physx_gm_reference.exe --nodes C:\dev\softtissue\physx_direct\data\gm_mri_p6_reference\nodes_f32.bin --tets C:\dev\softtissue\physx_direct\data\gm_mri_p6_reference\tets_u32.bin --fixed C:\dev\softtissue\physx_direct\data\gm_mri_p6_reference\fixed_nodes_u32.bin --steps 180 --dt 0.0166667 --young 12000 --poisson 0.45 --density 1000 --gravity 0 0 -9.81
```

Notes:
- This app is headless. It prints deformation stats to stdout.
- It uses the same tet mesh for simulation and collision for the first spike.
- Fixed nodes are applied through PhysX partially-kinematic nodes.
- It is intended to answer "does direct PhysX deformable volume behave sanely on our reference mesh?" before we add a plate/contact scene.


Live viewer:

```powershell
C:\dev\softtissue\physx_direct\presets\live_view_gm_mri_reference.cmd
```

Live viewer with self-collision enabled:

```powershell
C:\dev\softtissue\physx_direct\presets\live_view_gm_mri_reference_self_collision.cmd
```

Live viewer controls:
- `space` pause/resume display updates
- `r` rerun the sim from step 0
- `g` toggle gravity-drag mode
- `i`/`k` pitch gravity up/down
- `j`/`l` yaw gravity left/right
- `0` reset gravity to the preset value

Self-collision:
- The direct PhysX app can enable deformable-volume self collision with `--self-collision`.
- The app exposes `--self-collision-filter-distance` and `--self-collision-stress-tolerance`.
- If the filter distance is omitted, the app estimates it from the tet edge length and prints the effective value.

Vendor note:
- The local PhysX SDK build on this machine needed `patches/physx_cuda12_5_arch89.diff` applied under `third_party/PhysX` because CUDA 12.5 does not support the default Blackwell architecture list shipped in upstream PhysX 5.5.
