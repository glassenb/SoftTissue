# Simulation Run

Timestamp: 2026-01-31 09:25:59

## Command

`C:\dev\softtissue\sim\preview_fem.py --tet C:\dev\softtissue\Data\GM_STL\MRI\tet_mesh_gm_mri_p6.vtk --materials C:\dev\softtissue\materials.json --implicit --steps 30000 --dt 0.001 --damping 0.4 --stiffness-damping 0.05 --gravity-scale 1.0 --gravity-axis=-z --unit-scale 0.001 --fix-axis y --fix-side max --fix-mode plane --fix-extend-axis y --fix-extend-side max --fix-extend-frac 0.0 --record-dir C:\dev\softtissue\videos\frames_20260131_090611 --record-real-time --record-prefix frame --record-format png --record-video C:\dev\softtissue\videos\sim_20260131_090611_rt_plane_fix_colors.mp4 --record-fps 30 --stop-vel 0.5 --stop-steps 60 --out C:\dev\softtissue\videos\mesh_20260131_090611_rt_plane_fix_colors.vtk`

## Params

- E: 20000.0
- damping: 0.4
- decimate: 0.0
- default_E: 20000.0
- default_nu: 0.45
- default_rho: 1000.0
- dt: 0.001
- fix_axis: y
- fix_extend_axis: y
- fix_extend_frac: 0.0
- fix_extend_side: max
- fix_frac: 0.02
- fix_mode: plane
- fix_plane_tol: None
- fix_side: max
- fps: 30.0
- gravity_axis: -z
- gravity_ramp: 0
- gravity_scale: 1.0
- implicit: True
- keep_all: False
- keep_largest_component: False
- materials: C:\dev\softtissue\materials.json
- max_disp: 0.0
- max_volume: None
- min_comp_ratio: 0.0
- no_meshfix: False
- nu: 0.45
- out: C:\dev\softtissue\videos\mesh_20260131_090611_rt_plane_fix_colors.vtk
- quasi_static: False
- record_dir: C:\dev\softtissue\videos\frames_20260131_090611
- record_every: 1
- record_format: png
- record_fps: 30
- record_prefix: frame
- record_real_time: True
- record_video: C:\dev\softtissue\videos\sim_20260131_090611_rt_plane_fix_colors.mp4
- report_axis: None
- rho: 1000.0
- scale: 1.0
- screenshot: None
- show: False
- steps: 30000
- stiffness_damping: 0.05
- stl: None
- stop_steps: 60
- stop_vel: 0.5
- substeps: 1
- tet: C:\dev\softtissue\Data\GM_STL\MRI\tet_mesh_gm_mri_p6.vtk
- tet_npz: None
- unit_scale: 0.001

## Stats

- start: {'min': -456.7714538574219, 'max': -300.7714538574219, 'mean': -386.60084225753286, 'median': -390.7714538574219}
- end: {'min': -104927.17800934915, 'max': 4762.321645188918, 'mean': -50985.648926134614, 'median': -50453.34329053307}

## Materials

```json
{
  "default": {
    "name": "default",
    "E": 20000,
    "nu": 0.45,
    "rho": 1000
  },
  "materials": {
    "1": {
      "name": "skin",
      "E": 12000,
      "nu": 0.46,
      "rho": 1100
    },
    "2": {
      "name": "fat",
      "E": 2000,
      "nu": 0.49,
      "rho": 920
    },
    "3": {
      "name": "gland",
      "E": 8000,
      "nu": 0.48,
      "rho": 1050
    },
    "4": {
      "name": "implant",
      "E": 200000,
      "nu": 0.49,
      "rho": 980
    },
    "5": {
      "name": "tumor",
      "E": 50000,
      "nu": 0.49,
      "rho": 1050
    }
  }
}

```
