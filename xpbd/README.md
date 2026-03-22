pyPBD prototype path.

Current scope:
- gravity-only tet simulation from existing `.npz` meshes
- fixed-node slabs or explicit fixed-node files
- headless frame dump + mp4 render

Important limitation:
- the Python binding does not expose a gravity-vector setter cleanly, so `run_pypbd.py` remaps coordinates internally so the requested gravity axis becomes the library's default `-y` gravity.
- current runner uses one global solid material setting for the whole tet mesh
- rigid plate contact and self-collision are not implemented in this path yet

Run:
`python C:\dev\softtissue\xpbd\run_pypbd.py --config C:\dev\softtissue\configs\xpbd\gm_mri_p6_regression.json`
