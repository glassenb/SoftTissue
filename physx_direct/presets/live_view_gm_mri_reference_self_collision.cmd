@echo off
setlocal
set EXE=C:\dev\softtissue\physx_direct\build\app_vs\Release\st_physx_gm_reference.exe
set BUNDLE=C:\dev\softtissue\physx_direct\data\gm_mri_p6_reference
set DUMP=C:\dev\softtissue\physx_direct\outputs\gm_mri_p6_reference_live_self_collision
python C:\dev\softtissue\physx_direct\tools\live_view_physx.py --bundle-dir "%BUNDLE%" --dump-dir "%DUMP%" --camera side_y -- -- "%EXE%" --nodes "%BUNDLE%\nodes_f32.bin" --tets "%BUNDLE%\tets_u32.bin" --fixed "%BUNDLE%\fixed_nodes_u32.bin" --steps 1800 --dt 0.0166667 --young 12000 --poisson 0.45 --density 1000 --gravity 0 0 -9.81 --self-collision --control-file "%DUMP%\control_gravity.txt" --dump-dir "%DUMP%" --dump-every 1
