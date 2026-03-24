@echo off
setlocal
set EXE=C:\dev\softtissue\physx_direct\build\app_vs\Release\st_physx_gm_reference.exe
set BUNDLE=C:\dev\softtissue\physx_direct\data\gm_mri_p6_reference
set DUMP=C:\dev\softtissue\physx_direct\outputs\gm_mri_p6_reference_dump
set VIDEO=C:\dev\softtissue\videos\physx_gm_mri_p6_reference_latest.mp4
rmdir /s /q "%DUMP%" 2>nul
"%EXE%" --nodes "%BUNDLE%\nodes_f32.bin" --tets "%BUNDLE%\tets_u32.bin" --fixed "%BUNDLE%\fixed_nodes_u32.bin" --steps 180 --dt 0.0166667 --young 12000 --poisson 0.45 --density 1000 --gravity 0 0 -9.81 --dump-dir "%DUMP%" --dump-every 2
python C:\dev\softtissue\physx_direct\tools\render_physx_dump.py --bundle-dir "%BUNDLE%" --dump-dir "%DUMP%" --out-video "%VIDEO%" --fps 30 --camera oblique
