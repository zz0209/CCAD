# Launch P2 step 1 (artifacts/newtopics/p2/STEP1_PROTOCOL.md) on the local GPU through the shared resource manager.
# Start only after the coordinating session releases the GPU; add the progress-viewer entry in the same step.
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\circuit_local_v1\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$here = 'E:\Projects\SAE_Lab\CCAD\scripts\newtopics\p2'
$env:HF_HOME = 'D:\CCAD_Storage\hf_home'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUTF8 = '1'
& $py $rm run --resource gpu-0 --project CCAD --task P2_STEP1_v1 --wait-sec 0 --heartbeat-sec 20 -- `
    $py "$here\run_step1.py" `
    --manifest 'D:\CCAD_Storage\newtopics\p2\step1\download_manifest.json' `
    --fineweb 'D:\CCAD_Storage\runs\cross_seed_models\natural_fineweb_slice.json' `
    --parquet 'D:\CCAD_Storage\hf_home\hub\datasets--HuggingFaceFW--fineweb\snapshots\9bb295ddab0e05d785b879661af7260fed5140fc\sample\10BT\000_00000.parquet' `
    --task 'E:\Projects\SAE_Lab\CCAD\artifacts\newtopics\p2\P2_WORKSPACE_v1\task.json' `
    --olmo_jlens 'D:\CCAD_Storage\newtopics\p2\P2_WORKSPACE_v1\checkpoints\olmo1b_main\jlens.pt' `
    --arrays 'D:\CCAD_Storage\newtopics\p2\step1\P2_STEP1_v1' `
    --light 'E:\Projects\SAE_Lab\CCAD\artifacts\newtopics\p2\P2_STEP1_v1' `
    --gpu_cap_gib 14
