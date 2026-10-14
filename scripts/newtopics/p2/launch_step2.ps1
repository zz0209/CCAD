# Launch one phase of P2 step 2 (artifacts/newtopics/p2/STEP2_PROTOCOL.md) on the local GPU through the shared
# resource manager. The code is copied into the run folder first; add the phase's progress-viewer entry in the same step.
param([Parameter(Mandatory = $true)][ValidateSet(1, 2, 3)][int]$Phase)
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\circuit_local_v1\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$here = 'E:\Projects\SAE_Lab\CCAD\scripts\newtopics\p2'
$light = 'E:\Projects\SAE_Lab\CCAD\artifacts\newtopics\p2\P2_STEP2_v1'
$snap = "$light\code_snapshot\phase$Phase"
New-Item -ItemType Directory -Force $snap | Out-Null
Copy-Item "$here\*.py", "$here\*.ps1" $snap
$env:HF_HOME = 'D:\CCAD_Storage\hf_home'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUTF8 = '1'
& $py $rm run --resource gpu-0 --project CCAD --task "P2_STEP2_v1_phase$Phase" --wait-sec 0 --heartbeat-sec 20 -- `
    $py "$here\run_step2.py" --phase $Phase `
    --manifest 'D:\CCAD_Storage\newtopics\p2\step2\models_manifest.json' `
    --fineweb 'D:\CCAD_Storage\runs\cross_seed_models\natural_fineweb_slice.json' `
    --parquet 'D:\CCAD_Storage\hf_home\hub\datasets--HuggingFaceFW--fineweb\snapshots\9bb295ddab0e05d785b879661af7260fed5140fc\sample\10BT\000_00000.parquet' `
    --task_v1 'E:\Projects\SAE_Lab\CCAD\artifacts\newtopics\p2\P2_WORKSPACE_v1\task.json' `
    --task_v2 "$light\task_v2.json" `
    --concept_fit "$light\concept_fit.json" `
    --olmo_main_jlens 'D:\CCAD_Storage\newtopics\p2\P2_WORKSPACE_v1\checkpoints\olmo1b_main\jlens.pt' `
    --olmo_main_tl_full 'D:\CCAD_Storage\newtopics\p2\step1\P2_STEP1_v1\olmo1b_main_tl_full.pt' `
    --pythia410m_jlens 'D:\CCAD_Storage\newtopics\p2\P2_WORKSPACE_v1\checkpoints\pythia410m\jlens.pt' `
    --pythia28b_jlens 'D:\CCAD_Storage\newtopics\p2\step1\P2_STEP1_v1\pythia28b_jlens.pt' `
    --pythia28b_tl_full 'D:\CCAD_Storage\newtopics\p2\step1\P2_STEP1_v1\pythia28b_tl_full.pt' `
    --arrays 'D:\CCAD_Storage\newtopics\p2\step2\P2_STEP2_v1' `
    --light $light `
    --gpu_cap_gib 14
