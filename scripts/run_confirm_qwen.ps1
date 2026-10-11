# EXP-U confirmation readers for Qwen3-1.7B on the local GPU, after the Qwen joint runs and their development readers:
# the four fresh paradigms admitted by the source models and unused generic text, each run under its own gpu-0 lease.
# Finished runs are skipped.
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$R = 'D:\CCAD_Storage\runs\cross_seed_models'
$env:PYTHONPATH = 'E:\Projects\SAE_Lab\CCAD\src;D:\CCAD_Storage\environments\f4_sparse_overlay_v1'
Set-Location 'E:\Projects\SAE_Lab\CCAD\scripts'
$ended = { param($run) (Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"(PASS|FAIL)"') }
while (-not (& $ended "$R\CSM_UREADERS_QWEN17_s2_t1_ts2_v1_20261011")) { Start-Sleep -Seconds 60 }
foreach ($d in 's1_t2', 's2_t1') {
    foreach ($s in 0, 1, 2) {
        $run = "$R\CSM_CREADERS_QWEN17_${d}_ts${s}_v1_20261011"
        if ((Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"PASS"')) { continue }
        & $py $rm run --resource gpu-0 --project CCAD --task "EXP-U Qwen confirmation ${d}_ts$s" --wait-sec 3600 -- $py grammar_readers.py --config "..\configs\cross_seed_pilot\cureaders_qwen17_${d}_ts$s.json"
    }
}
Write-Output 'EXP-U Qwen confirmation finished'
