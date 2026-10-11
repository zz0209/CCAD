# EXP-U development readers on the local GPU, after every GPT-2 EXP-U run has passed. Finished runs are skipped.
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$env:PYTHONPATH = 'E:\Projects\SAE_Lab\CCAD\src;D:\CCAD_Storage\environments\f4_sparse_overlay_v1'
Set-Location 'E:\Projects\SAE_Lab\CCAD\scripts'
$dirs = 's1_t2', 's1_t3', 's1_t4', 's1_t5', 's2_t3', 's3_t4', 's4_t5', 's5_t1'
foreach ($d in $dirs) {
    $u = "D:\CCAD_Storage\runs\cross_seed_models\CSM_U_GPT2_${d}_v1_20261011\status.json"
    while (-not ((Test-Path $u) -and ((Get-Content $u -Raw) -match '"(PASS|FAIL)"'))) { Start-Sleep -Seconds 30 }
    if ((Get-Content $u -Raw) -notmatch '"PASS"') { throw "EXP-U run $d did not pass" }
}
foreach ($d in $dirs) {
    $run = "D:\CCAD_Storage\runs\cross_seed_models\CSM_UREADERS_GPT2_${d}_v1_20261011"
    if ((Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"PASS"')) { continue }
    & $py $rm run --resource gpu-0 --project CCAD --task "EXP-U readers $d" --wait-sec 1800 -- $py grammar_readers.py --config "..\configs\cross_seed_pilot\ureaders_gpt2_$d.json"
}
Write-Output 'EXP-U GPT-2 readers finished'
