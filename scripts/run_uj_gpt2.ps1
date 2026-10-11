# EXP-U joint form on the local GPU: the eight GPT-2 Medium directions, then their development readers, each under its
# own gpu-0 lease. Finished runs are skipped.
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$env:PYTHONPATH = 'E:\Projects\SAE_Lab\CCAD\src;D:\CCAD_Storage\environments\f4_sparse_overlay_v1'
Set-Location 'E:\Projects\SAE_Lab\CCAD\scripts'
$dirs = 's1_t2', 's1_t3', 's1_t4', 's1_t5', 's2_t3', 's3_t4', 's4_t5', 's5_t1'
$passed = { param($run) (Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"PASS"') }
foreach ($d in $dirs) {
    if (& $passed "D:\CCAD_Storage\runs\cross_seed_models\CSM_UJ_GPT2_${d}_v1_20261011") { continue }
    & $py $rm run --resource gpu-0 --project CCAD --task "EXP-U joint $d" --wait-sec 3600 -- $py cross_seed_transfer.py --config "..\configs\cross_seed_pilot\uj_gpt2_$d.json"
}
foreach ($d in $dirs) {
    if (& $passed "D:\CCAD_Storage\runs\cross_seed_models\CSM_UREADERS_GPT2_${d}_v1_20261011") { continue }
    & $py $rm run --resource gpu-0 --project CCAD --task "EXP-U readers $d" --wait-sec 3600 -- $py grammar_readers.py --config "..\configs\cross_seed_pilot\ureaders_gpt2_$d.json"
}
Write-Output 'EXP-U joint GPT-2 finished'
