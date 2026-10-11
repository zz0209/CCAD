# EXP-U joint form for Qwen3-1.7B on the local GPU, after the GPT-2 EXP-U readers: a smoke run, the six runs, then
# their development readers, each under its own gpu-0 lease. Finished runs are skipped.
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$R = 'D:\CCAD_Storage\runs\cross_seed_models'
$env:PYTHONPATH = 'E:\Projects\SAE_Lab\CCAD\src;D:\CCAD_Storage\environments\f4_sparse_overlay_v1'
Set-Location 'E:\Projects\SAE_Lab\CCAD\scripts'
$ended = { param($run) (Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"(PASS|FAIL)"') }
$passed = { param($run) (Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"PASS"') }
while (-not (& $ended "$R\CSM_UREADERS_GPT2_s5_t1_v1_20261011")) { Start-Sleep -Seconds 60 }
if (-not (& $passed "$R\CSM_UJ_QWEN17_s1_t2_ts0_v1_20261011_SMOKE")) {
    & $py $rm run --resource gpu-0 --project CCAD --task 'EXP-U Qwen smoke' --wait-sec 3600 -- $py cross_seed_transfer.py --config '..\configs\cross_seed_pilot\uj_qwen17_s1_t2_ts0.json' --smoke
}
if (-not (& $passed "$R\CSM_UJ_QWEN17_s1_t2_ts0_v1_20261011_SMOKE")) { throw 'Qwen smoke failed' }
$runs = foreach ($d in 's1_t2', 's2_t1') { foreach ($s in 0, 1, 2) { "${d}_ts$s" } }
foreach ($r in $runs) {
    if (& $passed "$R\CSM_UJ_QWEN17_${r}_v1_20261011") { continue }
    & $py $rm run --resource gpu-0 --project CCAD --task "EXP-U Qwen $r" --wait-sec 3600 -- $py cross_seed_transfer.py --config "..\configs\cross_seed_pilot\uj_qwen17_$r.json"
}
foreach ($r in $runs) {
    if (& $passed "$R\CSM_UREADERS_QWEN17_${r}_v1_20261011") { continue }
    & $py $rm run --resource gpu-0 --project CCAD --task "EXP-U Qwen readers $r" --wait-sec 3600 -- $py grammar_readers.py --config "..\configs\cross_seed_pilot\ureaders_qwen17_$r.json"
}
Write-Output 'EXP-U Qwen finished'
