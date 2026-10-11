# EXP-U Qwen development readers of the s1_t2 direction on the local GPU, after the running s1_t2_ts0 reader; the s2_t1
# readers and the confirmation readers run on Mayo (scripts/mayo/run_qwen_readers.sh). Finished runs are skipped.
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$root = 'D:\CCAD_Storage\runs\cross_seed_models'
$env:PYTHONPATH = 'E:\Projects\SAE_Lab\CCAD\src;D:\CCAD_Storage\environments\f4_sparse_overlay_v1'
Set-Location 'E:\Projects\SAE_Lab\CCAD\scripts'
$ended = { param($run) (Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"(PASS|FAIL)"') }
$passed = { param($run) (Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"PASS"') }
while (-not (& $ended "$root\CSM_UREADERS_QWEN17_s1_t2_ts0_v1_20261011")) { Start-Sleep -Seconds 60 }
foreach ($name in 's1_t2_ts1', 's1_t2_ts2') {
    if (& $passed "$root\CSM_UREADERS_QWEN17_${name}_v1_20261011") { continue }
    & $py $rm run --resource gpu-0 --project CCAD --task "EXP-U Qwen readers $name" --wait-sec 3600 -- $py grammar_readers.py --config "..\configs\cross_seed_pilot\ureaders_qwen17_$name.json"
}
Write-Output 'EXP-U Qwen local readers finished'
