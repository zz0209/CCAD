# EXP-U on the local GPU: the eight GPT-2 Medium directions in turn, each under its own gpu-0 lease. Finished runs are
# skipped.
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$env:PYTHONPATH = 'E:\Projects\SAE_Lab\CCAD\src;D:\CCAD_Storage\environments\f4_sparse_overlay_v1'
Set-Location 'E:\Projects\SAE_Lab\CCAD\scripts'
foreach ($d in 's1_t2', 's1_t3', 's1_t4', 's1_t5', 's2_t3', 's3_t4', 's4_t5', 's5_t1') {
    $run = "D:\CCAD_Storage\runs\cross_seed_models\CSM_U_GPT2_${d}_v1_20261011"
    if ((Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"PASS"')) { continue }
    & $py $rm run --resource gpu-0 --project CCAD --task "EXP-U GPT-2 $d" --wait-sec 1800 -- $py cross_seed_transfer.py --config "..\configs\cross_seed_pilot\u_gpt2_$d.json"
}
Write-Output 'EXP-U GPT-2 finished'
