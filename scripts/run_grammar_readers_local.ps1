# Held-out grammar readers for the eight GPT-2 Medium directions on the local GPU, one run per direction; finished runs
# are skipped. Called through the shared resource manager with the gpu-0 lease.
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:\CCAD_Storage\environments\f4_sparse_overlay_v1;E:\Projects\SAE_Lab\CCAD\scripts;E:\Projects\SAE_Lab\CCAD\src'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
Set-Location E:\Projects\SAE_Lab\CCAD
foreach ($d in 's1_t2', 's1_t3', 's1_t4', 's1_t5', 's2_t3', 's3_t4', 's4_t5', 's5_t1') {
    $run = "D:\CCAD_Storage\runs\cross_seed_models\CSM_READERS_GPT2_${d}_v1_20261011"
    if ((Test-Path "$run\status.json") -and ((Get-Content "$run\status.json" -Raw) -match '"PASS"')) { continue }
    & $py scripts\grammar_readers.py --config "configs\cross_seed\readers\readers_gpt2_$d.json"
    if ($LASTEXITCODE -ne 0) { throw "grammar readers failed for $d" }
}
