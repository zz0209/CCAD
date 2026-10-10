$ErrorActionPreference = 'Stop'
$root = 'E:\Projects\SAE_Lab\CCAD'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH = "$root\src;D:/CCAD_Storage/environments/f4_sparse_overlay_v1"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$bulk = 'D:\CCAD_Storage\runs\naacl_sprint_20261010'
foreach ($seed in 2, 3, 4, 5) {
    $run = "$bulk\NAACL_E3_shift_use_seed${seed}_v1_20261010"
    if (Test-Path "$run\status.json") {
        $value = (Get-Content "$run\status.json" -Raw | ConvertFrom-Json).status
        if ($value -eq 'PASS') { continue }
        throw "existing run $run has status $value; inspect before rerunning"
    }
    & $py 'scripts\run_shift_retraining.py' --config "configs\naacl_sprint_20261010\e3_shift_use_seed$seed.json"
    if ($LASTEXITCODE -ne 0) { throw "E3 target $seed failed" }
}
