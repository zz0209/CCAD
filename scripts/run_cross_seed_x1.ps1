$ErrorActionPreference = 'Stop'
$root = 'E:\Projects\SAE_Lab\CCAD'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH = "$root\src;D:/CCAD_Storage/environments/f4_sparse_overlay_v1"
$bulk = 'D:\CCAD_Storage\runs\cross_seed_models'
foreach ($target in 2, 3, 4, 5) {
    $run = "CSM_X1_gpt2_s1_t${target}_v1_20261010"
    $status = "$bulk\$run\status.json"
    if (Test-Path $status) {
        $value = (Get-Content $status -Raw | ConvertFrom-Json).status
        if ($value -eq 'PASS') { continue }
        throw "existing run $run has status $value; inspect before rerunning"
    }
    & $py 'scripts\cross_seed_transfer.py' --config "configs\cross_seed\x1_gpt2_t$target.json"
    if ($LASTEXITCODE -ne 0) { throw "X1 failed for target $target" }
}
