$ErrorActionPreference = 'Stop'
$root = 'E:\Projects\SAE_Lab\CCAD'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH = "$root\src;D:/CCAD_Storage/environments/f4_sparse_overlay_v1"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$bulk = 'D:\CCAD_Storage\runs\naacl_sprint_20261010'
function Test-Done([string]$run) {
    $status = "$bulk\$run\status.json"
    if (-not (Test-Path $status)) { return $false }
    $value = (Get-Content $status -Raw | ConvertFrom-Json).status
    if ($value -eq 'PASS') { return $true }
    throw "existing run $run has status $value; inspect before rerunning"
}
foreach ($trainingSeed in 43, 44) {
    foreach ($target in 2, 3, 4, 5) {
        $train = "NAACL_E14_shift_program_t${target}_ts${trainingSeed}_v1_20261010"
        if (-not (Test-Done $train)) {
            & $py 'scripts\train_shift_program.py' --config "configs\naacl_sprint_20261010\e14_shift_program_t${target}_ts${trainingSeed}.json"
            if ($LASTEXITCODE -ne 0) { throw "E14 training failed for $train" }
        }
        $eval = "NAACL_E14_shift_eval_t${target}_ts${trainingSeed}_v1_20261010"
        if (-not (Test-Done $eval)) {
            & $py 'scripts\run_shift_transfer.py' --config "configs\naacl_sprint_20261010\e14_shift_eval_t${target}_ts${trainingSeed}.json"
            if ($LASTEXITCODE -ne 0) { throw "E14 evaluation failed for $eval" }
        }
    }
}
