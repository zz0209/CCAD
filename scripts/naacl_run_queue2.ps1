$ErrorActionPreference = 'Stop'
$root = 'E:\Projects\SAE_Lab\CCAD'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH = "$root\src;D:/CCAD_Storage/environments/f4_sparse_overlay_v1"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$bulk = 'D:\CCAD_Storage\runs\naacl_sprint_20261010'
function Done($runId) {
    $status = "$bulk\$runId\status.json"
    if (-not (Test-Path $status)) { return $false }
    $value = (Get-Content $status -Raw | ConvertFrom-Json).status
    if ($value -eq 'PASS') { return $true }
    throw "existing run $runId has status $value; inspect before rerunning"
}
foreach ($seed in 1, 3, 4, 5) {
    if (-not (Done "NAACL_E2_infinitive_t${seed}_v1_20261010")) {
        & $py 'scripts\train_infinitive_program.py' --config "configs\naacl_sprint_20261010\e2_infinitive_t$seed.json"
        if ($LASTEXITCODE -ne 0) { throw "infinitive target $seed failed" }
    }
}
foreach ($seed in 2, 3, 4, 5) {
    if (-not (Done "NAACL_E1_GPT2_T${seed}_v1_20261010")) {
        & $py 'scripts\train_grammar_member_program.py' --config 'configs\naacl_sprint_20261010\e1_grammar_eval.json' --target-seed $seed
        if ($LASTEXITCODE -ne 0) { throw "GPT2 matching target $seed failed" }
    }
}
