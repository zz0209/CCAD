param(
    [Parameter(Mandatory = $true)][ValidateSet('dev', 'confirm')][string]$Stage
)
$ErrorActionPreference = 'Stop'
$root = 'E:\Projects\SAE_Lab\CCAD'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH = "$root\src;D:/CCAD_Storage/environments/f4_sparse_overlay_v1"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$bulk = 'D:\CCAD_Storage\runs\naacl_sprint_20261010'
if ($Stage -eq 'dev') {
    $jobs = @(@{config = 'e9_gpt2_fixed_dev_lr0.005'; seed = 2; run = 'NAACL_E9_GPT2_FIXED_DEV_LR0.005_T2_v1_20261010'},
              @{config = 'e9_gpt2_fixed_dev_lr0.02'; seed = 2; run = 'NAACL_E9_GPT2_FIXED_DEV_LR0.02_T2_v1_20261010'})
} else {
    $jobs = foreach ($seed in 2, 3, 4, 5) { @{config = 'e9_gpt2_fixed_confirm'; seed = $seed; run = "NAACL_E9_GPT2_FIXED_T${seed}_v1_20261010"} }
}
foreach ($job in $jobs) {
    $runDir = "$bulk\$($job.run)"
    if (Test-Path "$runDir\status.json") {
        $value = (Get-Content "$runDir\status.json" -Raw | ConvertFrom-Json).status
        if ($value -eq 'PASS') { continue }
        throw "existing run $runDir has status $value; inspect before rerunning"
    }
    & $py 'scripts\train_grammar_member_program.py' --config "configs\naacl_sprint_20261010\$($job.config).json" --target-seed $job.seed
    if ($LASTEXITCODE -ne 0) { throw "E9 $($job.run) failed" }
}
