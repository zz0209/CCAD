param(
    [int[]]$Seeds = @(2, 3, 4, 5),
    [string[]]$Panels = @('r59panel', 'ir04panel'),
    [switch]$MatchingOnly
)
$ErrorActionPreference = 'Stop'
$root = 'E:\Projects\SAE_Lab\CCAD'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH = "$root\src;D:/CCAD_Storage/environments/f4_sparse_overlay_v1"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
foreach ($seed in $Seeds) {
    $relation = "D:\CCAD_Storage\runs\naacl_sprint_20261010\matching_relations_seed$seed\MATCHING_RELATIONS.json"
    if (-not (Test-Path $relation)) {
        & $py 'scripts\naacl_matching_relations.py' --config "configs\naacl_sprint_20261010\e1_matching_relations_seed$seed.json"
        if ($LASTEXITCODE -ne 0) { throw "matching relations failed for seed $seed" }
    }
    if ($MatchingOnly) { continue }
    foreach ($panel in $Panels) {
        $runDir = "D:\CCAD_Storage\runs\naacl_sprint_20261010\NAACL_E1_shift_${panel}_seed${seed}_v2_20261010"
        if (Test-Path "$runDir\status.json") {
            $status = (Get-Content "$runDir\status.json" -Raw | ConvertFrom-Json).status
            if ($status -eq 'PASS') { continue }
            throw "existing run $runDir has status $status; inspect before rerunning"
        }
        & $py 'scripts\run_shift_transfer.py' --config "configs\naacl_sprint_20261010\e1_shift_${panel}_seed$seed.json"
        if ($LASTEXITCODE -ne 0) { throw "evaluation failed for $panel seed $seed" }
    }
}
