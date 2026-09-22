$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src;scripts;D:/CCAD_Storage/environments/r005a_sparsify_overlay'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$pythonPath = 'D:/CCAD_Storage/environments/r004/Scripts/python.exe'
$managerPath = 'E:/Projects/SAE_Lab/.resource_manager/resource_manager.py'
foreach ($part in @('verb','number','gender')) {
    $runPath = "D:/CCAD_Storage/runs/reuse_generalization_20260921_round11/RG11_CAPACITY_HOLDOUT_${part}_T2_20260921"
    if (Test-Path -LiteralPath "$runPath/status.json") {
        $status = Get-Content -LiteralPath "$runPath/status.json" -Raw | ConvertFrom-Json
        if ($status.status -eq 'PASS') { continue }
        throw "Capacity holdout $part has an incomplete run"
    }
    Write-Output "Starting capacity holdout $part at $([DateTime]::UtcNow.ToString('o'))"
    & $pythonPath $managerPath run --resource cpu-heavy --project CCAD --task "rg11-capacity-$part" --wait-sec 3600 --heartbeat-sec 20 -- $pythonPath $managerPath run --resource gpu-0 --project CCAD --task "rg11-capacity-$part" --wait-sec 3600 --heartbeat-sec 20 -- $pythonPath scripts/train_grammar_member_program.py --config configs/rg11_capacity_function_holdout.json --heldout-part $part --target-seed 2 2>&1 | Tee-Object -FilePath "artifacts/reuse_generalization_20260921_round11/CAPACITY_HOLDOUT_${part}_LAUNCH.log"
    if ($LASTEXITCODE -ne 0) { throw "Capacity holdout $part failed with exit code $LASTEXITCODE" }
}
