param([string[]]$HeldoutParts = @('verb','number','gender'), [int[]]$TargetSeeds = @(3,4,5))
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src;scripts;D:/CCAD_Storage/environments/r005a_sparsify_overlay'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$pythonPath = 'D:/CCAD_Storage/environments/r004/Scripts/python.exe'
$managerPath = 'E:/Projects/SAE_Lab/.resource_manager/resource_manager.py'
foreach ($heldoutPart in $HeldoutParts) {
    foreach ($targetSeed in $TargetSeeds) {
        $runPath = "D:/CCAD_Storage/runs/reuse_generalization_20260921_round10/RG10_FUNCTION_CONFIRM_${heldoutPart}_T${targetSeed}_20260921"
        if (Test-Path -LiteralPath "$runPath/status.json") {
            $status = Get-Content -LiteralPath "$runPath/status.json" -Raw | ConvertFrom-Json
            if ($status.status -eq 'PASS') {
                Write-Output "Completed omission $heldoutPart target $targetSeed retained"
                continue
            }
            throw "Omission $heldoutPart target $targetSeed has an incomplete run; inspect saved outputs"
        }
        Write-Output "Starting omission $heldoutPart target $targetSeed at $([DateTime]::UtcNow.ToString('o'))"
        & $pythonPath $managerPath run --resource cpu-heavy --project CCAD --task "rg10-$heldoutPart-t$targetSeed" --wait-sec 3600 --heartbeat-sec 20 -- $pythonPath $managerPath run --resource gpu-0 --project CCAD --task "rg10-$heldoutPart-t$targetSeed" --wait-sec 3600 --heartbeat-sec 20 -- $pythonPath scripts/train_grammar_member_program.py --config configs/rg10_grammar_function_holdout_confirmation.json --heldout-part $heldoutPart --target-seed $targetSeed 2>&1 | Tee-Object -FilePath "artifacts/reuse_generalization_20260921_round10/CONFIRM_${heldoutPart}_T${targetSeed}_LAUNCH.log"
        if ($LASTEXITCODE -ne 0) { throw "Omission $heldoutPart target $targetSeed failed with exit code $LASTEXITCODE" }
    }
}
