param([int[]]$SourceSeeds = @(2,3,4,5), [int[]]$TargetSeeds = @(2,3,4,5))
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src;scripts;D:/CCAD_Storage/environments/r005a_sparsify_overlay'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$pythonPath = 'D:/CCAD_Storage/environments/r004/Scripts/python.exe'
$managerPath = 'E:/Projects/SAE_Lab/.resource_manager/resource_manager.py'
foreach ($sourceSeed in $SourceSeeds) {
    foreach ($targetSeed in $TargetSeeds) {
        if ($sourceSeed -eq $targetSeed) { continue }
        $runPath = "D:/CCAD_Storage/runs/reuse_generalization_20260921_round09/RG09_GPT2_SOURCE_REUSE_CONFIRM_S${sourceSeed}_T${targetSeed}_20260921"
        if (Test-Path -LiteralPath "$runPath/status.json") {
            $status = Get-Content -LiteralPath "$runPath/status.json" -Raw | ConvertFrom-Json
            if ($status.status -eq 'PASS') {
                Write-Output "Completed source $sourceSeed target $targetSeed retained"
                continue
            }
            throw "Source $sourceSeed target $targetSeed has an incomplete run; inspect retained outputs"
        }
        Write-Output "Starting source $sourceSeed target $targetSeed at $([DateTime]::UtcNow.ToString('o'))"
        & $pythonPath $managerPath run --resource cpu-heavy --project CCAD --task "rg09-s$sourceSeed-t$targetSeed" --wait-sec 3600 --heartbeat-sec 20 -- $pythonPath $managerPath run --resource gpu-0 --project CCAD --task "rg09-s$sourceSeed-t$targetSeed" --wait-sec 3600 --heartbeat-sec 20 -- $pythonPath scripts/train_grammar_member_program.py --config configs/rg09_grammar_source_reuse_confirmation.json --source-seed $sourceSeed --target-seed $targetSeed 2>&1 | Tee-Object -FilePath "artifacts/reuse_generalization_20260921_round09/CONFIRM_S${sourceSeed}_T${targetSeed}_LAUNCH.log"
        if ($LASTEXITCODE -ne 0) { throw "Source $sourceSeed target $targetSeed failed with exit code $LASTEXITCODE" }
    }
}
