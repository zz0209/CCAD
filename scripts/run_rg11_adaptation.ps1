param([string[]]$NewParts = @('verb','number','gender'), [int[]]$ContextCounts = @(8,32))
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src;scripts;D:/CCAD_Storage/environments/r005a_sparsify_overlay'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$pythonPath = 'D:/CCAD_Storage/environments/r004/Scripts/python.exe'
$managerPath = 'E:/Projects/SAE_Lab/.resource_manager/resource_manager.py'
foreach ($newPart in $NewParts) {
    foreach ($contextCount in $ContextCounts) {
        $runPath = "D:/CCAD_Storage/runs/reuse_generalization_20260921_round11/RG11_ADAPT_DEV_${newPart}_N${contextCount}_T2_20260921"
        if (Test-Path -LiteralPath "$runPath/status.json") {
            $status = Get-Content -LiteralPath "$runPath/status.json" -Raw | ConvertFrom-Json
            if ($status.status -eq 'PASS') {
                Write-Output "Completed adaptation $newPart with $contextCount contexts retained"
                continue
            }
            throw "Adaptation $newPart with $contextCount contexts has an incomplete run"
        }
        Write-Output "Starting adaptation $newPart with $contextCount contexts at $([DateTime]::UtcNow.ToString('o'))"
        & $pythonPath $managerPath run --resource cpu-heavy --project CCAD --task "rg11-$newPart-n$contextCount" --wait-sec 3600 --heartbeat-sec 20 -- $pythonPath $managerPath run --resource gpu-0 --project CCAD --task "rg11-$newPart-n$contextCount" --wait-sec 3600 --heartbeat-sec 20 -- $pythonPath scripts/train_grammar_member_program.py --config configs/rg11_function_adaptation_development.json --adapt-part $newPart --new-fit-pairs $contextCount --target-seed 2 2>&1 | Tee-Object -FilePath "artifacts/reuse_generalization_20260921_round11/DEV_${newPart}_N${contextCount}_LAUNCH.log"
        if ($LASTEXITCODE -ne 0) { throw "Adaptation $newPart failed with exit code $LASTEXITCODE" }
    }
}
