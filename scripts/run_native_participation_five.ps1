$ErrorActionPreference = 'Stop'
Set-Location 'E:/Projects/SAE_Lab/CCAD'
$env:PYTHONPATH = 'E:/Projects/SAE_Lab/CCAD/src;D:/CCAD_Storage/environments/f4_sparse_overlay_v1;D:/CCAD_Storage/environments/r005a_sparsify_overlay;D:/CCAD_Storage/references/source/sparsify_42c0645'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$taskPython = 'D:/CCAD_Storage/environments/r004/Scripts/python.exe'
$taskManager = 'E:/Projects/SAE_Lab/.resource_manager/resource_manager.py'
$taskReceipt = 'artifacts/final_five_research_20260908/r15_multisite/FIVE_SEQUENCE.jsonl'
foreach ($taskSourceSeed in 1..5) {
    $taskConfig = "configs/final5_r15_participation_five_s${taskSourceSeed}_v2.json"
    $taskResolved = Get-Content $taskConfig -Raw | ConvertFrom-Json
    $taskRunId = $taskResolved.run_id
    if (Test-Path "runs/$taskRunId") { throw "Existing immutable run: $taskRunId" }
    $taskStart = [DateTime]::UtcNow.ToString('o')
    & $taskPython $taskManager run --resource gpu-0 --project CCAD --task $taskRunId --wait-sec 600 --heartbeat-sec 20 -- $taskPython scripts/run_causalgym_native_participation.py --config $taskConfig
    $taskCode = $LASTEXITCODE
    $taskEnd = [DateTime]::UtcNow.ToString('o')
    @{run_id=$taskRunId; wrapper_started_at_utc=$taskStart; wrapper_ended_at_utc=$taskEnd; exit_code=$taskCode} | ConvertTo-Json -Compress | Add-Content $taskReceipt -Encoding utf8
    if ($taskCode -ne 0) { throw "Stopped sequence after failure: $taskRunId" }
}
