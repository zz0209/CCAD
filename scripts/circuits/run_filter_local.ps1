# Filtered-graph prediction of held-out interventions on the local GPU (v14 of the attribution-graph study), one job
# after the other, each under a gpu-0 lease of the shared resource manager. Arguments: job names among llama-clt,
# llama-plt, gemma-clt, gemma-plt, g3-clt, g3-plt, smoke-llama, smoke-gemma.
param([Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Jobs)
$ErrorActionPreference = 'Stop'
$env:HF_HOME = 'D:\CCAD_Storage\hf_home'
$Py = 'D:\CCAD_Storage\environments\circuit_local_v1\Scripts\python.exe'
$Rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$R = 'D:\CCAD_Storage\runs\circuits'
$Code = 'E:\Projects\SAE_Lab\CCAD\scripts\circuits'
$LlamaPlt = 'D:\CCAD_Storage\models\circuit_dictionaries\transcoder-llama-3.2-1b'
$G3Plt = 'mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine'
$G3Clt = 'mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine'
$Specs = @{
    'smoke-llama' = @(@('--model', 'meta-llama/Llama-3.2-1B', '--dictionary', $LlamaPlt, '--role', 'plt', '--max-prompts', '2', '--prompts', "$R\CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v4\ORACLE_ROWS.json"), 'CIRC_FILTER_LOCAL_SMOKE_LLAMA_PLT_v1')
    'smoke-gemma' = @(@('--model', 'google/gemma-2-2b', '--dictionary', 'mntss/clt-gemma-2-2b-426k', '--role', 'clt', '--max-prompts', '2', '--prompts', "$R\CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9\ORACLE_ROWS.json"), 'CIRC_FILTER_LOCAL_SMOKE_GEMMA_v1')
    'smoke-gemma-memory' = @(@('--model', 'google/gemma-2-2b', '--dictionary', 'mntss/clt-gemma-2-2b-426k', '--role', 'clt', '--max-prompts', '2', '--prompts', "$R\CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9\ORACLE_ROWS.json"), 'CIRC_FILTER_LOCAL_SMOKE_GEMMA_v2')
    'llama-clt' = @(@('--model', 'meta-llama/Llama-3.2-1B', '--dictionary', 'mntss/clt-llama-3.2-1b-524k', '--role', 'clt', '--prompts', "$R\HELDOUT_PROMPTS_llama-3.2-1b.json"), 'CIRC_FILTER_LOCAL_LLAMA_CLT524K_v1')
    'llama-plt' = @(@('--model', 'meta-llama/Llama-3.2-1B', '--dictionary', $LlamaPlt, '--role', 'plt', '--prompts', "$R\HELDOUT_PROMPTS_llama-3.2-1b.json"), 'CIRC_FILTER_LOCAL_LLAMA_PLT_v1')
    'gemma-clt' = @(@('--model', 'google/gemma-2-2b', '--dictionary', 'mntss/clt-gemma-2-2b-426k', '--role', 'clt', '--prompts', "$R\HELDOUT_PROMPTS_gemma-2-2b.json"), 'CIRC_FILTER_LOCAL_GEMMA_CLT426K_v1')
    'gemma-plt-resume' = @(@('--model', 'google/gemma-2-2b', '--dictionary', 'gemma', '--role', 'plt', '--prompts', "$R\HELDOUT_PROMPTS_gemma-2-2b.json", '--attr-batch', '16', '--resume-from', "$R\CIRC_FILTER_LOCAL_GEMMA_PLT_v1\FILTER_PREDICTION_ROWS.partial.json"), 'CIRC_FILTER_LOCAL_GEMMA_PLT_v2')
    'gemma-plt-resume2' = @(@('--model', 'google/gemma-2-2b', '--dictionary', 'gemma', '--role', 'plt', '--prompts', "$R\HELDOUT_PROMPTS_gemma-2-2b.json", '--attr-batch', '16', '--resume-from', "$R\CIRC_FILTER_LOCAL_GEMMA_PLT_v2\FILTER_PREDICTION_ROWS.partial.json"), 'CIRC_FILTER_LOCAL_GEMMA_PLT_v3')
    'gemma-plt-resume3' = @(@('--model', 'google/gemma-2-2b', '--dictionary', 'gemma', '--role', 'plt', '--prompts', "$R\HELDOUT_PROMPTS_gemma-2-2b.json", '--attr-batch', '16', '--resume-from', "$R\CIRC_FILTER_LOCAL_GEMMA_PLT_v2\FILTER_PREDICTION_ROWS.partial.json"), 'CIRC_FILTER_LOCAL_GEMMA_PLT_v4')
    'gemma-plt' = @(@('--model', 'google/gemma-2-2b', '--dictionary', 'gemma', '--role', 'plt', '--prompts', "$R\HELDOUT_PROMPTS_gemma-2-2b.json"), 'CIRC_FILTER_LOCAL_GEMMA_PLT_v1')
    'g3-clt' = @(@('--model', 'google/gemma-3-1b-pt', '--dictionary', $G3Clt, '--role', 'clt', '--edit-residual', '--prompts', "$R\HELDOUT_PROMPTS_gemma-3-1b.json"), 'CIRC_FILTER_LOCAL_G3_CLT262K_v1')
    'g3-plt' = @(@('--model', 'google/gemma-3-1b-pt', '--dictionary', $G3Plt, '--role', 'plt', '--prompts', "$R\HELDOUT_PROMPTS_gemma-3-1b.json"), 'CIRC_FILTER_LOCAL_G3_PLT_v1')
}
Set-Location $Code
foreach ($job in $Jobs) {
    $a, $run = $Specs[$job]
    $log = "$R\$run.log"
    # Attribution batch 32 unless the job sets its own (memory only; the graph is the same).
    $batch = if ($a -contains '--attr-batch') { @() } else { @('--attr-batch', '32') }
    & 'C:\Program Files\Python313\python.exe' $Rm run --resource gpu-0 --project CCAD --task "v14-$job" --wait-sec 3600 --heartbeat-sec 20 -- $Py filter_prediction.py @a @batch --out "$R\$run" *> $log
    if ($LASTEXITCODE -ne 0) { throw "job $job failed with exit code $LASTEXITCODE; see $log" }
}
'all done'
