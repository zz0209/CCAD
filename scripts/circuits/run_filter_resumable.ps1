# Finish the interrupted Gemma-2 PLT filter-prediction run (v14) under the gpu-0 lease, stopping and resuming around other
# work on this machine. Each attempt resumes from the latest saved rows into a new run directory, so saved prompts are never
# recomputed. An attempt is stopped right after the prompt in progress has been saved (at once while the model is still
# loading) when one condition holds on two consecutive samples about 10 s apart:
#   (a) its python workers hold more than 1 GiB of shared GPU memory;
#   (b) another process waits for or requests gpu-0 through the resource manager;
#   (c) after loading, a compute process of another project runs.
# When (a) alone holds and no other process uses the GPU, the run overflows its own memory: a new process resumes at once,
# at most MaxSelfRestarts times. Any other stop waits until the processes that caused it and all compute processes of other
# projects have ended and none has appeared for 10 minutes, then resumes, at most MaxResumes times; with -WaitFirst the
# script starts with this wait. Compute processes of other projects are node processes running speaker-eval and processes
# outside this run that use the GPU: compute-engine utilization above 1%, or a command line and 1 to 64 GiB of dedicated
# GPU memory (the desktop window manager has no readable command line, and some overlay processes report values in the
# terabytes). Short interface tests (tsx and the electron processes they start) do not count. While an attempt runs, free
# physical memory below 1 GiB for a minute is reported in the log.
# With -CheckGraph and -CheckReference, the first prompt saved by the first attempt is compared with its saved row
# (compare_filter_rows.py); a difference stops the run.
param([Parameter(Mandatory = $true)][int]$FirstVersion, [Parameter(Mandatory = $true)][string]$ResumeFrom,
      [Parameter(Mandatory = $true)][string]$Log, [int]$AttrBatch = 16, [int]$MaxResumes = 2, [int]$MaxSelfRestarts = 5,
      [int]$ResumesUsed = 0, [switch]$WaitFirst, [string]$CheckGraph = '', [string]$CheckReference = '')
$ErrorActionPreference = 'Stop'
$env:HF_HOME = 'D:\CCAD_Storage\hf_home'
$Py = 'D:\CCAD_Storage\environments\circuit_local_v1\Scripts\python.exe'
$Rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$R = 'D:\CCAD_Storage\runs\circuits'
$Code = 'E:\Projects\SAE_Lab\CCAD\scripts\circuits'
$Prefix = 'CIRC_FILTER_LOCAL_GEMMA_PLT'
$JobArgs = @('filter_prediction.py', '--model', 'google/gemma-2-2b', '--dictionary', 'gemma', '--role', 'plt', '--prompts',
             "$R\HELDOUT_PROMPTS_gemma-2-2b.json", '--attr-batch', "$AttrBatch")

function Say([string]$m) {
    $line = '{0} {1}' -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), $m
    Add-Content $Log $line
    [Console]::WriteLine($line)
}
$isOurs = { param($c) [bool]($c -and ($c -match "${Prefix}_v\d" -or $c -match 'run_filter_resumable\.ps1' -or
    $c -match 'run_filter_local\.ps1' -or $c -match 'watch_filter_memory\.ps1')) }
function OtherWs($procs) { [double](($procs | Where-Object { -not (& $isOurs $_.CommandLine) } | Measure-Object WorkingSetSize -Sum).Sum) }
function FreeMem { [double]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB) }
function OtherGpu($procs) {
    # Processes outside this run that use the GPU (see the header).
    $byPid = @{}
    foreach ($p in $procs) { $byPid[[int]$p.ProcessId] = $p }
    $found = @{}
    $counter = Get-Counter '\GPU Process Memory(*)\Dedicated Usage', '\GPU Engine(*engtype_Compute*)\Utilization Percentage' `
        -ErrorAction SilentlyContinue
    foreach ($s in ($counter.CounterSamples | Where-Object { $_.Status -eq 0 })) {
        $id = [int]($s.InstanceName -replace '^pid_(\d+)_.*', '$1')
        $p = $byPid[$id]
        if (-not $p -or (& $isOurs $p.CommandLine)) { continue }
        if ($s.Path -match 'utilization' -and $s.CookedValue -gt 1) { $found[$id] = $p.Name }
        if ($s.Path -match 'dedicated' -and $p.CommandLine -and $s.CookedValue -gt 1GB -and $s.CookedValue -lt 64GB) { $found[$id] = $p.Name }
    }
    $found
}
function OtherWork($procs) {
    # Compute processes of other projects: node running speaker-eval, and processes using the GPU.
    $found = OtherGpu $procs
    foreach ($p in $procs) {
        if ($p.Name -eq 'node.exe' -and $p.CommandLine -match 'speaker-eval' -and -not (& $isOurs $p.CommandLine)) {
            $found[[int]$p.ProcessId] = $p.Name
        }
    }
    $found
}
function WaitQuiet($culprits) {
    # Return once the given processes and all compute processes of other projects have been absent for 10 minutes.
    $quietSince = $null
    while ($true) {
        Start-Sleep -Seconds 10
        $alive = @($culprits | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        $work = OtherWork (Get-CimInstance Win32_Process)
        if ($alive.Count -eq 0 -and $work.Count -eq 0) {
            if (-not $quietSince) { $quietSince = Get-Date } elseif (((Get-Date) - $quietSince).TotalMinutes -ge 10) { return }
        } else { $quietSince = $null }
    }
}

$from = $ResumeFrom
$resumes = $ResumesUsed; $selfRestarts = 0
if ($WaitFirst) {
    Say "WAITING before the first attempt until compute processes of other projects have been absent for 10 minutes"
    WaitQuiet @()
    Say 'RESUMING'
}
for ($attempt = 0; ; $attempt++) {
    $run = '{0}_v{1}' -f $Prefix, ($FirstVersion + $attempt)
    $dir = "$R\$run"
    $out = "$R\$run.log"
    $start = Get-Date
    $argList = @($Rm, 'run', '--resource', 'gpu-0', '--project', 'CCAD', '--task', "v14-$run", '--wait-sec', '3600',
                 '--heartbeat-sec', '20', '--', $Py) + $JobArgs + @('--resume-from', $from, '--out', $dir)
    $job = Start-Process -FilePath 'C:\Program Files\Python313\python.exe' -ArgumentList $argList -WorkingDirectory $Code `
        -RedirectStandardOutput $out -RedirectStandardError "$R\$run.err.log" -PassThru -NoNewWindow
    $null = $job.Handle
    Say "attempt $attempt started $run from $from (wrapper pid $($job.Id))"
    $bad = 0; $trip = $null; $tripStage = $null; $tripLoaded = $false; $tripSelf = $false; $culprits = @(); $loaded = $false
    $base = 0.
    $lastStage = 'none'; $lowSince = $null; $lowSaid = $false
    while (-not $job.HasExited) {
        Start-Sleep -Seconds 10
        $procs = Get-CimInstance Win32_Process
        $workers = @($procs | Where-Object { $_.CommandLine -match 'filter_prediction\.py' -and $_.CommandLine -match $run -and
            $_.CommandLine -notmatch 'resource_manager' })
        if ($workers.Count -eq 0) { continue }
        $stage = if (Test-Path "$dir\progress.json") { (Get-Content "$dir\progress.json" -Raw).Trim() } else { 'none' }
        if ($stage -ne $lastStage) { Say "$run $stage"; $lastStage = $stage }
        if ($CheckGraph -and $stage -ne 'none') {
            & $Py "$Code\compare_filter_rows.py" --rows "$dir\FILTER_PREDICTION_ROWS.partial.json" --reference $CheckReference `
                --graph $CheckGraph --out "$R\$run.check.json" | ForEach-Object { Say "$run CHECK $_" }
            if ($LASTEXITCODE -ne 0) {
                $workers | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }
                Say "$run MISMATCH with the saved row of $CheckGraph (exit $LASTEXITCODE); stopped, see $R\$run.check.json"
                $job.WaitForExit()
                exit 3
            }
            Say "$run CHECK PASSED for $CheckGraph"
            $CheckGraph = ''
        }
        if (-not $loaded -and (Test-Path $out) -and (Select-String -Path $out -Pattern '"stage": "bank"' -Quiet)) {
            $loaded = $true
            $base = OtherWs $procs
            Say ('{0} loaded; working set of other processes {1:N2} GiB, free {2:N2} GiB' -f $run, ($base / 1GB), ((FreeMem) / 1GB))
        }
        # Get-Counter reports an error when a process exits during the read and still returns the other samples; samples
        # whose status is not valid are left out.
        $shared = 0.
        $counter = Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
        foreach ($s in ($counter.CounterSamples | Where-Object { $_.Status -eq 0 })) {
            foreach ($p in $workers.ProcessId) { if ($s.InstanceName -like "pid_$($p)_*") { $shared += $s.CookedValue } }
        }
        $free = FreeMem
        if ($free -lt 1GB) {
            if (-not $lowSince) { $lowSince = Get-Date }
            elseif (-not $lowSaid -and ((Get-Date) - $lowSince).TotalSeconds -ge 60) {
                Say ('{0} LOW MEMORY: free physical memory below 1 GiB for a minute ({1:N2} GiB now), loaded={2}, at {3}' -f
                    $run, ($free / 1GB), $loaded, $stage)
                $lowSaid = $true
            }
        } else { $lowSince = $null; $lowSaid = $false }
        $why = @(); $who = @(); $self = $false
        if ($shared -gt 1GB) {
            $gpu = OtherGpu $procs
            if ($gpu.Count) {
                $why += ('shared GPU memory {0:N2} GiB with other GPU processes {1}' -f ($shared / 1GB),
                    (($gpu.GetEnumerator() | ForEach-Object { "$($_.Key) $($_.Value)" }) -join ', '))
                $who += [int[]]$gpu.Keys
            } else {
                $why += ('shared GPU memory {0:N2} GiB of this run alone' -f ($shared / 1GB)); $self = $true
            }
        }
        $waiters = @($procs | Where-Object { $_.CommandLine -match 'resource_manager' -and $_.CommandLine -match 'gpu-0' -and
            $_.CommandLine -match '\s(run|acquire)\s' -and -not (& $isOurs $_.CommandLine) })
        if ($waiters.Count) { $why += 'gpu-0 requested by ' + ($waiters.ProcessId -join ','); $who += $waiters.ProcessId }
        if ($loaded) {
            $work = OtherWork $procs
            if ($work.Count) {
                $why += 'compute processes of other projects ' + (($work.GetEnumerator() | ForEach-Object {
                    "$($_.Key) $($_.Value)" }) -join ', ')
                $who += [int[]]$work.Keys
            }
        }
        if (-not $trip) {
            if ($why.Count) { $bad += 1 } else { $bad = 0 }
            if ($bad -ge 2) {
                $trip = $why -join '; '; $tripStage = $stage; $tripLoaded = $loaded; $culprits = $who
                $tripSelf = $self -and $why.Count -eq 1
                Say "$run TRIP ($trip) at $stage, loaded=$loaded, own overflow=$tripSelf"
            }
        }
        if ($trip -and (-not $tripLoaded -or $stage -ne $tripStage)) {
            $workers | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }
            Say "$run STOPPED after $stage (workers $($workers.ProcessId -join ','))"
            $job.WaitForExit()
        }
    }
    $job.WaitForExit()
    if (Test-Path "$dir\FILTER_PREDICTION_ROWS.json") { Say "$run DONE (wrapper exit $($job.ExitCode))"; exit 0 }
    if (-not $trip) { Say "$run FAILED with wrapper exit $($job.ExitCode) without a trip; see $out"; exit 1 }
    if (Test-Path "$dir\FILTER_PREDICTION_ROWS.partial.json") { $from = "$dir\FILTER_PREDICTION_ROWS.partial.json" }
    if ($tripSelf) {
        if ($selfRestarts -ge $MaxSelfRestarts) { Say "FINAL STOP after $selfRestarts restarts for own overflow; latest rows $from"; exit 2 }
        $selfRestarts += 1
        Say "RESTARTING at once for own overflow ($selfRestarts of $MaxSelfRestarts)"
        continue
    }
    if ($resumes -ge $MaxResumes) { Say "FINAL STOP after $resumes resumes for other work; latest rows $from"; exit 2 }
    $resumes += 1
    Say ("WAITING for processes $($culprits -join ',') and compute processes of other projects to be absent for 10 minutes " +
        "(resume $resumes of $MaxResumes)")
    WaitQuiet $culprits
    Say 'RESUMING'
}
