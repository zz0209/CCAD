# Watch a filter-prediction run every 10 s and stop it, right after the current prompt has been saved (progress.json
# changes after the partial rows are written), when it competes with other work: (a) its python workers hold more than
# 1 GiB of shared GPU memory, (b) a process of another task waits for or requests gpu-0 through the resource manager, or
# (c) the working sets of all processes outside this run grow by more than 3 GiB over their value at the start. A
# condition must hold on two consecutive samples. The resource manager wrapper releases gpu-0 when the workers end.
param([Parameter(Mandatory = $true)][string]$Run, [Parameter(Mandatory = $true)][string]$Log)
$R = 'D:\CCAD_Storage\runs\circuits'
$progress = "$R\$Run\progress.json"
$read = { if (Test-Path $progress) { (Get-Content $progress -Raw).Trim() } else { 'none' } }
# Processes of this run and of this watcher, excluded from the working-set sum of other work.
$ours = { param($c) $c -match [regex]::Escape($Run) -or $c -match 'run_filter_local\.ps1' -or $c -match 'watch_filter_memory\.ps1' }
$others = { (Get-CimInstance Win32_Process | Where-Object { -not (& $ours $_.CommandLine) } | Measure-Object WorkingSetSize -Sum).Sum }
$baseline = & $others
Add-Content $Log ('{0} baseline working set of other processes {1:N2} GiB' -f (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), ($baseline / 1GB))
$seen = $false; $bad = 0; $trip = $null
while ($true) {
    $procs = Get-CimInstance Win32_Process
    $workers = @($procs | Where-Object { $_.CommandLine -match 'filter_prediction\.py' -and $_.CommandLine -match [regex]::Escape($Run) -and
        $_.CommandLine -notmatch 'resource_manager' })
    if ($workers.Count -eq 0) {
        if ($seen) { Add-Content $Log ('{0} workers gone' -f (Get-Date).ToUniversalTime().ToString('HH:mm:ss')); break }
        Start-Sleep -Seconds 10; continue
    }
    $seen = $true
    $shared = 0.
    foreach ($s in (Get-Counter '\GPU Process Memory(*)\Shared Usage').CounterSamples) {
        foreach ($p in $workers.ProcessId) { if ($s.InstanceName -like "pid_$($p)_*") { $shared += $s.CookedValue } }
    }
    $waiters = @($procs | Where-Object { $_.CommandLine -match 'resource_manager' -and $_.CommandLine -match 'gpu-0' -and
        $_.CommandLine -match '\s(run|acquire)\s' -and -not (& $ours $_.CommandLine) -and $_.CommandLine -notmatch [regex]::Escape($Run) })
    $other = ($procs | Where-Object { -not (& $ours $_.CommandLine) } | Measure-Object WorkingSetSize -Sum).Sum
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB
    $stage = & $read
    $why = @()
    if ($shared -gt 1GB) { $why += 'shared' }
    if ($waiters.Count -gt 0) { $why += 'gpu-0 requested by ' + (($waiters | ForEach-Object { $_.ProcessId }) -join ',') }
    if ($other - $baseline -gt 3GB) { $why += 'other work' }
    Add-Content $Log ('{0} progress={1} shared_gib={2:N2} other_ws_gib={3:N2} free_gib={4:N2} {5}' -f
        (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $stage, ($shared / 1GB), ($other / 1GB), ($free / 1GB), ($why -join '; '))
    if (-not $trip) {
        if ($why.Count -gt 0) { $bad += 1 } else { $bad = 0 }
        if ($bad -ge 2) {
            $trip = $stage
            Add-Content $Log ('{0} TRIP at progress={1} ({2}); waiting for the current prompt' -f
                (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $stage, ($why -join '; '))
        }
    }
    if ($trip -and ((& $read) -ne $trip)) {
        $workers | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }
        Add-Content $Log ('{0} STOPPED after progress={1}; workers {2}' -f (Get-Date).ToUniversalTime().ToString('HH:mm:ss'),
            (& $read), ($workers.ProcessId -join ','))
        break
    }
    Start-Sleep -Seconds 10
}
