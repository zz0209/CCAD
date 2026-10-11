# Copy the Qwen3-1.7B model and the layer-14 Qwen-Scope transfer inputs from Mayo to D: for local EXP-U runs, then list
# SHA256 of every copied file next to the copy.
$ErrorActionPreference = 'Stop'
$scp = 'C:\Windows\System32\OpenSSH\scp.exe'
$opts = @('-F', 'C:\Users\zz\.codex\visualizations\2026\10\05\01a10e05-9ca7-7a03-bf79-5dad80863366\MayoLink\personal-ssh-config',
          '-o', 'UserKnownHostsFile=C:/Users/zz/.ssh/known_hosts', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes')
$M = 'mayo-workstation:/mnt/nvme3-data/harvardhds/ccad-m364569'
$D = 'D:\CCAD_Storage\runs\naacl_sprint_20261010\mayo_mirror\qwen_local'
$items = @(
    @{src = 'assets/qwen3-1.7b-base'; dst = 'qwen3-1.7b-base'; recurse = $true},
    @{src = 'runs/naacl_sprint_20261010/qwen17_prepare_l14_v2'; dst = 'qwen17_prepare_l14_v2'; recurse = $true},
    @{src = 'runs/naacl_sprint_20261010/qwen17_fresh_panel_rg09'; dst = 'qwen17_fresh_panel_rg09'; recurse = $true},
    @{src = 'data/cross_seed/qwen_natural_train.uint32.bin'; dst = 'qwen_natural_train.uint32.bin'; recurse = $false},
    @{src = 'data/cross_seed/qwen_natural_eval.uint32.bin'; dst = 'qwen_natural_eval.uint32.bin'; recurse = $false}
)
New-Item -ItemType Directory -Force $D | Out-Null
foreach ($i in $items) {
    if (Test-Path "$D\$($i.dst)") { continue }
    if ($i.recurse) { & $scp @opts -r "$M/$($i.src)" "$D\$($i.dst)" } else { & $scp @opts "$M/$($i.src)" "$D\$($i.dst)" }
    Write-Output "copied $($i.src)"
}
Get-ChildItem $D -Recurse -File | ForEach-Object { "{0}  {1}  {2}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Length, $_.FullName.Substring($D.Length + 1) } |
    Set-Content "$D\SHA256.txt"
Write-Output 'Qwen mirror finished'
