# Download the nine BLiMP agreement paradigms reserved for EXP-U confirmation (research lead decision 4) from the
# official repository at the commit the project already pins, next to the existing BLiMP files, and write a manifest
# with URL, size and SHA256 for each.
$ErrorActionPreference = 'Stop'
$commit = '3e56b06fcabca9b30822fc66435fca6b1aa40bb1'
$dir = "E:\Projects\SAE_Lab\CCAD\artifacts\scientific_reform_20260913\reference_material\blimp\$commit"
# The repository names the first adjective paradigm determiner_noun_agreement_with_adjective_1.
$names = 'irregular_plural_subject_verb_agreement_2', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2',
         'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adjective_1',
         'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1',
         'determiner_noun_agreement_with_adj_irregular_2'
$listing = Invoke-RestMethod -Uri "https://api.github.com/repos/alexwarstadt/blimp/contents/data?ref=$commit" -UseBasicParsing
$rows = foreach ($n in $names) {
    $entry = $listing | Where-Object { $_.name -eq "$n.jsonl" }
    if (-not $entry) { throw "$n not in the repository listing" }
    $url = $entry.download_url
    $path = "$dir\$n.jsonl"
    # Files from an earlier attempt are kept when their size equals the repository's.
    if (-not (Test-Path $path)) { Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing }
    $f = Get-Item $path
    if ($f.Length -ne $entry.size) { throw "$n has $($f.Length) bytes, repository lists $($entry.size)" }
    [ordered]@{ name = $n; url = $url; bytes = $f.Length; sha256 = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower();
                lines = (Get-Content $path | Measure-Object -Line).Lines }
}
[ordered]@{ written_utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); repository = 'https://github.com/alexwarstadt/blimp';
            commit = $commit; license = 'CC BY 4.0 (repository README)'; purpose = 'EXP-U confirmation paradigms, never used before';
            files = $rows } | ConvertTo-Json -Depth 4 | Set-Content "$dir\FRESH_PARADIGMS_MANIFEST.json"
Get-Content "$dir\FRESH_PARADIGMS_MANIFEST.json"
