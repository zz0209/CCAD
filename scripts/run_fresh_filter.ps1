# Source-only inclusion of the nine confirmation paradigms for GPT-2 Medium and Qwen3-1.7B, under the gpu-0 lease.
$ErrorActionPreference = 'Stop'
$py = 'D:\CCAD_Storage\environments\r004\Scripts\python.exe'
$rm = 'E:\Projects\SAE_Lab\.resource_manager\resource_manager.py'
$env:PYTHONPATH = 'E:\Projects\SAE_Lab\CCAD\src;D:\CCAD_Storage\environments\f4_sparse_overlay_v1'
Set-Location 'E:\Projects\SAE_Lab\CCAD\scripts'
$b = 'E:\Projects\SAE_Lab\CCAD\artifacts\scientific_reform_20260913\reference_material\blimp\3e56b06fcabca9b30822fc66435fca6b1aa40bb1'
$paradigms = 'irregular_plural_subject_verb_agreement_2', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2',
             'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adjective_1',
             'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1',
             'determiner_noun_agreement_with_adj_irregular_2' | ForEach-Object { "$b\$_.jsonl" }
$R = 'D:\CCAD_Storage\runs\cross_seed_models'
$gpt2 = 's1_t2', 's2_t3', 's3_t4', 's4_t5', 's5_t1' | ForEach-Object { "$R\CSM_READERS_GPT2_${_}_v1_20261011\config.resolved.json" }
$qwen = 's1_t2_ts0', 's2_t1_ts0' | ForEach-Object { "E:\Projects\SAE_Lab\CCAD\configs\cross_seed_pilot\ureaders_qwen17_$_.json" }
$out = 'E:\Projects\SAE_Lab\CCAD\artifacts\containment'
& $py $rm run --resource gpu-0 --project CCAD --task 'Fresh paradigm filter GPT-2' --wait-sec 3600 -- $py fresh_paradigm_filter.py --model 'GPT-2 Medium' --configs @gpt2 --paradigms @paradigms --output "$out\FRESH_PARADIGMS_GPT2.json"
& $py $rm run --resource gpu-0 --project CCAD --task 'Fresh paradigm filter Qwen' --wait-sec 3600 -- $py fresh_paradigm_filter.py --model 'Qwen3-1.7B' --configs @qwen --paradigms @paradigms --output "$out\FRESH_PARADIGMS_QWEN.json"
Write-Output 'Fresh paradigm filter finished'
