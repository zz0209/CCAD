"""Compact raw-backed five-seed quality ledger, not an FCC effect result."""
import json
import statistics
from pathlib import Path
from run_r008b_paired_codes import ROOT,sha256,write_json


def main():
    output=ROOT/'runs/R011_NR1_k128_seed5_v1_20260905'
    dest=output/'FIVE_SEED_QUALITY.json'
    if dest.exists():raise FileExistsError(dest)
    rows=[];inputs=[];reference=None;reference_config=None
    ignored={'run_id','purpose','init_seeds','scope_limit','evidence_level'}
    for seed in range(1,6):
        run=ROOT/'runs'/(f'R011_NR1_k128_seed{seed}_v1_20260904T054000Z' if seed<3 else f'R011_NR1_k128_seed{seed}_v1_20260905')
        raw=run/'metrics.raw.jsonl';r=json.loads(raw.read_text());v=r['validation'];cfg=json.loads((run/'config.resolved.json').read_text())
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        assert json.loads((run/'contract_validation.json').read_text())['ok']
        science={k:v for k,v in cfg.items() if k not in ignored}
        reference_config=science if reference_config is None else reference_config
        reference=r['training_input_hashes'] if reference is None else reference
        assert science==reference_config and r['training_input_hashes']==reference
        assert cfg['init_seeds']==[seed] and r['train_steps']==8192 and r['train_tokens']==4194304
        trace=r['train_loss_trace'];assert len(trace)==8192
        rows.append(dict(seed=seed,run=str(run),train_seconds=r['train_seconds'],train_tokens=r['train_tokens'],
            train_steps=r['train_steps'],training_input_hashes_match=True,scientific_config_match=True,
            fve=v['fve'],ce_recovered=v['ce_recovered'],actual_l0=v['actual_nonzero_l0'],selected_l0=v['selected_l0'],
            alive_features=v['alive_features'],dead_features=v['feature_firing_count_distribution']['dead_count'],
            decoder_norm_max_error=r['decoder_norm_max_error'],first256_fvu=statistics.mean(t['fvu'] for t in trace[:256]),
            last256_fvu=statistics.mean(t['fvu'] for t in trace[-256:]),safe_bytes=r['safe_checkpoint_bytes'],exact_bytes=r['exact_checkpoint_bytes'],
            checkpoint_sha256=sha256(run/'sae/sae.safetensors')))
        inputs.extend(dict(path=str(p),sha256=sha256(p)) for p in (raw,run/'config.resolved.json',run/'code_hashes.json'))
    result=dict(rows=rows,inputs=inputs,generator_sha256=sha256(Path(__file__)),
        new_seed_train_seconds=sum(r['train_seconds'] for r in rows if r['seed']>=3),
        scope='Five same-configuration k128 SAEs, identical ordered training batches; validation quality only. First/last256 training batches differ in content, not a controlled convergence proof. FCC behavior/compactness replication is a separate consumer. No new audit.')
    write_json(dest,result)
    lines=['# Five-seed long-k128 asset quality','',result['scope'],'',
        '|Seed|FVE|CE recovered|actual L0|alive|dead|first256 FVU|last256 FVU|train sec|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join(str(r[k]) for k in ('seed','fve','ce_recovered','actual_l0','alive_features','dead_features','first256_fvu','last256_fvu','train_seconds'))+'|')
    (output/'FIVE_SEED_QUALITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
