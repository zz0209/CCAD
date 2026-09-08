"""Repack retained public documents, preserving duplicate-source evidence."""
import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from run_r006a_capacity_manifest import ROOT, pack_split, score, write_uint16
from run_r011s1_raw_hook_asset import write_json as write, entry, aggregate
from ccad.artifacts import sha256, validate_run_directory
from ccad.data_manifest import validate_document_records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    cfg=json.loads(parser.parse_args().config.read_text())
    run=ROOT/'runs'/cfg['run_id']
    run.mkdir(exist_ok=False)
    artifacts=run/'artifacts'
    artifacts.mkdir()
    start=time.perf_counter()
    write(run/'config.resolved.json',cfg)
    files=[]
    for rel in ['scripts/repack_unique_training_corpus.py','scripts/run_r006a_capacity_manifest.py',
                'scripts/run_r011s1_raw_hook_asset.py','src/ccad/token_packing.py','src/ccad/data_manifest.py','src/ccad/artifacts.py','src/ccad/http_range.py']:
        source=ROOT/rel
        dest=run/'source_snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
        files.append(dict(path=rel,sha256=sha256(source),bytes=source.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=files,aggregate_sha256=aggregate(files),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='training.repack.v1',run_id=cfg['run_id'],run_parent='FINAL_FIVE_R14',
        purpose=cfg['purpose'],milestone='fresh-natural-material',evidence_level='training_material_asset',
        started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(files),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        mean_constants_source_split='not_applicable_training',threshold_source_split='Exact text hash dedup; no outcome selection',
        statistics_unit='document',device='cpu',seeds=[],resource_lease=None,
        resource_lease_reason='Bounded single-core tokenization/linear packing of existing170MB public documents; no network/GPU/new bulk'))
    for fn in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/fn).touch()
    write(run/'status.json',dict(status='RUNNING'))
    inputs=[];checks={};error=None;env={}
    try:
        import transformers
        parent=ROOT/cfg['source_run']
        for fn in ['artifacts/sampled_documents.jsonl','artifacts/source_catalog.json','artifacts/document_token_records.json','config.resolved.json','metrics.summary.json']:
            inputs.append(entry(parent/fn,'Retained fixed-version FineWeb range sample','Existing public data; original failure retained','input'))
        rows=[json.loads(line) for line in (parent/'artifacts/sampled_documents.jsonl').read_text().splitlines()]
        unique=[];seen={};dropped=[]
        for row in sorted(rows,key=lambda r:(r['split'],score(cfg['selection_salt']+'-order',r['document_id']))):
            if row['text_sha256'] in seen:
                dropped.append(dict(removed_document_id=row['document_id'],retained_document_id=seen[row['text_sha256']],
                    text_sha256=row['text_sha256'],original_split=row['split'],reason='Exact duplicate text; parent record retained'))
            else:
                unique.append(row);seen[row['text_sha256']]=row['document_id']
        report=validate_document_records(unique)
        checks['unique_texts']=report['unique_text_hashes']
        checks['unique_document_ids']=report['unique_document_ids']
        checks['train_validation_disjoint']=not({r['text_sha256'] for r in unique if r['split']=='train'} & {r['text_sha256'] for r in unique if r['split']=='validation'})
        write(artifacts/'duplicate_dispositions.json',dict(source_run=cfg['source_run'],rows=dropped))
        tokenizer=transformers.AutoTokenizer.from_pretrained(cfg['tokenizer_local_dir'],local_files_only=True)
        all_documents=[];all_sequences=[];outputs={}
        for split in ['train','validation']:
            tokens,documents,sequences=pack_split([r for r in unique if r['split']==split],tokenizer,cfg,split)
            path=artifacts/(split+'.uint16.bin')
            write_uint16(path,tokens)
            outputs[split]=dict(path=path.relative_to(run).as_posix(),tokens=len(tokens),sequences=len(sequences),documents=len(documents),sha256=sha256(path))
            checks[split+'_exact_tokens']=len(tokens)==cfg[split+'_sequences']*cfg['context_length']
            all_documents.extend(documents);all_sequences.extend(sequences)
            progress=dict(stage='PACKED',split=split,tokens=len(tokens),seconds=time.perf_counter()-start)
            write(run/'progress.json',progress);print(json.dumps(progress),flush=True)
        write(artifacts/'document_token_records.json',dict(documents=all_documents))
        write(artifacts/'sequence_records.json',dict(sequences=all_sequences))
        used={r['document_id'] for r in all_documents}
        with (artifacts/'sampled_documents.jsonl').open('w',encoding='utf-8') as stream:
            for row in unique:
                if row['document_id'] in used:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        write(artifacts/'token_manifest.json',dict(dataset_id=cfg['dataset_id'],dataset_commit=cfg['dataset_commit'],
            source_run=cfg['source_run'],deduplicated_documents=len(dropped),outputs=outputs,source_catalog_sha256=sha256(parent/'artifacts/source_catalog.json')))
        env=dict(python=sys.executable,python_version=platform.python_version(),transformers=transformers.__version__,tokenizer_parallelism='disabled',network_bytes=0)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        (run/'stderr.log').write_text(traceback.format_exc())
    for key,value in checks.items():
        with (run/'metrics.raw.jsonl').open('a') as stream:stream.write(json.dumps(dict(check=key,passed=value))+'\n')
    status='PASS' if error is None and all(checks.values()) else 'FAIL'
    write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env)
    summary=dict(status=status,error=error,checks=checks,wall_seconds=time.perf_counter()-start,
        metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),scope=cfg['scope_limit'],
        generator_script_path='scripts/repack_unique_training_corpus.py',generator_script_sha256=sha256(run/'source_snapshot/scripts/repack_unique_training_corpus.py'))
    write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary)
    write(run/'status.json',dict(status=status,error=error,ended_utc=datetime.now(timezone.utc).isoformat()))
    contract=validate_run_directory(run)
    write(run/'contract_validation.json',dict(ok=contract.ok,errors=list(contract.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=contract.ok,errors=list(contract.errors))),flush=True)
    return 0 if status=='PASS' and contract.ok else 1


if __name__=='__main__':raise SystemExit(main())
