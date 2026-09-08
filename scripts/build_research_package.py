"""Bundle retained study stages with raw results and explicit cache omissions.

This streams files into a new local ZIP. It does not download assets, rerun an
experiment, change source files, or publish anything. Use the project's disk-I/O
resource lease for the multi-GB input hashing and archive verification.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('artifacts/seven_round_rebuild_20260906')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-bytes',type=int,default=900_000_000)
    parser.add_argument('--receipt',type=Path,default=ROOT/BASE/'r7_science_package/PACKAGE_BUILD.json',
                        help='Round-specific build receipt; use a new path for each later package.')
    parser.add_argument('--artifact-root',type=Path,action='append',default=[],help='Additional project-relative research artifact root; repeat for later stages.')
    parser.add_argument('--run-prefix',action='append',default=[],help='Additional immutable run prefix; repeat for later stages.')
    parser.add_argument('--scope',default='Local original-stage research snapshot; not a public release or clean-machine experiment rerun. Mutable tracker/master log reflect snapshot time; final closure is appended as a separately named receipt.')
    args=parser.parse_args();timer=time.monotonic();started=dt.datetime.now(dt.timezone.utc).isoformat()
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    paths=set();omitted=[];external=[];all_inputs=[]
    def omit(path,kind,recovery):
        omitted.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest(path),kind=kind,recovery=recovery))
    roots=['src','scripts','tests','configs','paper',BASE.as_posix(),'artifacts/bonus_audit_20260907','.aris/compute']+args.artifact_root
    for rel in roots:
        resolved=(ROOT/rel).resolve()
        if not resolved.is_relative_to(ROOT):raise ValueError('Artifact root leaves project')
        for p in resolved.rglob('*'):
            if p.is_file() and not any(x in {'__pycache__','.pytest_cache','.git'} for x in p.parts) and p.suffix not in {'.pyc','.pyo'}:
                if p.name.endswith('.uint16.bin'):
                    omit(p,'packed public-corpus token stream','Retain original local stream or regenerate from the pinned corpus/document/packing manifests. Verify its exact hash before training. See REPRODUCTION.md.')
                elif p.name=='sampled_documents.jsonl':
                    omit(p,'full sampled natural-document text','Retain the original corpus sample or obtain the pinned public dataset rows identified by document metadata. Reduced exclusion ledgers preserve the original sampling exclusions without copying web prose.')
                elif 'reference_views' in p.parts and p.suffix.lower() not in {'.json','.jsonl'}:
                    omit(p,'third-party reference cache','Retain the original local paper/code/data copy or obtain the exact official source recorded in the adjacent SOURCE_MANIFEST.json and REFERENCE_REGISTRY.md. Project adaptations and their required license notices are included separately.')
                else:paths.add(p)
    for name in ['AGENTS.md','EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','REFERENCE_REGISTRY.md','DERIVATION_PACKAGE.md','master_log.md','RUN_ARTIFACT_CONTRACT.md',
                 'goal_aligned_subspace_consistency_complete_proofs.pdf','goal_aligned_subspace_consistency_research_program.md','PAPER_SNAPSHOT_20260906.md']:
        p=ROOT/name
        if p.exists():paths.add(p)
    corpus=ROOT/'runs/R011_NR1_long_budget_corpus_v2_20260904T012000Z'
    for rel in ['config.resolved.json','config.json','manifest.json','inputs.json','environment.json','artifacts/token_manifest.json','artifacts/document_token_records.json']:
        p=corpus/rel
        if p.is_file():paths.add(p)
    runs=sorted({run for prefix in ['SEVEN_']+args.run_prefix for run in (ROOT/'runs').glob(prefix+'*') if run.is_dir()})
    for run in runs:
        for p in run.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in {'.pyc','.pyo'}:continue
            if p.name.endswith('.uint16.bin'):
                omit(p,'packed public-corpus token stream','Retain the original stream or regenerate it with the recorded document selection, tokenizer and packing recipe; verify its exact hash before training.')
            elif p.name=='sampled_documents.jsonl':
                omit(p,'full sampled natural-document text','Restore the retained pinned corpus sample or regenerate it using this run configuration and the matching reduced exclusion ledger. Document/token metadata and original hashes remain supplied.')
            elif p.name=='raw_cache.npz' or (p.name.startswith('seed') and p.name.endswith('_codes.npz')) or p.suffix=='.npy':
                omit(p,'regenerable activation or task-code cache','Retain original local file, or rerun the recorded generator with the same pinned assets and a new run ID; see this run configuration/input/source snapshot and README.md. No raw metric rows are omitted.')
            else:paths.add(p)
        inp=run/'inputs.json'
        if inp.exists():
            for entry in json.loads(inp.read_text(encoding='utf-8')).get('inputs',[]):
                all_inputs.append(dict(recorded_by=inp.relative_to(ROOT).as_posix(),**entry))
    # Retain the delivery/ layout used by paper links, without recursively
    # nesting older delivery archives in the new all-stage bundle.
    for p in (ROOT/'delivery').iterdir():
        if not p.is_file():continue
        if p.suffix.lower()=='.zip':
            omit(p,'prior local delivery snapshot','Preserved at the recorded local path. The new bundle includes the underlying study stages directly rather than nesting this historical ZIP.')
        elif p.suffix.lower() in {'.md','.json','.jsonl'}:paths.add(p)
    names={p.relative_to(ROOT).as_posix():p for p in paths}
    for name in ['README.md','REPRODUCTION.md','ROUND_INDEX.md','corpus_exclusions.jsonl','CORPUS_EXCLUSION_PROVENANCE.json']:
        names[name]=ROOT/'delivery'/name
    for pattern in ['corpus_exclusions_*.jsonl','CORPUS_EXCLUSION_PROVENANCE_*.json']:
        for p in (ROOT/'delivery').glob(pattern):names[p.name]=p
    packed_paths={str(p.resolve()).casefold() for p in names.values()}
    omitted_paths={str((ROOT/r['path']).resolve()).casefold() for r in omitted}
    for entry in all_inputs:
        recorded=Path(entry['path']);recorded=recorded if recorded.is_absolute() else ROOT/recorded
        identity=str(recorded.resolve()).casefold()
        if identity not in packed_paths:
            external.append(dict(**entry,package_status='omitted_regenerable_cache' if identity in omitted_paths else 'external_or_prior_stage_input',
                recovery_scope='Use the exact input identity and producer configuration/source snapshot; bulk weights and natural token streams are acquired or regenerated separately.'))
    # Sources may contain additional credentials in the future. The intentionally
    # narrow input roots exclude user configurations, accounts, model caches and .git.
    assert len(names)==len(set(names))
    entries=[]
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for i,(name,p) in enumerate(sorted(names.items())):
            before=p.stat();h=hashlib.sha256()
            with p.open('rb') as source,archive.open(name,'w',force_zip64=True) as target:
                while chunk:=source.read(1<<20):h.update(chunk);target.write(chunk)
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError(f'Source changed during packaging: {name}')
            entries.append(dict(path=name,bytes=after.st_size,sha256=h.hexdigest()))
            if i%200==0:print(json.dumps(dict(stage='packing',files=i+1,total=len(names),zip_bytes=output.stat().st_size)),flush=True)
            if output.stat().st_size>args.max_bytes:raise RuntimeError('Package size budget exceeded; partial archive retained')
        manifest=dict(written_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),started_at_utc=started,
            scope=args.scope,
            campaign_run_directories=[p.relative_to(ROOT).as_posix() for p in runs],files=entries,
            omitted_local_caches=omitted,omission_schema_note='Legacy field name; each record has its exact kind, including regenerable caches, packed tokens and third-party reference PDFs.',external_inputs_from_retained_manifests=external,
            source_bytes=sum(x['bytes'] for x in entries),omitted_cache_bytes=sum(x['bytes'] for x in omitted))
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        bad=archive.testzip()
        if bad is not None:raise RuntimeError(f'Archive CRC failure: {bad}')
        assert len(archive.namelist())==len(set(archive.namelist()))
    receipt=dict(started_at_utc=started,completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),wall_seconds=time.monotonic()-timer,
        output=output.relative_to(ROOT).as_posix(),bytes=output.stat().st_size,sha256=digest(output),
        files=len(entries)+1,campaign_runs=len(runs),source_bytes=manifest['source_bytes'],
        omitted_caches=len(omitted),omitted_cache_bytes=manifest['omitted_cache_bytes'],
        all_zip_crc_pass=True,max_package_bytes=args.max_bytes,scope=manifest['scope'])
    receipt_path=args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
