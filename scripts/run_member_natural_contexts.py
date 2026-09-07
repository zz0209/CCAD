"""Activation exemplars of already selected members in existing natural validation text."""
import argparse,json,time,traceback
from pathlib import Path
from composition_runtime import CompositionRun,ROOT,write,np
from ccad.artifacts import sha256
from ccad.data_manifest import canonical_sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_member_natural_contexts.py',['src/ccad/data_manifest.py']);error=None
    try:
        work.load();cfg=work.cfg;torch=work.torch
        train=json.loads(work.checked(ROOT/cfg['training_config']).read_text());path=work.checked(ROOT/train['validation_token_path'])
        assert sha256(path)==train['validation_token_sha256']
        tokens=np.fromfile(path,dtype='<u2').astype(np.int64);assert len(tokens)==32768
        documents=json.loads(work.checked(ROOT/train['document_records']).read_text())['documents'];documents=[d for d in documents if d['split']=='validation']
        spans=[];offset=1;full_checks=[]
        for doc in documents:
            end=offset+doc['included_token_count'];spans.append((offset,min(end,len(tokens)),doc))
            if end<=len(tokens):full_checks.append(canonical_sha256(tokens[offset:end].tolist())==doc['included_token_sha256'])
            offset=end+1
        work.checks['natural_document_token_identity']=all(full_checks) and len(full_checks)>0
        from sparsify import SparseCoder
        maps=np.load(work.checked(ROOT/cfg['correspondence_run']/'maps_s1_t2.npz'))
        src=np.load(work.checked(ROOT/cfg['source_run']/'seed1_source_coordinates.npz'))
        selected={1:np.unique(np.concatenate([src['number_16_support'],src['time_32_support']])),2:np.unique(np.concatenate([maps['number_fcc_members'],maps['time_fcc_members']]))}
        saes={}
        for seed in selected:
            spec=next(s for s in cfg['sae_checkpoints'] if s['seed']==seed);path=Path(spec['path']);work.checked(path/'sae.safetensors');work.checked(path/'cfg.json')
            saes[seed]=SparseCoder.load_from_disk(path,device='cuda').eval()
        activations={seed:[] for seed in selected};holder={}
        def capture(module,inputs):holder['h']=inputs[0].detach()
        handle=work.model.gpt_neox.final_layer_norm.register_forward_pre_hook(capture)
        sequences=tokens.reshape(-1,128);batch=cfg['natural_batch_size']
        try:
            with torch.no_grad():
                for off in range(0,len(sequences),batch):
                    if time.perf_counter()-work.start>cfg['budget_seconds']:raise TimeoutError('Natural exemplar budget exhausted')
                    ids=torch.as_tensor(sequences[off:off+batch],device='cuda')
                    work.model.gpt_neox(input_ids=ids,use_cache=False);work.forwards+=len(ids)
                    h=holder['h'].reshape(-1,holder['h'].shape[-1])
                    for seed,sae in saes.items():
                        act,ind,_=sae.encode(h);dense=torch.zeros((len(act),sae.num_latents),device='cuda').scatter_(1,ind,act)
                        activations[seed].append(dense[:,selected[seed]].cpu().numpy())
        finally:handle.remove()
        arrays={};counts=[]
        for seed,pieces in activations.items():
            x=np.concatenate(pieces);arrays[f'seed{seed}_activations']=x;arrays[f'seed{seed}_members']=selected[seed]
            for column,member in enumerate(selected[seed]):
                values=x[:,column];order=np.argsort(-values,kind='stable');contexts=[]
                for index in order:
                    if values[index]<=0 or len(contexts)==3:break
                    span=next(((a,b,d) for a,b,d in spans if a<=index<b),None)
                    if span is None:continue
                    a,b,doc=span;seq_start=(int(index)//128)*128
                    lo=max(a,seq_start,int(index)-8);hi=min(b,seq_start+128,int(index)+3)
                    contexts.append(dict(activation=float(values[index]),token_offset=int(index),sequence_index=int(index)//128,
                        before=work.tokenizer.decode(tokens[lo:index].tolist()),token=work.tokenizer.decode([int(tokens[index])]),after=work.tokenizer.decode(tokens[index+1:hi].tolist()),
                        token_ids=tokens[lo:hi].tolist(),document_id=doc['document_id'],url=doc['source_metadata']['url'],text_sha256=doc['text_sha256']))
                record=dict(kind='natural_member',seed=seed,member=int(member),positions=len(values),active_positions=int(np.sum(values>0)),mean_activation=float(values.mean()),max_activation=float(values.max()),contexts=contexts)
                work.record(record);counts.append((seed,int(member)))
        np.savez_compressed(work.run/'natural_member_activations.npz',**arrays)
        work.checks['all_selected_members']=len(counts)==sum(map(len,selected.values()))
        work.checks['unique']=len(set(counts))==len(counts)
        work.checks['all_natural_tokens']=all(arrays[f'seed{s}_activations'].shape[0]==32768 for s in selected)
        write(work.run/'natural_context_scope.json',dict(tokens=32768,documents=len(documents),checked_full_document_spans=len(full_checks),seeds=[1,2],
            selection='All frozen source1 N16/T32 members and target2 N/time FCC members; natural outcomes do not select members. Top3 positive activations per member, stable position tie break; clips stay within document and actual128token context. Includes inactive members.',
            interpretation='Activation exemplars only, no natural-task causal or semantic-label validation. Validation stream was previously used for SAE quality; this is not new held-out performance evidence.',dataset='HuggingFaceFW/fineweb',revision=train['dataset_commit'],license=train['dataset_license']))
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
