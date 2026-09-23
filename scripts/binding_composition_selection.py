import json
import time
from pathlib import Path

import numpy as np

from binding_member_selection import composition_fields
from ccad.artifacts import sha256
from run_causalgym_multisite import ROOT, write


CANDIDATES=['geometry','source_cond0','source_cond1','raw_cond0','raw_cond1','random']
PARTS=['entity','attribute']
MODELS=['source_additive','source_conditional','raw_additive','raw_conditional']


def response_basis(values,conditional):
    e0,e1,a0,a1=values.T
    terms=[e0,e1,e0*e0,e0*e1,e1*e1,a0,a1,a0*a0,a0*a1,a1*a1]
    if conditional:terms.extend([e0*a0,e0*a1,e1*a0,e1*a1])
    return np.stack(terms,axis=-1)


def run_composition_selection(c,w,torch,model,rows,codecs,means,cache,batch,forward,control):
    cfg=c['composition_selection'];device=w.device;layers=list(range(len(means)))
    assert cfg['members']==16 and cfg['candidate_members']==64 and cfg['ridge']>0
    candidate_names={part:list(CANDIDATES) for part in PARTS};r57={}
    if cfg.get('r57_bank'):
        path=w.checked(ROOT/cfg['r57_bank']);assert sha256(path)==cfg['r57_bank_sha256']
        saved=np.load(path)
        for family in ['source','raw']:
            name=family+'_r57_path';candidate_names['entity'].append(name)
            r57[name]={l:torch.tensor(saved[f'{family}_path_both_pre{l}'],device=device) for l in codecs}
    ne,na=[len(candidate_names[part]) for part in PARTS]
    phase=cfg.get('analysis_phase','development');bs=c['batch_size']
    fit_worlds=sorted({r['component'] for r in rows if r['split']=='fit'})[:cfg['fit_worlds']]
    eval_worlds=sorted({r['component'] for r in rows if r['split']=='development'})[:cfg['evaluation_worlds']]
    assert len(eval_worlds)==cfg['evaluation_worlds'] and eval_worlds
    fit_ids=[r['row_id'] for r in rows if r['split']=='fit' and r['component'] in fit_worlds]
    eval_ids=[r['row_id'] for r in rows if r['split']=='development' and r['component'] in eval_worlds]
    banks={family:{part:{condition:{l:torch.zeros(6,model.config.hidden_size,device=device) for l in layers}
        for condition in [0,1]} for part in PARTS} for family in ['source','raw']}
    models={};counts=dict(gradient_batches=0,source_response_batches=0,target_response_batches=0,capture_batches=0)
    identity=dict(source_seed=c['source_seed'],model_revision=c['model_revision'],
                  frozen_means_sha256=c['frozen_means_sha256'],sae_layers=sorted(codecs),
                  checkpoint_step=c['checkpoint_step'],members=16,candidate_members=64,
                  random_seed=cfg['random_seed'],candidates=candidate_names,
                  r57_bank_sha256=cfg.get('r57_bank_sha256'))
    output=w.run/'composition_selection_arrays';output.mkdir()

    def observe(b,ids,delta,gradient=False):
        control(dict(name='composition_selection',layers=layers,role_fields=delta),'both')
        lp=forward(b,gradient=gradient)
        ix=torch.arange(len(ids),device=device)
        original=torch.tensor([rows[i]['answer_id'] for i in ids],device=device)
        swap=torch.tensor([rows[i]['swap_answer_id'] for i in ids],device=device)
        return lp[ix,original]-lp[ix,swap],lp.argmax(-1)

    def capture(ids):
        b=batch(ids);control(capture_states=True);lp=forward(b);counts['capture_batches']+=1
        ix=torch.arange(len(ids),device=device)
        original=torch.tensor([rows[i]['answer_id'] for i in ids],device=device)
        swap=torch.tensor([rows[i]['swap_answer_id'] for i in ids],device=device)
        fields,codes=composition_fields(torch,ids,rows,codecs,means,cache,64)
        return b,(lp[ix,original]-lp[ix,swap]).cpu().numpy(),fields,codes

    def make_candidates(ids,fields,codes,kind):
        candidates={part:[] for part in PARTS};saved={};member_counts={}
        for part in PARTS:
            member_counts[part]=[]
            for name in candidate_names[part]:
                delta=dict(fields[kind][part])
                actual=torch.zeros(len(ids),dtype=torch.long,device=device)
                for l,asset in codecs.items():
                    coeff,base=codes[kind][part][l];decoder=asset['ds' if kind=='source' else 'dt']
                    if name=='geometry':score=coeff.abs()*decoder.norm(dim=1)
                    elif name=='random':
                        score=torch.empty_like(coeff)
                        for j,i in enumerate(ids):
                            generator=torch.Generator(device=device)
                            generator.manual_seed(cfg['random_seed']+i*1009+l*37+PARTS.index(part)*100003)
                            score[j]=torch.rand(coeff.shape[1:],generator=generator,device=device)
                    elif name in r57:
                        score=coeff*(r57[name][l]@decoder.T)[None]
                    else:
                        family,condition=name.split('_cond');condition=int(condition)
                        score=coeff*(banks[family][part][condition][l]@decoder.T)[None]
                        score=score*(1 if condition else -1)
                    chosen=score.masked_fill(coeff==0,-torch.inf).topk(16,dim=-1).indices
                    values=coeff.gather(-1,chosen)
                    selected=torch.zeros_like(coeff).scatter(-1,chosen,values)
                    delta[l]=selected@decoder
                    actual=torch.maximum(actual,(values!=0).sum(-1).max(-1).values)
                    prefix=f'{part}_{name}_pre{l}'
                    saved[prefix+'_support']=chosen.masked_fill(values==0,-1).cpu().numpy()
                    saved[prefix+'_coefficients']=values.cpu().numpy()
                    saved[prefix+'_baseline_code']=base.gather(-1,chosen).masked_fill(values==0,0).cpu().numpy()
                candidates[part].append(delta)
                member_counts[part].append(actual.cpu().numpy())
            member_counts[part]=np.stack(member_counts[part],axis=1)
            saved[part+'_actual_members']=member_counts[part]
            assert member_counts[part].max()<=16
        return candidates,saved,member_counts

    def coordinates(candidates):
        result={}
        for family in banks:
            result[family]={}
            for part in PARTS:
                result[family][part]=np.stack([np.stack([
                    sum((delta[l]*banks[family][part][condition][l]).sum((1,2)) for l in layers).cpu().numpy()
                    for condition in [0,1]],axis=-1) for delta in candidates[part]],axis=1)
        return result

    def pool_responses(b,ids,candidates,category):
        margins={};predictions={}
        for part in PARTS:
            measured=[observe(b,ids,delta) for delta in candidates[part]]
            margins[part]=np.stack([v[0].cpu().numpy() for v in measured],axis=1)
            predictions[part]=np.stack([v[1].cpu().numpy() for v in measured],axis=1)
        measured=[observe(b,ids,{l:e[l]+a[l] for l in layers})
                  for e in candidates['entity'] for a in candidates['attribute']]
        margins['both']=np.stack([v[0].cpu().numpy() for v in measured],axis=1).reshape(len(ids),ne,na)
        predictions['both']=np.stack([v[1].cpu().numpy() for v in measured],axis=1).reshape(len(ids),ne,na)
        counts[category]+=ne+na+ne*na
        return margins,predictions

    def design(coordinate):
        n=len(coordinate['entity'])
        entity=np.concatenate([coordinate['entity'],np.zeros_like(coordinate['entity'])],axis=-1)
        attribute=np.concatenate([np.zeros_like(coordinate['attribute']),coordinate['attribute']],axis=-1)
        both=np.concatenate([np.broadcast_to(coordinate['entity'][:,:,None,:],(n,ne,na,2)),
                             np.broadcast_to(coordinate['attribute'][:,None,:,:],(n,ne,na,2))],axis=-1)
        return dict(entity=entity,attribute=attribute,both=both)

    frozen=cfg.get('frozen_model_run')
    if frozen:
        source_run=ROOT/Path(frozen)
        assert json.loads(w.checked(source_run/'status.json').read_text())['status']=='PASS'
        path=w.checked(source_run/'composition_model.npz');assert sha256(path)==cfg['frozen_model_sha256']
        saved=np.load(path)
        assert json.loads(str(saved['identity']))==identity
        assert float(saved['ridge'])==cfg['ridge']
        for family in banks:
            for part in PARTS:
                for condition in [0,1]:
                    for l in layers:banks[family][part][condition][l]=torch.tensor(saved[f'{family}_{part}_{condition}_pre{l}'],device=device)
        for name in MODELS:models[name]={k:saved[f'{name}_{k}'] for k in ['coordinate_scale','feature_scale','weight']}
        (w.run/'composition_model.npz').write_bytes(path.read_bytes())
    else:
        assert len(fit_worlds)==cfg['fit_worlds'] and fit_ids
        for offset in range(0,len(fit_ids),bs):
            ids=fit_ids[offset:offset+bs];b,clean,fields,codes=capture(ids)
            saved_gradients={}
            for family in banks:
                gradients={}
                for e,a in [(0,0),(1,0),(0,1),(1,1)]:
                    leaf={l:(e*fields[family]['entity'][l]+a*fields[family]['attribute'][l]).detach().requires_grad_(True) for l in layers}
                    margin,_=observe(b,ids,leaf,True)
                    values=torch.autograd.grad(margin.sum(),list(leaf.values()))
                    gradients[e,a]={l:v.detach() for l,v in zip(layers,values)}
                    saved_gradients[f'{family}_margin_{e}{a}']=margin.detach().cpu().numpy()
                    counts['gradient_batches']+=1
                for condition in [0,1]:
                    for l in layers:
                        banks[family]['entity'][condition][l]+=(gradients[0,condition][l]+gradients[1,condition][l]).sum(0)/2
                        banks[family]['attribute'][condition][l]+=(gradients[condition,0][l]+gradients[condition,1][l]).sum(0)/2
            np.savez_compressed(output/f'fit_gradients_{offset:06d}.npz',row_ids=np.array(ids),**saved_gradients)
            w.progress('COMPOSITION_GRADIENTS',completed_rows=offset+len(ids),total_rows=len(fit_ids),counts=counts)
        for family in banks:
            for part in PARTS:
                for condition in [0,1]:
                    for l in layers:banks[family][part][condition][l]/=len(fit_ids)
                    assert all(torch.isfinite(value).all() for value in banks[family][part][condition].values())
        features={family:[] for family in banks};outcomes=[]
        for offset in range(0,len(fit_ids),bs):
            ids=fit_ids[offset:offset+bs];b,clean,fields,codes=capture(ids)
            candidates,support,member_counts=make_candidates(ids,fields,codes,'source')
            coordinate=coordinates(candidates);truth,prediction=pool_responses(b,ids,candidates,'source_response_batches')
            payload=dict(row_ids=np.array(ids),clean_margin=clean,**support)
            for family in banks:
                values=design(coordinate[family])
                features[family].append(np.concatenate([values[s].reshape(-1,4) for s in ['entity','attribute','both']]))
                payload.update({f'{family}_features_{s}':v for s,v in values.items()})
            outcomes.append(np.concatenate([(truth[s]-clean.reshape((-1,)+(1,)*(truth[s].ndim-1))).ravel() for s in ['entity','attribute','both']]))
            payload.update({f'true_margin_{s}':v for s,v in truth.items()})
            payload.update({f'prediction_token_{s}':v for s,v in prediction.items()})
            np.savez_compressed(output/f'fit_responses_{offset:06d}.npz',**payload)
            w.progress('COMPOSITION_SOURCE_RESPONSES',completed_rows=offset+len(ids),total_rows=len(fit_ids),counts=counts)
        y=np.concatenate(outcomes).astype(np.float64)
        for name in MODELS:
            family,form=name.split('_');x=np.concatenate(features[family]).astype(np.float64)
            coordinate_scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-12)
            z=response_basis(x/coordinate_scale,form=='conditional')
            feature_scale=np.maximum(np.sqrt(np.mean(z*z,axis=0)),1e-12);z=z/feature_scale
            weight=np.linalg.solve(z.T@z+cfg['ridge']*len(y)*np.eye(z.shape[1]),z.T@y)
            assert np.isfinite(weight).all()
            models[name]=dict(coordinate_scale=coordinate_scale,feature_scale=feature_scale,weight=weight)
        payload=dict(identity=np.array(json.dumps(identity,sort_keys=True)),fit_worlds=np.array(fit_worlds),
                     fit_row_ids=np.array(fit_ids),ridge=np.array(cfg['ridge']))
        payload.update({f'{family}_{part}_{condition}_pre{l}':value.cpu().numpy()
                        for family,parts in banks.items() for part,conditions in parts.items()
                        for condition,ll in conditions.items() for l,value in ll.items()})
        payload.update({f'{name}_{key}':value for name,parameters in models.items() for key,value in parameters.items()})
        np.savez_compressed(w.run/'composition_model.npz',**payload)
    model_sha=sha256(w.run/'composition_model.npz')
    write(w.run/'composition_selection_protocol.json',dict(identity,analysis_phase=phase,fit_worlds=fit_worlds,
        evaluation_worlds=eval_worlds,model_sha256=model_sha,frozen_model_run=frozen,models=MODELS,
        coordinates='Each part projects its full actual hidden update including raw background on two fit-world gradient banks conditioned on the other part being absent or full; average its own zero/full endpoint gradients.',
        response='All four regressors fit identical source candidate responses; fixed original-minus-swapped margin minus the same clean margin offset.',
        selection='Maximize min(-predicted_entity_margin,-predicted_attribute_margin,predicted_joint_margin); first candidate in saved order resolves ties.',
        target_information='One clean forward per row supplies the shared offset; target candidate responses are used only after predictions and selection for scoring.',
        candidate_direction='cond0 ranks negative fixed-margin contributions, cond1 positive contributions; one fixed random candidate and geometry complete the shared pool.',
        fitting='Ridge with coordinate RMS and basis RMS scaling; additive has ten within-part linear/quadratic terms, conditional adds four entity-by-attribute products; no intercept.',
        family_operation='SAE layers execute selected original decoder contributions, all other layers retain the same R57 raw background according to the active part.',counts=counts))
    for offset in range(0,len(eval_ids),bs):
        ids=eval_ids[offset:offset+bs];b,clean,fields,codes=capture(ids)
        candidates,support,member_counts=make_candidates(ids,fields,codes,'target')
        coordinate=coordinates(candidates);predicted={};chosen={}
        for name in MODELS:
            family,form=name.split('_');parameters=models[name];values=design(coordinate[family]);predicted[name]={}
            for state,x in values.items():
                z=response_basis(x.reshape(-1,4)/parameters['coordinate_scale'],form=='conditional')/parameters['feature_scale']
                predicted[name][state]=(z@parameters['weight']).reshape(x.shape[:-1])+clean.reshape((-1,)+(1,)*(x.ndim-2))
            p=predicted[name]
            p['utility']=np.minimum(np.minimum(-p['entity'][:,:,None],-p['attribute'][:,None,:]),p['both'])
            chosen[name]=p['utility'].reshape(len(ids),-1).argmax(-1)
        truth,prediction=pool_responses(b,ids,candidates,'target_response_batches')
        utility=np.minimum(np.minimum(-truth['entity'][:,:,None],-truth['attribute'][:,None,:]),truth['both'])
        payload=dict(row_ids=np.array(ids),clean_margin=clean,**support)
        for family in banks:payload.update({f'{family}_features_{s}':v for s,v in design(coordinate[family]).items()})
        payload.update({f'true_margin_{s}':v for s,v in truth.items()})
        payload.update({f'prediction_token_{s}':v for s,v in prediction.items()})
        payload.update({f'{name}_predicted_{s}':v for name,states in predicted.items() for s,v in states.items()})
        np.savez_compressed(output/f'evaluation_{offset:06d}.npz',**payload)
        for j,i in enumerate(ids):
            row=rows[i];shared=dict(row_id=i,component=row['component'],split=row['split'],analysis_phase=phase,
                panel_seed=c['panel_seed'],order=row['order'],template=row['template'],query=row['query'],
                seed=c['source_seed'],source_seed=c['source_seed'],target_seed=c['target_seed'],
                original_answer_id=row['answer_id'],swap_answer_id=row['swap_answer_id'],clean_margin=float(clean[j]))
            records=[]
            for ei,en in enumerate(candidate_names['entity']):
                for ai,an in enumerate(candidate_names['attribute']):
                    em=float(truth['entity'][j,ei]);am=float(truth['attribute'][j,ai]);jm=float(truth['both'][j,ei,ai])
                    tokens={s:int(prediction[s][j,ei] if s=='entity' else prediction[s][j,ai] if s=='attribute' else prediction[s][j,ei,ai]) for s in ['entity','attribute','both']}
                    correct={s:token==(row['answer_id'] if s=='both' else row['swap_answer_id']) for s,token in tokens.items()}
                    record=dict(shared,candidate_id=ei*na+ai,mode=f'entity_{ei}_attribute_{ai}',entity_candidate=en,attribute_candidate=an,
                        actual_members=max(int(member_counts['entity'][j,ei]),int(member_counts['attribute'][j,ai])),
                        margin_entity=em,margin_attribute=am,margin_both=jm,true_utility=float(utility[j,ei,ai]),
                        success_entity=em<0,success_attribute=am<0,success_both=jm>0,
                        all_success=em<0 and am<0 and jm>0,all_correct=all(correct.values()),
                        **{f'prediction_token_{s}':v for s,v in tokens.items()},**{f'correct_{s}':v for s,v in correct.items()})
                    for name,p in predicted.items():
                        record.update({f'prediction_{name}_entity':float(p['entity'][j,ei]),
                            f'prediction_{name}_attribute':float(p['attribute'][j,ai]),
                            f'prediction_{name}_both':float(p['both'][j,ei,ai]),
                            f'prediction_{name}_utility':float(p['utility'][j,ei,ai]),
                            f'selected_{name}':int(chosen[name][j])==ei*na+ai})
                    w.record(kind='composition_candidate',**record);records.append(record)
            oracle=float(utility[j].max())
            for name in MODELS:
                record=records[int(chosen[name][j])]
                w.record(kind='composition_choice',method=name,actual_utility=record['true_utility'],
                    oracle_utility=oracle,oracle_regret=oracle-record['true_utility'],**record)
            baselines={name:(name,name) for name in ['geometry','source_cond0','source_cond1','raw_cond0','raw_cond1']}
            baselines.update({name:(name,name.split('_')[0]+'_cond1') for name in r57})
            for name,(entity_name,attribute_name) in baselines.items():
                ei=candidate_names['entity'].index(entity_name);ai=candidate_names['attribute'].index(attribute_name)
                record=records[ei*na+ai]
                w.record(kind='composition_choice',method=name,actual_utility=record['true_utility'],
                    oracle_utility=oracle,oracle_regret=oracle-record['true_utility'],**record)
        w.progress('COMPOSITION_TARGET_SELECTION',completed_rows=offset+len(ids),total_rows=len(eval_ids),counts=counts)
    write(w.run/'composition_selection_cost.json',dict(counts=counts,fit_rows=0 if frozen else len(fit_ids),
        evaluation_rows=len(eval_ids),model_sha256=model_sha,max_members=16,seconds=time.perf_counter()-w.wall_start))
    w.checks.update(source_only_response_fit=True,frozen_before_target_scoring=True,shared_candidate_pool=True,
                    fixed_answer_margin=True,shared_clean_margin_offset=True,allowed_member_budget=True)
