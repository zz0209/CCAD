"""Discover and causally test digit-specific source SAE components.

A donor differs in both answer digits. A successful unit/tens intervention
copies just that digit and preserves the other. All edits start at the same
prompt-final hidden state; no correct answer prefix is supplied to generation.
Configurations select source fitting, target adaptation, or frozen evaluation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, re, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                  OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
from run_causalgym_multisite import MultisiteWork, ROOT, write
from ccad.artifacts import sha256


def panel(cfg):
    if cfg.get('frozen_rule_evaluation'):
        saved=json.loads((ROOT/cfg['frozen_rule_evaluation']['panel']).read_text())
        return saved['rows'],saved['pairs']
    rows=[]
    operands=cfg.get("evaluation_operand_pairs")
    if operands is None:
        operands=[[a,b] for a in range(*cfg["operand_range"]) for b in range(a,cfg["operand_range"][1])]
    for t,template in enumerate(cfg["templates"]):
        for a,b in operands:
            if [a,b] in cfg.get("excluded_pairs",[]):continue
            key=f"{cfg['split_salt']}:{a}:{b}"
            split="fit" if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2==0 else "development"
            if cfg.get("evaluation_operand_pairs") is not None:split="development"
            if t and split=="fit":continue
            total=a+b
            assert 10<=total<100
            rows.append(dict(template=t,a=a,b=b,total=total,unit=total%10,tens=total//10,
                             carry=(a%10+b%10)>=10,split=split,prompt=template.format(a=a,b=b)))
    excluded_operands=set()
    if cfg.get("evaluation_holdout"):
        prior=json.loads((ROOT/cfg["evaluation_holdout"]["from_run"]/"panel.json").read_text())
        assert prior["rows"]==rows
        excluded_operands={(prior["rows"][i]["a"],prior["rows"][i]["b"]) for p in prior["pairs"] for i in [p["recipient"],p["donor"]]}
    pairs=[]
    for t in range(len(cfg["templates"])):
        ii=[i for i,r in enumerate(rows) if r["template"]==t and r["split"]=="development" and (r["a"],r["b"]) not in excluded_operands]
        pair_salt=cfg.get("evaluation_holdout",{}).get("pair_salt",cfg["split_salt"])
        ii=sorted(ii,key=lambda i:hashlib.sha256(f"pair:{pair_salt}:{rows[i]['a']}:{rows[i]['b']}".encode()).hexdigest())
        used=set()
        for i in ii:
            if i in used:continue
            j=next((j for j in ii if j not in used and rows[i]["unit"]!=rows[j]["unit"] and rows[i]["tens"]!=rows[j]["tens"] and
                    not {rows[i]["a"],rows[i]["b"]}&{rows[j]["a"],rows[j]["b"]}),None)
            if j is None:continue
            used.update([i,j]);r,d=rows[i],rows[j]
            pairs.append(dict(recipient=i,donor=j,template=t,unit_answer=10*r["tens"]+d["unit"],tens_answer=10*d["tens"]+r["unit"],
                              base_answer=r["total"],donor_answer=d["total"],recipient_carry=r["carry"],donor_carry=d["carry"]))
            if sum(x["template"]==t for x in pairs)==cfg["pairs_per_template"]:break
    assert all(sum(x["template"]==t for x in pairs)==cfg["pairs_per_template"] for t in range(len(cfg["templates"])))
    return rows,pairs


def fisher(z,labels):
    import torch
    z=z.double();mean=z.mean(0);between=torch.zeros_like(mean);within=torch.zeros_like(mean)
    for label in sorted(set(labels)):
        x=z[[i for i,v in enumerate(labels) if v==label]];mu=x.mean(0)
        between+=len(x)*(mu-mean).square();within+=(x-mu).square().sum(0)
    return between/within.clamp_min(1e-10)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    sources=["scripts/run_arithmetic_digit_components.py","scripts/run_causalgym_multisite.py","scripts/run_r011s1_raw_hook_asset.py","src/ccad/artifacts.py"]
    if cfg.get("counterfactual_fit"):
        sources += ["scripts/arithmetic_counterfactual_fit.py", "src/ccad/semantic_participation.py"]
    if cfg.get("relation_transfer"):
        sources += ["scripts/arithmetic_relation_transfer.py", "scripts/fit_component_correspondence.py"]
    if cfg.get("response_relation"):
        sources += ["scripts/arithmetic_response_relation.py", "scripts/fit_component_correspondence.py"]
    if cfg.get("position_relation"):
        sources += ["scripts/arithmetic_position_relation.py", "scripts/fit_component_correspondence.py"]
    if cfg.get("member_queries"):
        sources += ["scripts/arithmetic_member_queries.py"]
    if cfg.get('query_readouts'):
        sources += ['scripts/arithmetic_readout_queries.py']
    if cfg.get('request_writer_fit') or cfg.get('request_writers'):
        sources += ['scripts/arithmetic_request_writer.py']
    if cfg.get('native_readouts'):
        sources += ['scripts/adaptive_native_execution.py']
        if any(n.get('response_steps') for n in cfg['native_readouts']):
            sources += ['scripts/native_response_projection.py']
    if cfg.get('functional_rule_test'):
        sources += ['scripts/arithmetic_functional_rules.py']
    if cfg.get('frozen_rule_evaluation'):
        sources += ['scripts/arithmetic_frozen_rules.py']
    if cfg.get('source_function_refit'):
        sources += ['scripts/arithmetic_carry_source_fit.py']
        if cfg['source_function_refit'].get('read_write'):
            sources += ['scripts/arithmetic_carry_readwrite.py','scripts/analyze_carry_readouts.py']
    if cfg.get('carry_relation_fit'):
        sources += ['scripts/arithmetic_carry_relation_fit.py','scripts/arithmetic_counterfactual_fit.py']
    if cfg.get('native_projection_steps'):
        sources += ['scripts/arithmetic_native_execution.py','src/ccad/native_operation.py']
    if cfg.get('source_function_refit',{}).get('state_write') or cfg.get('state_write_evaluation'):
        sources += ['scripts/arithmetic_state_write.py']
    w=MultisiteWork(cfg,args.config,sources)
    error=None
    try:
        import numpy as np, torch, transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision("high")
        w.torch=torch;w.device=torch.device(cfg["device"])
        tr=w.checked(ROOT/cfg["training_run"] / "config.resolved.json").parent;tc=json.loads((tr/"config.resolved.json").read_text())
        assert json.loads(w.checked(tr/"status.json").read_text())["status"]=="PASS"
        snapshots=json.loads(w.checked(tr/"checkpoints.json").read_text())["checkpoints"]
        for name in ["config.json","model.safetensors","tokenizer.json","tokenizer_config.json"]:w.checked(Path(tc["model_local_dir"])/name)
        tok=transformers.AutoTokenizer.from_pretrained(tc["model_local_dir"],local_files_only=True,trust_remote_code=False)
        model=transformers.AutoModelForCausalLM.from_pretrained(tc["model_local_dir"],local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation="eager").eval().to(w.device)
        model.requires_grad_(False);module=model.get_submodule(cfg.get("hook_override",tc["hook_module_path"]))
        sys.path.extend([tc["dictionary_source_dir"],tc["dictionary_overlay_dir"]])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(tc["dictionary_source_dir"])/"dictionary_learning/trainers/top_k.py")
        w.checked(Path(tc["dictionary_source_dir"])/"LICENSE")
        saes={}
        for seed in cfg["seeds"]:
            snap=next(s for s in snapshots if s["seed"]==seed and s["objective"]=="topk" and s["step"]==cfg["checkpoint_step"])
            path=w.checked(snap["path"]);assert sha256(path)==snap["sha256"]
            ae=AutoEncoderTopK(model.config.hidden_size,tc["dict_size"],tc["k"]).to(w.device)
            ae.load_state_dict(torch.load(path,map_location=w.device,weights_only=True));ae.eval();ae.requires_grad_(False);saes[seed]=ae
        if cfg.get('frozen_rule_evaluation'):
            w.checked(ROOT/cfg['frozen_rule_evaluation']['panel'])
            w.checked(ROOT/cfg['frozen_rule_evaluation']['freeze'])
        rows,pairs=panel(cfg);write(w.run/"panel.json",dict(rows=rows,pairs=pairs,scope=cfg["scope"]))
        if cfg.get("evaluation_operand_pairs") is not None:
            assert (cfg.get("frozen_adaptation") or cfg.get('member_queries')) and not cfg.get("counterfactual_fit")
            exclusion_runs=cfg.get('evaluation_exclusion_runs')
            if exclusion_runs is None:
                exclusion_runs=[cfg["frozen_adaptation"]["development_panel_run"]]
            for prior_run in exclusion_runs:
                prior_panel=json.loads(w.checked(ROOT/prior_run/'panel.json').read_text())
                if cfg.get('evaluation_exclusion_unit')=='prompt':
                    prior_text=set()
                    for r in prior_panel['rows']:
                        texts=[r[k] for k in ['prompt','sentence_good','sentence_bad'] if k in r]
                        assert texts, 'Unrecognized historical prompt schema'
                        prior_text.update(texts)
                    assert not {r['prompt'] for r in rows}&prior_text
                else:
                    assert not {tuple(sorted((r['a'],r['b']))) for r in rows}&{tuple(sorted((r['a'],r['b']))) for r in prior_panel['rows']}
            if cfg.get('evaluation_exclusion_unit')=='prompt':
                w.checks['prompts_disjoint_from_designated_development_panels']=True
            else:
                w.checks["operand_questions_disjoint_from_designated_development_panel"]=True
        tokenrows=[tok.encode(r["prompt"],add_special_tokens=False) for r in rows]
        batch=cfg["batch_size"];padding=cfg["max_length"]
        physical_length=max(map(len,tokenrows))+cfg["max_new_tokens"]
        assert physical_length<=padding
        def budget():
            if time.perf_counter()-w.wall_start>cfg["budget_seconds"]:raise TimeoutError("Arithmetic source development budget")
        trajectory=cfg.get("intervention_span")=="generated_prefix"
        def generate(indices,delta=None,patch=None,response=None):
            padded=indices+[indices[0]]*(batch-len(indices));ids=torch.full((batch,padding),tok.eos_token_id,device=w.device,dtype=torch.long)
            mask=torch.zeros_like(ids);pos=[]
            for j,i in enumerate(padded):
                tokens=tokenrows[i];ids[j,:len(tokens)]=torch.tensor(tokens,device=w.device);mask[j,:len(tokens)]=1;pos.append(len(tokens)-1)
            pos=torch.tensor(pos,device=w.device);ix=torch.arange(batch,device=w.device);cache={"trajectory":[]};generated=[];finished=torch.zeros(batch,device=w.device,dtype=torch.bool)
            def hook(m,a,out):
                h=out[0] if isinstance(out,tuple) else out
                if "hidden" not in cache:cache["hidden"]=h[ix,pos].detach().clone()
                cache["trajectory"].append(h[ix,pos+step].detach().clone())
                if patch is not None:
                    sites=pos[:len(indices),None]+torch.arange(step+1,device=w.device)[None,:]
                    current=h[ix[:len(indices),None],sites]
                    change=patch(current,step)
                    hh=h.clone();hh[ix[:len(indices),None],sites]+=change
                    cache["edit_norm"]=change.flatten(1).norm(dim=1)
                elif delta is not None:
                    hh=h.clone();hh[ix[:len(indices)],pos[:len(indices)]]+=delta
                    cache["edit_norm"]=delta.norm(dim=1)
                else:return out
                return (hh,)+out[1:] if isinstance(out,tuple) else hh
            handle=module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    for step in range(cfg["max_new_tokens"]):
                        length=physical_length
                        output=model(ids[:,:length],attention_mask=mask[:,:length],use_cache=False)
                        logits=output.logits[ix,pos+step]
                        if response is not None:
                            from native_response_projection import refine
                            assert len(indices)==batch
                            handle.remove()
                            try:
                                logits,realized,diagnostic=refine(model,module,ids[:,:length],mask[:,:length],
                                    pos[:,None]+torch.arange(step+1,device=w.device)[None,:],pos+step,logits,
                                    response['decoder'],response['initial'],response['lower'],response['field'],w,
                                    steps=response['spec']['response_steps'],lr=response['spec']['response_lr'],
                                    field_anchor=response['spec'].get('field_anchor',0.),
                                    parameterization=response['spec'].get('parameterization','members'))
                                response['diagnostics'].append(dict(step=step,**diagnostic))
                                cache['edit_norm']=realized.flatten(1).norm(dim=1)
                            finally:handle=module.register_forward_hook(hook)
                        nxt=logits.argmax(-1)
                        nxt=torch.where(finished,torch.full_like(nxt,tok.eos_token_id),nxt);finished|=nxt==tok.eos_token_id;generated.append(nxt)
                        ids[ix,pos+step+1]=nxt;mask[ix,pos+step+1]=1
                        w.sequence_forwards+=batch;w.token_forwards+=batch*length
            finally:handle.remove()
            text=tok.batch_decode(torch.stack(generated,1),skip_special_tokens=True)
            answers=[]
            for s in text[:len(indices)]:
                match=re.match(r"\s*(\d+)",s);answers.append(int(match[1]) if match else None)
            return answers,text[:len(indices)],(torch.stack(cache["trajectory"],1) if trajectory else cache["hidden"])[:len(indices)],cache.get("edit_norm",torch.zeros(len(indices),device=w.device))
        all_h=[];base=[];base_text=[];source_parent=None
        if cfg.get("source_cache_run"):
            source_parent=ROOT/cfg["source_cache_run"]
            assert json.loads(w.checked(source_parent/"status.json").read_text())["status"]=="PASS"
            old_cfg=json.loads(w.checked(source_parent/"config.resolved.json").read_text())
            assert all(old_cfg[k]==cfg[k] for k in ["training_run","checkpoint_step","model_revision","intervention_span","max_new_tokens"])
            saved_panel=json.loads(w.checked(source_parent/"panel.json").read_text())
            assert saved_panel["rows"]==rows
            if cfg.get("evaluation_holdout"):
                old_ids={i for p in saved_panel["pairs"] for i in [p["recipient"],p["donor"]]}
                new_ids={i for p in pairs for i in [p["recipient"],p["donor"]]}
                assert not old_ids.intersection(new_ids)
                w.checked(ROOT/cfg["evaluation_holdout"]["from_run"]/"panel.json")
                w.checks["new_intervention_operands_disjoint_from_development_pairs"]=True
            else:assert saved_panel["pairs"]==pairs
            with np.load(w.checked(source_parent/"states.npz")) as data:
                h=torch.tensor(data["hidden"],device=w.device)
            old_base=[json.loads(x) for x in w.checked(source_parent/"metrics.raw.jsonl").read_text().splitlines() if json.loads(x)["kind"]=="base"]
            old_base.sort(key=lambda r:r["row_id"])
            assert [r["row_id"] for r in old_base]==list(range(len(rows)))
            base=[r["answer"] for r in old_base];base_text=[r["generated_text"] for r in old_base]
        else:
            for off in range(0,len(rows),batch):
                indices=list(range(off,min(off+batch,len(rows))));ans,txt,h,_=generate(indices);all_h.append(h);base.extend(ans);base_text.extend(txt)
                if off%(batch*16)==0:w.progress("SOURCE_STATES",completed=off+len(indices),total=len(rows))
                budget()
            h=torch.cat(all_h)
        view_map=None;original_count=len(rows)
        if cfg.get("source_view_augmentation"):
            assert source_parent is not None and trajectory
            vc=cfg['source_view_augmentation'] if isinstance(cfg['source_view_augmentation'],dict) else {}
            view_template=vc.get('template',0)
            view_map={};view_by_total={}
            for i in range(original_count):
                r=rows[i]
                if r['split']!='fit':continue
                answer=r['total']
                if answer not in view_by_total:
                    a,b=1,answer-1
                    assert not any(q['a']==a and q['b']==b for q in rows[:original_count])
                    v=dict(template=view_template,a=a,b=b,total=answer,unit=answer%10,tens=answer//10,
                           carry=(a%10+b%10)>=10,split='fit',source_view=True,
                           prompt=cfg['templates'][view_template].format(a=a,b=b))
                    view_by_total[answer]=len(rows);rows.append(v)
                    tokenrows.append(tok.encode(v['prompt'],add_special_tokens=False))
                    assert len(tokenrows[-1])+cfg['max_new_tokens']<=physical_length
                view_map[i]=view_by_total[answer]
            view_h=[]
            for off in range(original_count,len(rows),batch):
                indices=list(range(off,min(off+batch,len(rows))))
                ans,txt,hh,_=generate(indices);view_h.append(hh);base.extend(ans);base_text.extend(txt)
            h=torch.cat([h,torch.cat(view_h)])
            if vc.get('require_base_correct'):
                view_map={i:(v if base[i]==rows[i]['total'] and base[v]==rows[v]['total'] else i)
                          for i,v in view_map.items()}
            write(w.run/'panel.json',dict(rows=rows,pairs=pairs,scope=cfg['scope'],source_view_map=view_map))
            np.savez_compressed(w.run/'source_view_states.npz',hidden=h[original_count:].cpu().numpy(),
                                original_indices=np.array(list(view_map)),view_indices=np.array(list(view_map.values())))
            w.progress('SOURCE_EQUIVALENT_VIEWS',original_fit_rows=len(view_map),new_questions=len(view_by_total),
                       changed_fit_rows=sum(i!=v for i,v in view_map.items()),
                       correct=sum(base[i]==rows[i]['total'] for i in view_by_total.values()))
        fitix=[i for i,r in enumerate(rows) if r["split"]=="fit"]
        for i,r in enumerate(rows):w.record(kind="base",task="template_"+str(r["template"]),row_id=i,component='+'.join(str(r[k]) for k in ['a','b','c'] if k in r),method="greedy",split=r["split"],answer=base[i],correct_answer=r["total"],correct=base[i]==r["total"],generated_text=base_text[i])
        codes={};gates={};metadata=[]
        for seed,ae in saes.items():
            if source_parent is not None:
                with np.load(w.checked(source_parent/f"source_seed{seed}.npz")) as data:
                    codes[seed]=torch.tensor(data["codes"],device=w.device)
                    for rule in cfg["reuse_rules"]:
                        for k in cfg["members"]:
                            gates[seed,rule,k]=torch.tensor(data[f"{rule}_{k}"],device=w.device)
                if view_map is not None:
                    with torch.no_grad():
                        extra=ae.encode(h[original_count:].reshape(-1,h.shape[-1])).reshape(len(rows)-original_count,h.shape[1],-1)
                    codes[seed]=torch.cat([codes[seed],extra])
                continue
            with torch.no_grad():z=torch.cat([ae.encode(h[i:i+256].reshape(-1,h.shape[-1])).reshape(*h[i:i+256].shape[:-1],-1) for i in range(0,len(h),256)])
            codes[seed]=z
            if cfg.get("evaluation_operand_pairs") is not None or cfg.get('frozen_rule_evaluation'):
                np.savez_compressed(w.run/f"evaluation_seed{seed}.npz",codes=z.cpu().numpy())
                continue
            fs=torch.stack([fisher(z[fitix,cfg["source_selection_step"][factor]] if trajectory else z[fitix],[rows[i][factor] for i in fitix]) for factor in ["unit","tens"]],1)
            for rule in cfg["selection_rules"]:
                score=fs if rule=="fisher" else fs-fs.flip(1)
                for k in cfg["members"]:
                    gate=torch.zeros((z.shape[-1],2),device=w.device)
                    for c in range(2):gate[torch.argsort(score[:,c],descending=True,stable=True)[:k],c]=1
                    gates[seed,rule,k]=gate
                    metadata.append(dict(seed=seed,rule=rule,members=k,unit_members=torch.where(gate[:,0]>0)[0].cpu().tolist(),tens_members=torch.where(gate[:,1]>0)[0].cpu().tolist(),overlap=int((gate[:,0]*gate[:,1]).sum())))
            if cfg.get("counterfactual_fit"):
                from arithmetic_counterfactual_fit import fit_gates
                assert trajectory
                for k in cfg["members"]:
                    learned=fit_gates(w,cfg,model,module,tok,ae,rows,tokenrows,z,
                                      gates[seed,"fisher_contrast",k],seed,k,physical_length,budget)
                    gates[seed,"counterfactual_weighted",k]=learned
                    gates[seed,"counterfactual_binary",k]=(learned>0).to(learned.dtype)
                    for rule in ["counterfactual_weighted","counterfactual_binary"]:
                        gate=gates[seed,rule,k]
                        metadata.append(dict(seed=seed,rule=rule,members=k,
                                             unit_members=torch.where(gate[:,0]>0)[0].cpu().tolist(),
                                             tens_members=torch.where(gate[:,1]>0)[0].cpu().tolist(),
                                             overlap=int(((gate[:,0]>0)&(gate[:,1]>0)).sum())))
            payload={f"{rule}_{k}":gate.cpu().numpy() for (s,rule,k),gate in gates.items() if s==seed}
            np.savez_compressed(w.run/f"source_seed{seed}.npz",codes=z.cpu().numpy(),fisher=fs.cpu().numpy(),**payload)
        if source_parent is None:
            np.savez_compressed(w.run/"states.npz",hidden=h.cpu().numpy(),fit_indices=np.array(fitix))
        if cfg.get('frozen_rule_evaluation'):
            from arithmetic_frozen_rules import evaluate_frozen_rules
            w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
                numpy=np.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
                precision='float32 matmul high',hook=cfg.get('hook_override',tc['hook_module_path']),model=tc['model_id'])
            evaluate_frozen_rules(w,cfg,rows,h,codes,saes,generate,budget,base,model_context=(model,module,tok))
            return w.finish()
        if cfg.get('functional_rule_test'):
            from arithmetic_functional_rules import evaluate_rules
            w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
                numpy=np.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
                precision='float32 matmul high',hook=cfg.get('hook_override',tc['hook_module_path']),model=tc['model_id'])
            evaluate_rules(w,cfg,rows,h,codes,saes,generate,budget,base)
            return w.finish()
        if cfg.get("source_refit"):
            from arithmetic_counterfactual_fit import fit_gates, natural_latent_reference
            sc=cfg["source_refit"]
            latent=(natural_latent_reference(w,rows,h,base,rank=sc["latent_rank"])
                    if sc.get('use_latent_reference',True) else None)
            refitted={};k=cfg["members"][0]
            for seed in cfg["seeds"]:
                for name,weight in sc["objectives"].items():
                    fit_cfg={**cfg,'counterfactual_fit':{**cfg['counterfactual_fit'],**sc.get('fit_overrides',{}).get(name,{})}}
                    learned=fit_gates(w,fit_cfg,model,module,tok,saes[seed],rows,tokenrows,codes[seed],
                                      gates[seed,"fisher_contrast",k],seed,k,physical_length,budget,
                                      tag=name,latent_context=latent,latent_weight=weight,view_map=view_map,
                                      selection_batches=sc.get('selection_batches',{}).get(name,0))
                    refitted[seed,name,k]=learned
                    metadata.append(dict(seed=seed,source_objective=name,latent_weight=weight,
                                         target_dictionaries_used=False,updates=cfg["counterfactual_fit"]["steps"]))
            gates=refitted
        if cfg.get("relation_transfer"):
            from arithmetic_relation_transfer import fit_relations
            transferred,relation_meta=fit_relations(w,cfg,saes,codes,gates,h,source_parent,generate,budget,
                                                  rows=rows,view_map=view_map)
            gates.update(transferred);metadata.extend(relation_meta)
        if cfg.get("response_relation"):
            from arithmetic_response_relation import fit_response_relations
            gates,response_meta=fit_response_relations(w,cfg,model,module,tok,rows,tokenrows,
                                                       saes,codes,physical_length,budget)
            metadata.extend(response_meta)
        if cfg.get("position_relation"):
            from arithmetic_position_relation import fit_positions
            gates,position_meta=fit_positions(w,cfg,saes,codes,rows,budget)
            metadata.extend(position_meta)
        if cfg.get("adaptation"):
            from arithmetic_counterfactual_fit import fit_gates
            ac=cfg["adaptation"];relation_parent=ROOT/ac["relation_run"]
            assert json.loads(w.checked(relation_parent/"status.json").read_text())["status"]=="PASS"
            prior=json.loads(w.checked(relation_parent/"config.resolved.json").read_text())
            assert prior["source_cache_run"]==cfg["source_cache_run"]
            relation_panel=json.loads(w.checked(relation_parent/"panel.json").read_text())
            assert relation_panel["rows"]==rows and relation_panel["pairs"]==pairs
            adapted={};k=cfg["members"][0]
            for source,seed in prior["relation_transfer"]["seed_pairs"]:
                with np.load(w.checked(relation_parent/f"relation_s{source}_t{seed}.npz")) as data:
                    starts={name:torch.tensor(data[name],device=w.device) for name in ["clean","assignment"]}
                starts["fisher_contrast"]=gates[seed,"fisher_contrast",k]*.5
                requested=ac.get("initializations",list(starts))
                if "clean_swapped" in requested:starts["clean_swapped"]=starts["clean"].flip(1)
                if "clean_fixed" in requested:starts["clean_fixed"]=starts["clean"].clone()
                if "clean_swapped_fixed" in requested:starts["clean_swapped_fixed"]=starts["clean"].flip(1)
                for name in ["target_gradient","target_integrated_gradient","source_path_gradient","source_swapped_path_gradient"]:
                    if name in requested:starts[name]=starts["fisher_contrast"].clone()
                for name in requested:
                    if name.startswith("target_gradient_s"):starts[name]=starts["fisher_contrast"].clone()
                for name in requested:
                    initial=starts[name]
                    fit_cfg=cfg
                    if name in ac.get("steps_by_initialization",{}):
                        fit_cfg={**cfg,"counterfactual_fit":{**cfg["counterfactual_fit"],"steps":ac["steps_by_initialization"][name]}}
                    saved_steps=ac.get("updates_by_initialization",{}).get(name,ac["updates"])
                    source_context=None
                    if name in ["source_path_gradient","source_swapped_path_gradient"]:
                        sg=gates[source,"counterfactual_weighted",k]
                        if name=="source_swapped_path_gradient":sg=sg.flip(1)
                        source_context=(saes[source],codes[source],sg)
                    _,saved=fit_gates(w,fit_cfg,model,module,tok,saes[seed],rows,tokenrows,codes[seed],
                                     initial,seed,k,physical_length,budget,tag=name,initial_scale=1.,checkpoints=saved_steps,
                                     selection_batches=ac.get("gradient_selection_by_initialization",{}).get(name,ac.get("gradient_selection_batches",0)) if (name.startswith("target_") or source_context is not None) else 0,
                                     integrated_steps=ac.get("integrated_steps",4) if (name=="target_integrated_gradient" or source_context is not None) else 1,
                                     fixed_support=name.endswith("fixed"),source_context=source_context)
                    for updates,g in saved.items():adapted[seed,f"adapt_{name}_u{updates}",k]=g
                    metadata.append(dict(source_seed=source,target_seed=seed,initialization=name,updates=saved_steps,
                                         optimization="Same deterministic labelled fit bank; alternating requests. Direct gradient selection is included in total backward budget; IG repeats batches at each midpoint. Shared labels already available."))
            gates=adapted
        if cfg.get("frozen_adaptation"):
            fc=cfg["frozen_adaptation"]
            loaded={};k=cfg["members"][0]
            for seed in cfg["seeds"]:
                for name in fc["initializations"]:
                    prior=ROOT/fc.get('initialization_seed_runs',{}).get(name,{}).get(
                        str(seed),fc.get("initialization_runs",{}).get(name,fc.get("run","")))
                    assert json.loads(w.checked(prior/"status.json").read_text())["status"]=="PASS"
                    pc=json.loads(w.checked(prior/"config.resolved.json").read_text())
                    assert pc.get("source_cache_run",prior.relative_to(ROOT).as_posix())==cfg.get("source_cache_run",fc.get("source_cache_identity"))
                    assert all(pc[key]==cfg[key] for key in ["model_revision","training_run","checkpoint_step","intervention_span"])
                    pattern=fc.get("file_patterns",{}).get(name,"counterfactual_seed{seed}_k{members}_"+name+".npz")
                    source=fc.get('source_seed_for_target',{}).get(str(seed),seed)
                    maskfile=w.checked(prior/pattern.format(seed=seed,members=k,source=source))
                    if fc.get("mask_sha256"):assert sha256(maskfile)==fc["mask_sha256"][maskfile.relative_to(ROOT).as_posix()]
                    with np.load(maskfile) as data:
                        for u in fc["updates"]:
                            array_key=fc.get("array_keys",{}).get(name,{}).get(str(u),f"updates_{u}")
                            loaded[seed,f"adapt_{name}_u{u}",k]=torch.tensor(data[array_key],device=w.device)
            gates=loaded
            metadata.append(dict(frozen_adaptation=fc,current_fit_updates=0,current_output_gradients=0,
                                 original_source_and_target_fit_labels_shared=True))
        if cfg.get('member_queries'):
            from arithmetic_member_queries import prepare_queries
            gates,query_meta=prepare_queries(w,cfg,saes)
            metadata.append(query_meta)
        readouts={}
        if cfg.get('query_readouts'):
            from arithmetic_readout_queries import prepare_readouts
            readouts,readout_meta=prepare_readouts(w,cfg)
            metadata.append(dict(query_readouts=readout_meta))
        if cfg.get('request_writer_fit'):
            from arithmetic_request_writer import fit_writers
            fit_writers(w,cfg,model,module,tok,saes)
            metadata.append(dict(request_writer_fit=json.loads((w.run/'REQUEST_WRITER_FIT.json').read_text()),
                                 source_fit_contexts_only=True,output_response_supervision=True))
        if cfg.get('request_writers'):
            from arithmetic_request_writer import attach_writers
            attach_writers(w,cfg,readouts)
        write(w.run/"SOURCE_FREEZE.json",dict(written_at_utc=datetime.now(timezone.utc).isoformat(),metadata=metadata,target_dictionaries_used=bool(cfg.get("relation_transfer") or cfg.get("response_relation") or cfg.get("position_relation") or cfg.get("adaptation") or cfg.get("frozen_adaptation") or cfg.get("member_queries")),task_output_gradients="inherited shared two-role response bank; no new gradients; see POSITION_FREEZE.json" if cfg.get("position_relation") else "shared fit-pair requested/preserved margins; see RESPONSE_FREEZE.json" if cfg.get("response_relation") else "fit split hybrid CE" if cfg.get("counterfactual_fit") else "inherited source and target fit only" if cfg.get("frozen_adaptation") else "inherited source fit only" if source_parent else 0,source_functional_outcomes_used_for_selection=bool(cfg.get("counterfactual_fit") or source_parent),development_functional_outcomes_used_for_selection=False,base_outputs_already_observed=True,files=[dict(path=p.name,sha256=sha256(p)) for p in w.run.glob("source_seed*.npz")]))
        completed_keys=set()
        if cfg.get('resume_interventions'):
            previous=ROOT/cfg['resume_interventions']['run']
            previous_cfg=json.loads(w.checked(previous/'config.resolved.json').read_text())
            for key in ['model_revision','training_run','checkpoint_step','source_cache_run','seeds','templates','members','member_queries','max_new_tokens','intervention_span']:
                assert previous_cfg[key]==cfg[key],key
            previous_panel=json.loads(w.checked(previous/'panel.json').read_text())
            current_panel=json.loads((w.run/'panel.json').read_text())
            # Scope is descriptive prose; rows, pairs and any view identities
            # must remain byte-equivalent after JSON decoding.
            assert {k:v for k,v in previous_panel.items() if k!='scope'}=={k:v for k,v in current_panel.items() if k!='scope'}
            rawpath=w.checked(previous/'metrics.raw.jsonl')
            assert sha256(rawpath)==cfg['resume_interventions']['raw_sha256']
            for line in rawpath.read_text().splitlines():
                row=json.loads(line)
                if row['kind']=='source_patch':
                    key=(row['seed'],row['method'],row['operation'],row['row_id'])
                    assert key not in completed_keys
                    completed_keys.add(key)
            w.checks['resume_scientific_inputs_and_panel_identical']=True
        native_diagnostics=[]
        def evaluate(seed,method,k,operation,deltas,gate=None,raw_trajectory=False,readout=None,dose=None,full_gate=None,native=None):
            for off in range(0,len(pairs),batch):
                pp=pairs[off:off+batch];ix=[p["recipient"] for p in pp];donors=[p["donor"] for p in pp]
                if all((seed,method,operation,off+j) in completed_keys for j in range(len(pp))):
                    continue
                patch=None
                response=dict(spec=native,diagnostics=[]) if native and native.get('response_steps') else None
                if raw_trajectory:patch=lambda current,step:h[donors,:step+1]-current
                elif readout is not None:
                    ae=saes[seed]
                    def patch(current,step):
                        if native is not None or readout['kind']!='raw':
                            target_current=ae.encode(current.reshape(-1,current.shape[-1])).reshape(*current.shape[:-1],-1)
                            code_difference=codes[seed][donors,:step+1]-target_current
                        if readout['kind']=='raw':
                            difference=h[donors,:step+1]-current
                        elif readout['kind']=='reconstruction':
                            difference=code_difference@ae.decoder.weight.T
                        elif readout['kind']=='full_activation':
                            difference=code_difference
                        else:
                            difference=code_difference[...,readout['indices']]
                        shift=torch.tensor([1-rows[i]['template'] for i in ix],device=w.device)
                        roles=torch.arange(step+1,device=w.device)[None]+shift[:,None]
                        predicted=torch.einsum('blj,blji->bli',difference,readout['coef'][roles])
                        field=(predicted*readout['q'])@readout['decoder']
                        if readout.get('writer'):
                            writer=readout['writer']
                            coeff=torch.einsum('bli,blij->blj',predicted*readout['q'],writer['matrix'][roles])
                            if writer['kind']=='native':
                                bank=readout['indices']
                                coeff=torch.maximum(coeff,-target_current[...,bank])
                                native_diagnostics.append(dict(seed=seed,method=method,operation=operation,
                                    offset=off,step=step,min_edited_code=float((target_current[...,bank]+coeff).min()),
                                    max_changed_members=int((coeff!=0).sum(-1).max()),runtime_output_gradients=0))
                            field=coeff@writer['decoder']
                        if dose is not None:
                            assert not dose.get('nonnegative',False)
                            if dose['kind']=='norm_match':
                                full_field=predicted@readout['decoder']
                                scale=full_field.norm(dim=-1,keepdim=True)/field.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                            else:scale=float(dose['scale'])
                            field=field*scale
                        if native is not None:
                            from adaptive_native_execution import realize
                            bank=readout['indices'];decoder=ae.decoder.weight.T[bank]
                            source_field=field
                            field,coeff,indices,diagnostic=realize(field,code_difference[...,bank],decoder,
                                members=k,steps=native['steps'],refine_steps=0,
                                batch_size=128,candidate_limit=len(bank),
                                current_codes=target_current[...,bank])
                            diagnostic.update(seed=seed,method=method,operation=operation,reader_kind=readout['kind'],
                                offset=off,step=step,positions=coeff.numel()//len(bank),
                                min_edited_code=float((target_current[...,bank].gather(-1,indices)+coeff).min()),
                                max_changed_members=int((coeff!=0).sum(-1).max()))
                            native_diagnostics.append(diagnostic)
                            if response is not None:
                                initial=torch.zeros_like(target_current[...,bank]).scatter(-1,indices,coeff)
                                response.update(initial=initial,lower=-target_current[...,bank],
                                                field=source_field,decoder=decoder)
                                return source_field
                        return field
                elif gate is not None:
                    ae=saes[seed]
                    role_schema=any(cfg.get(key,{}).get('role_schema',False) for key in ['position_relation','relation_transfer','frozen_adaptation','counterfactual_fit','member_queries'])
                    def patch(current,step):
                        g=gate
                        if gate.ndim==2:
                            if role_schema:
                                shift=torch.tensor([1-rows[i]['template'] for i in ix],device=w.device)
                                g=gate[torch.arange(step+1,device=w.device)[None]+shift[:,None]]
                            else:g=gate[:step+1]
                        current_code=ae.encode(current.reshape(-1,current.shape[-1])).reshape(*current.shape[:-1],-1)
                        difference=codes[seed][donors,:step+1]-current_code
                        change=difference*g
                        if dose is not None:
                            if dose['kind']=='norm_match':
                                fg=full_gate
                                if full_gate.ndim==2:
                                    fg=full_gate[torch.arange(step+1,device=w.device)[None]+shift[:,None]] if role_schema else full_gate[:step+1]
                                full_field=(difference*fg)@ae.decoder.weight.T
                                part_field=change@ae.decoder.weight.T
                                scale=full_field.norm(dim=-1,keepdim=True)/part_field.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                            else:
                                scale=float(dose['scale'])
                            change=change*scale
                            if dose.get('nonnegative',False):
                                change=torch.maximum(change,-current_code)
                        return change@ae.decoder.weight.T
                ans,txt,hh,norm=generate(ix,deltas[off:off+len(pp)] if deltas is not None else None,patch,response=response)
                if response is not None:
                    response_diagnostics.append(dict(seed=seed,method=method,operation=operation,offset=off,records=response['diagnostics']))
                replay=float(((hh[:,0]-h[ix,0]) if trajectory else (hh-h[ix])).abs().max());assert replay<cfg["hidden_atol"],replay
                for j,p in enumerate(pp):
                    if (seed,method,operation,off+j) in completed_keys:
                        continue
                    expected=p[operation+"_answer"] if operation in ["unit","tens"] else p["donor_answer"]
                    a=ans[j];numeric=a is not None and 10<=a<100
                    target=(a%10==p["donor_answer"]%10 if operation=="unit" else a//10==p["donor_answer"]//10) if numeric else False
                    preserve=(a//10==p["base_answer"]//10 if operation=="unit" else a%10==p["base_answer"]%10) if numeric else False
                    w.record(kind="source_patch",task="template_"+str(p["template"]),row_id=off+j,component=str(p["recipient"])+"<-"+str(p["donor"]),seed=seed,method=method,mode="k"+str(k),operation=operation,
                             answer=a,expected_answer=expected,exact_hybrid=a==expected,target_digit_success=target,preserve_digit_success=preserve,
                             base_answer=p["base_answer"],donor_answer=p["donor_answer"],base_correct=base[p["recipient"]]==p["base_answer"],donor_correct=base[p["donor"]]==p["donor_answer"],
                             recipient_carry=p["recipient_carry"],donor_carry=p["donor_carry"],generated_text=txt[j],hidden_replay_error=replay,edit_norm=float(norm[j]))
                budget()
            w.progress("SOURCE_PATCH",seed=seed,method=method,members=k,operation=operation)
        response_diagnostics=[]
        ii=[p["recipient"] for p in pairs];jj=[p["donor"] for p in pairs]
        evaluate(0,"no_edit",0,"unit",None)
        if trajectory:
            evaluate(0,"raw_prompt_patch",model.config.hidden_size,"unit",h[jj,0]-h[ii,0])
            evaluate(0,"raw_generated_prefix_patch",model.config.hidden_size,"unit",None,raw_trajectory=True)
        else:evaluate(0,"raw_full_patch",model.config.hidden_size,"unit",h[jj]-h[ii])
        for (seed,rule,k),gate in gates.items():
            if cfg.get('evaluation_seeds') and seed not in cfg['evaluation_seeds']:
                continue
            if cfg.get('evaluation_rules') and rule not in cfg['evaluation_rules']:
                continue
            z=codes[seed];D=saes[seed].decoder.weight.T
            for c,operation in enumerate(["unit","tens"]):
                if trajectory:evaluate(seed,rule,k,operation,None,gate=gate[...,c])
                else:
                    delta=((z[jj]-z[ii])*gate[:,c])@D
                    evaluate(seed,rule,k,operation,delta)
        for dose in cfg.get('part_doses',[]):
            assert trajectory
            for (seed,rule,k),gate in gates.items():
                if rule not in dose['rules'] or (cfg.get('evaluation_seeds') and seed not in cfg['evaluation_seeds']):
                    continue
                full_gate=gates[seed,dose.get('full_rule','source_full'),k]
                for c,operation in enumerate(['unit','tens']):
                    evaluate(seed,rule+'_'+dose['name'],k,operation,None,gate=gate[...,c],dose=dose,full_gate=full_gate[...,c])
        for (seed,rule,k),operations in readouts.items():
            if cfg.get('evaluation_seeds') and seed not in cfg['evaluation_seeds']:
                continue
            if cfg.get('evaluation_rules') and rule not in cfg['evaluation_rules']:
                continue
            for operation,data in operations.items():
                evaluate(seed,rule,k,operation,None,readout=data)
            for dose in cfg.get('readout_doses',[]):
                if rule in dose['rules']:
                    for operation,data in operations.items():
                        evaluate(seed,rule+'_'+dose['name'],k,operation,None,readout=data,dose=dose)
        for native in cfg.get('native_readouts',[]):
            for (seed,rule,k),operations in readouts.items():
                if seed not in native['seeds'] or rule not in native['rules']:
                    continue
                native_rule=native.get('name','native_code')+rule.split('_readout',1)[1]
                for operation,data in operations.items():
                    evaluate(seed,native_rule,k,operation,None,readout=data,native=native)
                for dose in native.get('doses',[]):
                    if rule in dose['rules']:
                        for operation,data in operations.items():
                            evaluate(seed,native_rule+'_'+dose['name'],k,operation,None,readout=data,native=native,dose=dose)
        if native_diagnostics:
            write(w.run/'NATIVE_EXECUTION.json',dict(records=native_diagnostics,
                scope='Euclidean initial solutions in a fixed target bank, with nonnegative target code updates. For response_code methods these initialize additional output fitting recorded in RESPONSE_EXECUTION.json.'))
        if response_diagnostics:
            write(w.run/'RESPONSE_EXECUTION.json',dict(records=response_diagnostics,
                scope='Current-prefix source-readout output distillation into target codes. Runtime model outputs and gradients are used; no correct task answers are supplied.'))
        w.checks.update(source_selection_only_fit_data=True,donor_changes_both_digits=True,real_generation_no_answer_prefix=True,all_failed_base_cases_retained=True)
        if not any(cfg.get(k) for k in ["relation_transfer","response_relation","position_relation","adaptation","frozen_adaptation","member_queries"]):
            w.checks["no_target_used"]=True
        if not cfg.get("counterfactual_fit") and source_parent is None:
            w.checks["source_selection_only_fit_labels_and_activations"]=True
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),precision="float32 matmul high",hook=cfg.get("hook_override",tc["hook_module_path"]),model=tc["model_id"],forward_accounting="Actual no-cache generation forward calls including padded batch members and lengths")
    except Exception as exc:
        error=repr(exc);(w.run/"traceback.log").write_text(traceback.format_exc())
    return w.finish(error)


if __name__=="__main__":raise SystemExit(main())
