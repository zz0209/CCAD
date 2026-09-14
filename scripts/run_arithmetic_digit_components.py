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
        rows,pairs=panel(cfg);write(w.run/"panel.json",dict(rows=rows,pairs=pairs,scope=cfg["scope"]))
        if cfg.get("evaluation_operand_pairs") is not None:
            assert cfg.get("frozen_adaptation") and not cfg.get("counterfactual_fit")
            prior_panel=json.loads(w.checked(ROOT/cfg["frozen_adaptation"]["development_panel_run"]/"panel.json").read_text())
            assert not {(r["a"],r["b"]) for r in rows}&{(r["a"],r["b"]) for r in prior_panel["rows"]}
            w.checks["new_operand_questions_disjoint_from_all_source_fit_and_prior_evaluation"]=True
        tokenrows=[tok.encode(r["prompt"],add_special_tokens=False) for r in rows]
        batch=cfg["batch_size"];padding=cfg["max_length"]
        physical_length=max(map(len,tokenrows))+cfg["max_new_tokens"]
        assert physical_length<=padding
        def budget():
            if time.perf_counter()-w.wall_start>cfg["budget_seconds"]:raise TimeoutError("Arithmetic source development budget")
        trajectory=cfg.get("intervention_span")=="generated_prefix"
        def generate(indices,delta=None,patch=None):
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
                        logits=output.logits[ix,pos+step];nxt=logits.argmax(-1)
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
        fitix=[i for i,r in enumerate(rows) if r["split"]=="fit"]
        for i,r in enumerate(rows):w.record(kind="base",task="template_"+str(r["template"]),row_id=i,component=f"{r['a']}+{r['b']}",method="greedy",split=r["split"],answer=base[i],correct_answer=r["total"],correct=base[i]==r["total"],generated_text=base_text[i])
        codes={};gates={};metadata=[]
        for seed,ae in saes.items():
            if source_parent is not None:
                with np.load(w.checked(source_parent/f"source_seed{seed}.npz")) as data:
                    codes[seed]=torch.tensor(data["codes"],device=w.device)
                    for rule in cfg["reuse_rules"]:
                        for k in cfg["members"]:
                            gates[seed,rule,k]=torch.tensor(data[f"{rule}_{k}"],device=w.device)
                continue
            with torch.no_grad():z=torch.cat([ae.encode(h[i:i+256].reshape(-1,h.shape[-1])).reshape(*h[i:i+256].shape[:-1],-1) for i in range(0,len(h),256)])
            codes[seed]=z
            if cfg.get("evaluation_operand_pairs") is not None:
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
        if cfg.get("relation_transfer"):
            from arithmetic_relation_transfer import fit_relations
            transferred,relation_meta=fit_relations(w,cfg,saes,codes,gates,h,source_parent,generate,budget)
            gates.update(transferred);metadata.extend(relation_meta)
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
                    prior=ROOT/fc.get("initialization_runs",{}).get(name,fc.get("run",""))
                    assert json.loads(w.checked(prior/"status.json").read_text())["status"]=="PASS"
                    pc=json.loads(w.checked(prior/"config.resolved.json").read_text())
                    assert pc.get("source_cache_run",prior.relative_to(ROOT).as_posix())==cfg.get("source_cache_run",fc.get("source_cache_identity"))
                    assert all(pc[key]==cfg[key] for key in ["model_revision","training_run","checkpoint_step","intervention_span"])
                    pattern=fc.get("file_patterns",{}).get(name,"counterfactual_seed{seed}_k{members}_"+name+".npz")
                    maskfile=w.checked(prior/pattern.format(seed=seed,members=k))
                    if fc.get("mask_sha256"):assert sha256(maskfile)==fc["mask_sha256"][maskfile.relative_to(ROOT).as_posix()]
                    with np.load(maskfile) as data:
                        for u in fc["updates"]:
                            array_key=fc.get("array_keys",{}).get(name,{}).get(str(u),f"updates_{u}")
                            loaded[seed,f"adapt_{name}_u{u}",k]=torch.tensor(data[array_key],device=w.device)
            gates=loaded
            metadata.append(dict(frozen_adaptation=fc,current_fit_updates=0,current_output_gradients=0,
                                 original_source_and_target_fit_labels_shared=True))
        write(w.run/"SOURCE_FREEZE.json",dict(written_at_utc=datetime.now(timezone.utc).isoformat(),metadata=metadata,target_dictionaries_used=bool(cfg.get("relation_transfer") or cfg.get("adaptation") or cfg.get("frozen_adaptation")),task_output_gradients="fit split hybrid CE" if cfg.get("counterfactual_fit") else "inherited source and target fit only" if cfg.get("frozen_adaptation") else "inherited source fit only" if source_parent else 0,source_functional_outcomes_used_for_selection=bool(cfg.get("counterfactual_fit") or source_parent),development_functional_outcomes_used_for_selection=False,base_outputs_already_observed=True,files=[dict(path=p.name,sha256=sha256(p)) for p in w.run.glob("source_seed*.npz")]))
        def evaluate(seed,method,k,operation,deltas,gate=None,raw_trajectory=False):
            for off in range(0,len(pairs),batch):
                pp=pairs[off:off+batch];ix=[p["recipient"] for p in pp];donors=[p["donor"] for p in pp]
                patch=None
                if raw_trajectory:patch=lambda current,step:h[donors,:step+1]-current
                elif gate is not None:
                    ae=saes[seed]
                    patch=lambda current,step:((codes[seed][donors,:step+1]-ae.encode(current.reshape(-1,current.shape[-1])).reshape(*current.shape[:-1],-1))*gate)@ae.decoder.weight.T
                ans,txt,hh,norm=generate(ix,deltas[off:off+len(pp)] if deltas is not None else None,patch)
                replay=float(((hh[:,0]-h[ix,0]) if trajectory else (hh-h[ix])).abs().max());assert replay<cfg["hidden_atol"],replay
                for j,p in enumerate(pp):
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
        ii=[p["recipient"] for p in pairs];jj=[p["donor"] for p in pairs]
        evaluate(0,"no_edit",0,"unit",None)
        if trajectory:
            evaluate(0,"raw_prompt_patch",model.config.hidden_size,"unit",h[jj,0]-h[ii,0])
            evaluate(0,"raw_generated_prefix_patch",model.config.hidden_size,"unit",None,raw_trajectory=True)
        else:evaluate(0,"raw_full_patch",model.config.hidden_size,"unit",h[jj]-h[ii])
        for (seed,rule,k),gate in gates.items():
            z=codes[seed];D=saes[seed].decoder.weight.T
            for c,operation in enumerate(["unit","tens"]):
                if trajectory:evaluate(seed,rule,k,operation,None,gate=gate[:,c])
                else:
                    delta=((z[jj]-z[ii])*gate[:,c])@D
                    evaluate(seed,rule,k,operation,delta)
        w.checks.update(source_selection_only_fit_data=True,donor_changes_both_digits=True,real_generation_no_answer_prefix=True,all_failed_base_cases_retained=True)
        if not cfg.get("relation_transfer") and not cfg.get("adaptation") and not cfg.get("frozen_adaptation"):
            w.checks["no_target_used"]=True
        if not cfg.get("counterfactual_fit") and source_parent is None:
            w.checks["source_selection_only_fit_labels_and_activations"]=True
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),precision="float32 matmul high",hook=cfg.get("hook_override",tc["hook_module_path"]),model=tc["model_id"],forward_accounting="Actual no-cache generation forward calls including padded batch members and lengths")
    except Exception as exc:
        error=repr(exc);(w.run/"traceback.log").write_text(traceback.format_exc())
    return w.finish(error)


if __name__=="__main__":raise SystemExit(main())
