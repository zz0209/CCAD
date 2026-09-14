"""Discover and causally test digit-specific source SAE components.

A donor differs in both answer digits. A successful unit/tens intervention
copies just that digit and preserves the other. All edits start at the same
prompt-final hidden state; no correct answer prefix is supplied to generation.
Feature selection uses source activations and arithmetic labels only.
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
    for t,template in enumerate(cfg["templates"]):
        for a in range(*cfg["operand_range"]):
            for b in range(a,cfg["operand_range"][1]):
                key=f"{cfg['split_salt']}:{a}:{b}"
                split="fit" if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2==0 else "development"
                if t and split=="fit":continue
                total=a+b
                assert 10<=total<100
                rows.append(dict(template=t,a=a,b=b,total=total,unit=total%10,tens=total//10,
                                 carry=(a%10+b%10)>=10,split=split,prompt=template.format(a=a,b=b)))
    pairs=[]
    for t in range(len(cfg["templates"])):
        ii=[i for i,r in enumerate(rows) if r["template"]==t and r["split"]=="development"]
        ii=sorted(ii,key=lambda i:hashlib.sha256(f"pair:{cfg['split_salt']}:{rows[i]['a']}:{rows[i]['b']}".encode()).hexdigest())
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
    cfg=json.loads(args.config.read_text());w=MultisiteWork(cfg,args.config,["scripts/run_arithmetic_digit_components.py","scripts/run_causalgym_multisite.py","scripts/run_r011s1_raw_hook_asset.py","src/ccad/artifacts.py"])
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
        model.requires_grad_(False);module=model.get_submodule(tc["hook_module_path"])
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
        tokenrows=[tok.encode(r["prompt"],add_special_tokens=False) for r in rows]
        batch=cfg["batch_size"];padding=cfg["max_length"]
        physical_length=max(map(len,tokenrows))+cfg["max_new_tokens"]
        assert physical_length<=padding
        def budget():
            if time.perf_counter()-w.wall_start>cfg["budget_seconds"]:raise TimeoutError("Arithmetic source development budget")
        def generate(indices,delta=None):
            padded=indices+[indices[0]]*(batch-len(indices));ids=torch.full((batch,padding),tok.eos_token_id,device=w.device,dtype=torch.long)
            mask=torch.zeros_like(ids);pos=[]
            for j,i in enumerate(padded):
                tokens=tokenrows[i];ids[j,:len(tokens)]=torch.tensor(tokens,device=w.device);mask[j,:len(tokens)]=1;pos.append(len(tokens)-1)
            pos=torch.tensor(pos,device=w.device);ix=torch.arange(batch,device=w.device);cache={};generated=[];finished=torch.zeros(batch,device=w.device,dtype=torch.bool)
            def hook(m,a,out):
                h=out[0] if isinstance(out,tuple) else out
                if "hidden" not in cache:cache["hidden"]=h[ix,pos].detach().clone()
                if delta is None:return out
                hh=h.clone();hh[ix[:len(indices)],pos[:len(indices)]]+=delta
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
            return answers,text[:len(indices)],cache["hidden"][:len(indices)]
        all_h=[];base=[];base_text=[]
        for off in range(0,len(rows),batch):
            indices=list(range(off,min(off+batch,len(rows))));ans,txt,h=generate(indices);all_h.append(h);base.extend(ans);base_text.extend(txt)
            if off%(batch*16)==0:w.progress("SOURCE_STATES",completed=off+len(indices),total=len(rows))
            budget()
        h=torch.cat(all_h);fitix=[i for i,r in enumerate(rows) if r["split"]=="fit"]
        for i,r in enumerate(rows):w.record(kind="base",task="template_"+str(r["template"]),row_id=i,component=f"{r['a']}+{r['b']}",method="greedy",split=r["split"],answer=base[i],correct_answer=r["total"],correct=base[i]==r["total"],generated_text=base_text[i])
        codes={};gates={};metadata=[]
        for seed,ae in saes.items():
            with torch.no_grad():z=torch.cat([ae.encode(h[i:i+256]) for i in range(0,len(h),256)])
            codes[seed]=z;fs=torch.stack([fisher(z[fitix],[rows[i][factor] for i in fitix]) for factor in ["unit","tens"]],1)
            for rule in cfg["selection_rules"]:
                score=fs if rule=="fisher" else fs-fs.flip(1)
                for k in cfg["members"]:
                    gate=torch.zeros((z.shape[1],2),device=w.device)
                    for c in range(2):gate[torch.argsort(score[:,c],descending=True,stable=True)[:k],c]=1
                    gates[seed,rule,k]=gate
                    metadata.append(dict(seed=seed,rule=rule,members=k,unit_members=torch.where(gate[:,0]>0)[0].cpu().tolist(),tens_members=torch.where(gate[:,1]>0)[0].cpu().tolist(),overlap=int((gate[:,0]*gate[:,1]).sum())))
            payload={f"{rule}_{k}":gate.cpu().numpy() for (s,rule,k),gate in gates.items() if s==seed}
            np.savez_compressed(w.run/f"source_seed{seed}.npz",codes=z.cpu().numpy(),fisher=fs.cpu().numpy(),**payload)
        np.savez_compressed(w.run/"states.npz",hidden=h.cpu().numpy(),fit_indices=np.array(fitix))
        write(w.run/"SOURCE_FREEZE.json",dict(written_at_utc=datetime.now(timezone.utc).isoformat(),metadata=metadata,target_dictionaries_used=False,task_output_gradients=0,source_functional_outcomes_used_for_selection=False,base_outputs_already_observed=True,files=[dict(path=p.name,sha256=sha256(p)) for p in w.run.glob("source_seed*.npz")]))
        def evaluate(seed,method,k,operation,deltas):
            for off in range(0,len(pairs),batch):
                pp=pairs[off:off+batch];ix=[p["recipient"] for p in pp];ans,txt,hh=generate(ix,deltas[off:off+len(pp)] if deltas is not None else None)
                replay=float((hh-h[ix]).abs().max());assert replay<cfg["hidden_atol"],replay
                for j,p in enumerate(pp):
                    expected=p[operation+"_answer"] if operation in ["unit","tens"] else p["donor_answer"]
                    a=ans[j];numeric=a is not None and 10<=a<100
                    target=(a%10==p["donor_answer"]%10 if operation=="unit" else a//10==p["donor_answer"]//10) if numeric else False
                    preserve=(a//10==p["base_answer"]//10 if operation=="unit" else a%10==p["base_answer"]%10) if numeric else False
                    w.record(kind="source_patch",task="template_"+str(p["template"]),row_id=off+j,component=str(p["recipient"])+"<-"+str(p["donor"]),seed=seed,method=method,mode="k"+str(k),operation=operation,
                             answer=a,expected_answer=expected,exact_hybrid=a==expected,target_digit_success=target,preserve_digit_success=preserve,
                             base_answer=p["base_answer"],donor_answer=p["donor_answer"],base_correct=base[p["recipient"]]==p["base_answer"],donor_correct=base[p["donor"]]==p["donor_answer"],
                             recipient_carry=p["recipient_carry"],donor_carry=p["donor_carry"],generated_text=txt[j],hidden_replay_error=replay,edit_norm=float(deltas[off+j].norm()) if deltas is not None else 0.)
                budget()
            w.progress("SOURCE_PATCH",seed=seed,method=method,members=k,operation=operation)
        ii=[p["recipient"] for p in pairs];jj=[p["donor"] for p in pairs]
        evaluate(0,"no_edit",0,"unit",None);evaluate(0,"raw_full_patch",model.config.hidden_size,"unit",h[jj]-h[ii])
        for (seed,rule,k),gate in gates.items():
            z=codes[seed];D=saes[seed].decoder.weight.T
            for c,operation in enumerate(["unit","tens"]):
                delta=((z[jj]-z[ii])*gate[:,c])@D
                evaluate(seed,rule,k,operation,delta)
        w.checks.update(source_selection_only_fit_labels_and_activations=True,no_target_used=True,donor_changes_both_digits=True,real_generation_no_answer_prefix=True,all_failed_base_cases_retained=True)
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),precision="float32 matmul high",hook=tc["hook_module_path"],model=tc["model_id"],forward_accounting="Actual no-cache generation forward calls including padded batch members and lengths")
    except Exception as exc:
        error=repr(exc);(w.run/"traceback.log").write_text(traceback.format_exc())
    return w.finish(error)


if __name__=="__main__":raise SystemExit(main())
