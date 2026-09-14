"""Measure exact generated arithmetic answers before investing in SAE material."""
import argparse,json,os,re,time,platform,traceback
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_causalgym_multisite import MultisiteWork,ROOT,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
    w=MultisiteWork(cfg,a.config,['scripts/run_arithmetic_model_screen.py','scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    error=None
    try:
        import numpy as np,torch,transformers
        torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        w.torch=torch;w.device=torch.device(cfg['device'])
        for name in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','LICENSE']:w.checked(Path(cfg['model_local_dir'])/name)
        tok=transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'],local_files_only=True,trust_remote_code=False,padding_side='left')
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        model=transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().to(w.device);model.requires_grad_(False)
        lo,hi=cfg.get('operand_range',[0,cfg.get('operand_limit',10)])
        pairs=[(x,y) for x in range(lo,hi) for y in range(lo,hi) if [x,y] not in cfg.get('excluded_pairs',[])]
        if cfg.get('sample_pairs'):
            rng=np.random.default_rng(cfg['data_seed']);pairs=[pairs[i] for i in rng.choice(len(pairs),size=cfg['sample_pairs'],replace=False)]
        rows=[dict(template=t,a=x,b=y,total=x+y,prompt=template.format(a=x,b=y)) for t,template in enumerate(cfg['templates']) for x,y in pairs]
        write(w.run/'panel.json',dict(rows=rows));records=[]
        for off in range(0,len(rows),cfg['batch_size']):
            rr=rows[off:off+cfg['batch_size']];inputs=tok([r['prompt'] for r in rr],padding=True,return_tensors='pt').to(w.device)
            with torch.no_grad():out=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            generated=tok.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            for r,s in zip(rr,generated):
                match=re.match(r'\s*(\d+)',s);answer=int(match[1]) if match else None
                w.record(kind='arithmetic_capability',task='template_'+str(r['template']),row_id=r['a']*hi+r['b'],component=f"{r['a']}+{r['b']}",method='greedy_generation',prompt=r['prompt'],generated_text=s,answer=answer,correct_answer=r['total'],correct=answer==r['total'])
                records.append(dict(**r,generated=s,answer=answer,correct=answer==r['total']))
            w.sequence_forwards+=len(rr);w.token_forwards+=int(inputs['attention_mask'].sum())+int(out.shape[1]-inputs['input_ids'].shape[1])*len(rr)
            w.progress('CAPABILITY_BATCH',completed=len(records),total=len(rows))
            if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Screen budget')
        quality=[dict(template=t,n=sum(r['template']==t for r in records),exact_answer_accuracy=float(np.mean([r['correct'] for r in records if r['template']==t])),parsed_fraction=float(np.mean([r['answer'] is not None for r in records if r['template']==t]))) for t in range(len(cfg['templates']))]
        write(w.run/'capability_results.json',dict(quality=quality,scope=cfg['scope'],criterion='First generated integer after optional whitespace must equal the exact sum; any nonnumeric prefix or wrong integer is incorrect.'))
        w.environment=dict(python=os.sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),model=cfg['model_id'],generation_forward_accounting='sequence_forwards counts generation requests; token_forwards counts prompt and generated tokens, not repeated transformer FLOPs')
        w.checks['all_configured_prompts_evaluated']=len(records)==len(rows)
    except Exception as e:error=repr(e);(w.run/'stderr.log').write_text(traceback.format_exc())
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
