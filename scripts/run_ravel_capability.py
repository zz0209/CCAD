"""Measure unedited model knowledge on pinned RAVEL training entities/templates.

This producer does not select by any SAE or intervention outcome. Its first-token
metric is explicit; it is not the complete RAVEL generation benchmark.
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
import traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from run_causalgym_multisite import MultisiteWork, ROOT, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    w=MultisiteWork(cfg,args.config,source_files=[
        'scripts/run_ravel_capability.py','scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    manifest=json.loads((w.run/'manifest.json').read_text())
    manifest.update(schema_version='ravel.capability.v1',run_parent='FINAL_FIVE_R16',
        milestone='semantic-attribute-capability',statistics_unit='RAVEL entity and original prompt template',
        threshold_source_split='Fixed structural/template choices; unedited train-only knowledge screening')
    write(w.run/'manifest.json',manifest)
    error=None
    try:
        import torch
        import transformers
        w.torch=torch;torch.set_num_threads(cfg['cpu_threads'])
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        w.device=torch.device(cfg['device']);torch.cuda.set_device(w.device);torch.cuda.reset_peak_memory_stats()
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
            torch=torch.__version__,transformers=transformers.__version__,device=str(w.device),
            matmul_precision=torch.get_float32_matmul_precision(),gpu=torch.cuda.get_device_name(),cpu_threads=torch.get_num_threads())
        data=Path(cfg['dataset_dir']);loaded={}
        for name in ['ravel_city_entity_attributes.json','ravel_city_entity_to_split.json',
                     'ravel_city_attribute_to_prompts.json','ravel_city_prompt_to_split.json']:
            loaded[name]=json.loads(w.checked(data/name,'Official RAVEL '+cfg['dataset_revision'],'MIT repository').read_text())
        w.checked(data.parent/'LICENSE','RAVEL repository license','MIT')
        w.checked(ROOT/'.aris/compute/local-r006b1-env-spec.json')
        attributes=loaded['ravel_city_entity_attributes.json'];es=loaded['ravel_city_entity_to_split.json']
        prompts=loaded['ravel_city_attribute_to_prompts.json'];ps=loaded['ravel_city_prompt_to_split.json']
        modeldir=Path(cfg['model_local_dir'])
        for name in ['config.json','tokenizer.json','model.safetensors']:
            w.checked(modeldir/name,'Pinned EleutherAI Pythia1B '+cfg['model_revision'],'Apache-2.0')
        tok=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
        tok.pad_token=tok.eos_token
        model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.config.use_cache=False
        for p in model.parameters():p.requires_grad_(False)
        panel=[];excluded=[]
        entities=[e for e in attributes if es[e]=='train']
        if cfg.get('maximum_entities'):entities=entities[:cfg['maximum_entities']]
        for entity in entities:
            for attr,indices in cfg['templates'].items():
                for ti in indices:
                    template=prompts[attr][ti]
                    if ps[template]!='train':raise ValueError('Only official training templates in capability producer')
                    prefix,suffix=template.split('%s')
                    if entity in prefix or entity in suffix:
                        excluded.append(dict(entity=entity,attribute=attr,template_index=ti,reason='Entity already in demonstration'));continue
                    text=template%entity;input_ids=tok.encode(text,add_special_tokens=False)
                    continuation_prefix='' if text.endswith('"') else ' '
                    values=[v.strip() for v in attributes[entity][attr].split(',') if v.strip()]
                    labels=[continuation_prefix+v for v in values]
                    full_ids=[tok.encode(text+label,add_special_tokens=False) for label in labels]
                    if any(x[:len(input_ids)]!=input_ids or len(x)<=len(input_ids) for x in full_ids):
                        excluded.append(dict(entity=entity,attribute=attr,template_index=ti,reason='Non-concatenative answer boundary'));continue
                    answer_ids=[x[len(input_ids):] for x in full_ids]
                    # Entity-end locations remain available for later source capture.
                    entity_tokens=tok.encode(prefix+entity,add_special_tokens=False)
                    if input_ids[:len(entity_tokens)]!=entity_tokens:
                        excluded.append(dict(entity=entity,attribute=attr,template_index=ti,reason='Retokenized entity-end boundary'));continue
                    panel.append(dict(row_id=len(panel),entity=entity,attribute=attr,template_index=ti,template=template,
                        text=text,tokens=input_ids,entity_end=len(entity_tokens)-1,answer_values=values,
                        answer_token_ids=answer_ids,original_entity_split='train',original_template_split='train'))
        write(w.run/'panel.json',dict(rows=panel,exclusions=excluded,
            scope='Raw city dictionaries include all official splits; only original train entities and train templates are selected, model-evaluated or used for screening. No test-model outcomes.'))
        w.progress('CAPABILITY_PANEL',rows=len(panel),entities=len(entities),structural_exclusions=len(excluded))
        entity_scores=defaultdict(lambda:defaultdict(list))
        for ids in w.batches(np.arange(len(panel))):
            enc=tok([panel[i]['text'] for i in ids],add_special_tokens=False,padding=True,return_tensors='pt').to(w.device)
            with torch.no_grad():
                hidden=model.gpt_neox(**enc,use_cache=False).last_hidden_state
                logits=model.get_output_embeddings()(hidden[torch.arange(len(ids),device=w.device),enc.attention_mask.sum(1)-1])
                lp=torch.log_softmax(logits.double(),dim=-1).cpu().numpy()
            w.sequence_forwards+=len(ids);w.token_forwards+=int(enc.attention_mask.sum())
            for i,values in zip(ids,lp):
                row=panel[i];answers=sorted({a[0] for a in row['answer_token_ids']})
                top=np.argsort(-values)[:10];correct=int(int(top[0]) in answers)
                record=dict(kind='capability',task=row['attribute'],row_id=int(i),mode='template_'+str(row['template_index']),
                    method='unedited_model',seed=0,target_seed=None,operation='knowledge',split='official_train_development',
                    component=row['entity'],entity=row['entity'],template_index=row['template_index'],
                    base_label_correct=correct,first_token_correct=correct,expected_first_token_ids=answers,
                    expected_probability=float(np.exp(values[answers]).sum()),
                    answer_ranks=[int((values>values[a]).sum()+1) for a in answers],
                    top_ids=top.tolist(),top_text=[tok.decode([int(t)]) for t in top],top_logprobs=values[top].tolist(),
                    full_answer_token_lengths=[len(x) for x in row['answer_token_ids']])
                w.record(**record);entity_scores[row['entity']][row['attribute']].append(correct)
            if w.sequence_forwards%(cfg['batch_size']*20)==0:w.progress('CAPABILITY_FORWARD',completed=w.sequence_forwards,total=len(panel))
        scores={e:{a:dict(correct=sum(v),total=len(v),fraction=float(np.mean(v))) for a,v in aa.items()} for e,aa in entity_scores.items()}
        write(w.run/'entity_knowledge.json',dict(entities=scores,criterion=cfg['screening_rule'],
            scope='First-token screening only; all correctness/failures retained. No SAE/source/target operation outcome is used.'))
        w.checks['official_train_entities_and_templates_only']=all(r['original_entity_split']==r['original_template_split']=='train' for r in panel)
        w.checks['every_selected_prompt_evaluated']=w.sequence_forwards==len(panel)
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
