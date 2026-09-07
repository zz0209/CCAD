"""Authored number/time factorial material; no model-dependent item selection."""
from __future__ import annotations


def make_panel(cfg):
    rows=[];lookup={};nouns=cfg['noun_pairs']
    for block,noun in enumerate(nouns):
        partition=cfg.get('lexical_partition_size',len(nouns));assert len(nouns)%partition==0
        other=(block//partition)*partition+(block+1)%partition
        for cue_id,cues in enumerate(cfg['time_cues']):
            for template in cfg['templates']:
                for number in [0,1]:
                    for past in [0,1]:
                        for distractor in [0,1]:
                            middle={'pp':' near the','subject_relative':' that praised the'}[template]
                            spans=['<|endoftext|>'+cues[past],', the',' '+noun[number],middle,' '+nouns[other][distractor]]
                            key=(block,cue_id,template,number,past,distractor);i=len(rows);lookup[key]=i
                            rows.append(dict(id=i,block=block,cue_id=cue_id,template=template,number=number,past=past,distractor=distractor,spans=spans,text=''.join(spans),positions_by_region={'time':0,'number':2,'distractor':4},expected_label_index=2*past+number))
    pairs=[]
    for r in rows:
        common=(r['block'],r['cue_id'],r['template']);n,t,d=r['number'],r['past'],r['distractor']
        pairs.append(dict(base=r['id'],number=lookup[common+(1-n,t,d)],time=lookup[common+(n,1-t,d)],joint=lookup[common+(1-n,1-t,d)]))
    return rows,pairs
