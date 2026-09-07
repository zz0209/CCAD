"""Authored number/time factorial material; no model-dependent item selection."""
from __future__ import annotations


def make_panel(cfg):
    if 'cue_roles' in cfg:
        rows=[];pairs=[]
        for role in cfg['cue_roles']:
            part_cfg={key:value for key,value in cfg.items() if key!='cue_roles'}
            part_cfg['cue_role']=role
            part_rows,part_pairs=make_panel(part_cfg);offset=len(rows)
            for row in part_rows:
                rows.append(dict(row,id=row['id']+offset,cue_role=role))
            pairs.extend({key:value+offset for key,value in pair.items()} for pair in part_pairs)
        return rows,pairs
    rows=[];lookup={};nouns=cfg['noun_pairs']
    for block,noun in enumerate(nouns):
        partition=cfg.get('lexical_partition_size',len(nouns));assert len(nouns)%partition==0
        other=(block//partition)*partition+(block+1)%partition
        for cue_id,cues in enumerate(cfg['time_cues']):
            for template in cfg['templates']:
                for number in [0,1]:
                    for past in [0,1]:
                        for distractor in [0,1]:
                            cue=cues[past]; semantic_past=0 if cfg.get('cue_role')=='quoted' else past
                            prefix='<|endoftext|>'+cue; subject=' '+noun[number]; distract=' '+nouns[other][distractor]
                            regions={'time':0,'number':2,'distractor':4}
                            if template in ['pp','subject_relative']:
                                middle={'pp':' near the','subject_relative':' that praised the'}[template]
                                spans=[prefix,', the',subject,middle,distract]
                            elif template=='object_relative':
                                spans=[prefix,', the',subject,' that the',distract,' praised']
                            elif template=='subject_late':
                                spans=[prefix,', near the',distract,', the',subject]; regions={'time':0,'number':4,'distractor':2}
                            else: raise ValueError('Unknown template '+template)
                            if cfg.get('cue_role')=='quoted':
                                # The time slot remains at the CHANGED quoted
                                # cue's end, before the fixed present-time cue.
                                # Capturing the latter would confound cue site.
                                spans=['<|endoftext|>The title is "',cue,'". Right now']+spans[1:]
                                regions={name:(1 if name=='time' else index+2) for name,index in regions.items()}
                            regions['final']=len(spans)-1
                            key=(block,cue_id,template,number,past,distractor);i=len(rows);lookup[key]=i
                            row=dict(id=i,block=block,cue_id=cue_id,template=template,number=number,past=past,distractor=distractor,spans=spans,text=''.join(spans),positions_by_region=regions,expected_label_index=2*semantic_past+number)
                            rows.append(row)
    pairs=[]
    for r in rows:
        common=(r['block'],r['cue_id'],r['template']);n,t,d=r['number'],r['past'],r['distractor']
        pairs.append(dict(base=r['id'],number=lookup[common+(1-n,t,d)],time=lookup[common+(n,1-t,d)],joint=lookup[common+(1-n,1-t,d)]))
    return rows,pairs
