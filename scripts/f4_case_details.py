"""Read-only detail export for preselected rank-one donor-difference replays."""
import json
import numpy as np


def select_cases(selections, payload):
    choices={(r['source_seed'],r['source_atom'],r['condition']):r for r in payload['choices']}
    if len(choices)!=len(payload['choices']): raise ValueError('Duplicate case choice')
    result=[];seen=set()
    for unit in selections:
        chosen=[]
        for condition in ('positive','negative'):
            key=(unit['source_seed'],unit['source_atom'],condition);seen.add(key)
            choice=choices[key];entry=choice['entry']
            if entry is None: continue
            matches=[e for e in unit['sequences'] if e==entry]
            if len(matches)!=1 or not choice['source_scope']['supported']:
                raise ValueError('Frozen case entry no longer matches source-only selection')
            chosen.append(matches[0])
        result.append(dict(unit,sequences=chosen))
    if seen!=set(choices): raise ValueError('Case query panel mismatch')
    return result


def atom_terms(ids, difference, beta, scale):
    """All nonzero signed scalar terms; sum reconstructs the rank-one coordinate."""
    ids=np.asarray(ids,dtype=int);difference=np.asarray(difference);beta=np.asarray(beta)
    terms=difference*beta*scale
    active=np.flatnonzero(terms!=0)
    order=active[np.lexsort((ids[active],-np.abs(terms[active])))]
    rows=[dict(atom=int(ids[i]),code_difference=float(difference[i]),coefficient=float(beta[i]),signed_contribution=float(terms[i])) for i in order]
    return dict(total=float(terms.sum()),positive_sum=float(terms[terms>0].sum()),negative_sum=float(terms[terms<0].sum()),
                absolute_sum=float(np.abs(terms).sum()),nonzero_atoms=len(rows),terms=rows)


def context(tokenizer, tokens, position):
    values=[int(v) for v in tokens];bound=max([i for i in range(position) if values[i]==0],default=-1)
    start=max(bound+1,position-12)
    return dict(position=int(position),token_id=values[position],token=tokenizer.decode([values[position]]),
                prefix=tokenizer.decode(values[start:position]),next_observed_token=tokenizer.decode(values[position+1:position+2]),
                prefix_start=start,contains_future_context=False)


def effect_tokens(tokenizer, baseline_logits, candidate_logits, source_logits, pos, limit=6):
    # Actual intervention-induced change = intervened minus baseline, opposite to removal-effect convention.
    source=source_logits[0,pos]-baseline_logits[0,pos];source=source-source.mean()
    candidate=candidate_logits[0,pos]-baseline_logits[0,pos];candidate=candidate-candidate.mean()
    ids=np.arange(len(source));positive=np.lexsort((ids,-source))[:limit];negative=np.lexsort((ids,source))[:limit]
    def rows(order):
        return [dict(token_id=int(i),token=tokenizer.decode([int(i)]),source_delta=float(source[i]),candidate_delta=float(candidate[i])) for i in order]
    return dict(source_top_increased=rows(positive),source_top_decreased=rows(negative),
                candidate_top_increased=rows(np.lexsort((ids,-candidate))[:limit]),
                candidate_top_decreased=rows(np.lexsort((ids,candidate))[:limit]))


def export_case(path, *, tokenizer, tokens, entry, s, a, t, method, z, ids, dec, b, mapped_beta,
                scale, source, delta, baseline, ref, out, endpoints):
    base_logits=baseline['next_logits'].cpu().numpy();src_logits=ref['next_logits'].cpu().numpy();out_logits=out['next_logits'].cpu().numpy()
    sbeta=(dec[s][ids]@b)[:,0]
    positions=[]
    for pos,dp in zip(entry['intervention_positions'],entry['donor_positions']):
        src=atom_terms(ids,z[s][pos,ids],sbeta,scale)
        target=atom_terms(np.arange(z[t].shape[1]),z[t][pos],mapped_beta,scale) if mapped_beta is not None else None
        source_residual=float(np.linalg.norm(src['total']*b[:,0]-source[pos]))
        target_residual=float(np.linalg.norm(target['total']*b[:,0]-delta[pos])) if target is not None else None
        if source_residual>1e-8 or (target_residual is not None and target_residual>1e-8):
            raise ValueError('Signed atom export does not reconstruct intervention')
        positions.append(dict(recipient=context(tokenizer,tokens[entry['sequence']],pos),donor=context(tokenizer,tokens[entry['donor_sequence']],dp),
                              source_atoms=src,target_atoms=target,source_reconstruction_residual=source_residual,
                              target_reconstruction_residual=target_residual,
                              logits=effect_tokens(tokenizer,base_logits,out_logits,src_logits,pos)))
    record=dict(source_seed=s,source_atom=a,target_seed=t,method=method,sequence=entry['sequence'],condition=entry['condition'],
                donor_sequence=entry['donor_sequence'],document_ids=entry['document_ids'],donor_document_ids=entry['donor_document_ids'],
                common_source_dose_scale=scale,endpoints=endpoints,positions=positions,
                sign_convention='Centered next-token logits: intervened minus baseline. Signed atom terms are subtracted from recipient hook.',
                atom_scope='Source candidate contribution and source-aligned target readout, not native deletion. Raw has no target atom decomposition.')
    with path.open('a',encoding='utf-8') as stream: stream.write(json.dumps(record,sort_keys=True)+'\n')
