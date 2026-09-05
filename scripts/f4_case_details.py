"""Read-only detail export for preselected rank-one donor-difference replays."""
import json
import numpy as np


def token_class(text):
    """Coarse character class, not POS/semantics; byte fragments stay separate."""
    if not text:return 'empty'
    if '\ufffd' in text:return 'byte_fragment'
    if text.isspace():return 'whitespace'
    if any(c.isalnum() for c in text):return 'word_number'
    return 'punctuation_symbol'


def ordered_class_donor(recipient, positions, donors, coordinates, classes):
    """Original sorted-position pairing and energy/tie rule, with a class constraint."""
    candidates=[];n=len(positions);tested=[]
    for donor in donors:
        dp=donor['intervention_positions'][:n];seq=donor['sequence']
        compatible=bool(n and len(dp)==n and [classes[recipient,p] for p in positions]==[classes[seq,p] for p in dp])
        energy=float(np.sum((coordinates[recipient][positions]-coordinates[seq][dp])**2)) if len(dp)==n else None
        tested.append(dict(sequence=seq,positions=dp,compatible=compatible,source_difference_energy=energy))
        if compatible:candidates.append((energy,-seq,dp,donor))
    return (max(candidates,key=lambda x:x[:2]) if candidates else None),tested


def select_cases(selections, payload, *, tokenizer=None, tokens=None, selected_only=False):
    choices={(r['source_seed'],r['source_atom'],r['condition']):r for r in payload['choices']}
    if len(choices)!=len(payload['choices']): raise ValueError('Duplicate case choice')
    result=[];seen=set()
    for unit in selections:
        chosen=[]
        for condition in ('positive','negative'):
            key=(unit['source_seed'],unit['source_atom'],condition);seen.add(key)
            choice=choices[key];entry=choice['entry'];override=payload.get('donor_override',False)
            original=choice['original_entry'] if override else entry
            if original is not None and sum(e==original for e in unit['sequences'])!=1:
                raise ValueError('Frozen original recipient no longer matches')
            if entry is None: continue
            if not choice['source_scope']['supported']:
                raise ValueError('Frozen case entry no longer matches source-only selection')
            if override:
                allowed={'donor_sequence','donor_positions','donor_document_ids','donor_source_difference_energy','donor_status'}
                if original is None or any(entry.get(k)!=original.get(k) for k in (set(entry)|set(original))-allowed):
                    raise ValueError('Donor override changed recipient fields')
                donors=[e for e in unit['sequences'] if e['sequence']==entry['donor_sequence'] and e['condition']!=condition]
                pos=entry['intervention_positions'];dp=entry['donor_positions']
                if len(donors)!=1 or dp!=donors[0]['intervention_positions'][:len(pos)] or len(dp)!=len(pos) or len(set(dp))!=len(dp):
                    raise ValueError('Donor override left original ordered position pool')
                if entry['donor_document_ids']!=donors[0]['document_ids'] or set(entry['document_ids'])&set(entry['donor_document_ids']):
                    raise ValueError('Donor document identity mismatch')
                if tokenizer is None or tokens is None:raise ValueError('Donor override needs token-class verification')
                if any(token_class(tokenizer.decode([int(tokens[entry['sequence'],p])]))!=token_class(tokenizer.decode([int(tokens[entry['donor_sequence'],d])])) for p,d in zip(pos,dp)):
                    raise ValueError('Frozen donor token classes no longer match')
                if selected_only and not choice['source_scope']['selected']:continue
                if not selected_only and payload.get('evaluate_changed_only') and choice['matching_status']=='UNCHANGED_REUSABLE':
                    if any(entry[k]!=original[k] for k in ('donor_sequence','donor_positions','donor_document_ids')):
                        raise ValueError('Unchanged case has changed donor')
                    continue
            elif sum(e==entry for e in unit['sequences'])!=1:
                raise ValueError('Frozen case entry no longer matches source-only selection')
            chosen.append(entry)
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
