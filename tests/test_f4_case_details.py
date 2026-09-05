import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from f4_case_details import atom_terms, select_cases, context, effect_tokens, token_class, ordered_class_donor, freeze_source_cases


class Tokenizer:
    def decode(self, ids): return ','.join(str(i) for i in ids)


class CaseDetailsTests(unittest.TestCase):
    def test_source_case_preference_and_missing(self):
        entries=[dict(condition='positive',sequence=s) for s in [8,2,4]]
        unit=dict(source_seed=1,source_atom=2,sequences=entries)
        scope=[dict(source_seed=1,source_atom=2,condition='positive',sequence=s,supported=True,selected=s!=8) for s in [8,2,4]]
        result=freeze_source_cases([unit],scope,'test')
        self.assertEqual(result['choices'][0]['entry']['sequence'],2)
        self.assertIsNone(result['choices'][1]['entry']);self.assertFalse(result['prior_endpoint_exposure'])
        for r in scope:r['selected']=False
        self.assertEqual(freeze_source_cases([unit],scope,'test')['choices'][0]['entry']['sequence'],8)

    def test_token_classes_not_semantic_labels(self):
        self.assertEqual([token_class(v) for v in (' is',"'s",' 2','.', '\n', '\ufffd','')],
                         ['word_number','word_number','word_number','punctuation_symbol','whitespace','byte_fragment','empty'])

    def test_matching_filters_donor_not_position_order(self):
        donors=[dict(sequence=1,intervention_positions=[0,1]),dict(sequence=2,intervention_positions=[0,1])]
        coords={0:np.array([[1.],[2.]]),1:np.array([[10.],[20.]]),2:np.zeros((2,1))}
        classes={(s,p):'word_number' for s in coords for p in (0,1)};classes[1,1]='punctuation_symbol'
        result,records=ordered_class_donor(0,[0,1],donors,coords,classes)
        self.assertEqual((result[3]['sequence'],result[2]),(2,[0,1]));self.assertFalse(records[0]['compatible'])
        classes[2,0]='whitespace';self.assertIsNone(ordered_class_donor(0,[0,1],donors,coords,classes)[0])

    def test_override_verifies_recipient_pool_and_classes(self):
        class ClassTokenizer:
            def decode(self, ids):return {1:'word',2:'.',3:'other'}[ids[0]]
        original=dict(condition='positive',sequence=0,document_ids=['a'],intervention_positions=[0],donor_sequence=1,donor_positions=[0],donor_document_ids=['b'],donor_source_difference_energy=1.,donor_status='SELECTED_SOURCE_ONLY')
        donors=[dict(condition='negative',sequence=s,document_ids=[d],intervention_positions=[0]) for s,d in [(1,'b'),(2,'c')]]
        entry=dict(original,donor_sequence=2,donor_document_ids=['c'])
        choice=dict(source_seed=1,source_atom=2,condition='positive',original_entry=original,entry=entry,source_scope={'supported':True},matching_status='CHANGED_SUPPORTED')
        payload=dict(donor_override=True,evaluate_changed_only=True,choices=[choice,dict(source_seed=1,source_atom=2,condition='negative',original_entry=None,entry=None)])
        unit=dict(source_seed=1,source_atom=2,sequences=[original]+donors);tokens=np.array([[1],[2],[3]])
        self.assertEqual(len(select_cases([unit],payload,tokenizer=ClassTokenizer(),tokens=tokens)[0]['sequences']),1)
        choice['source_scope']['selected']=False
        self.assertEqual(len(select_cases([unit],payload,tokenizer=ClassTokenizer(),tokens=tokens,selected_only=True)[0]['sequences']),0)
        choice['source_scope']['selected']=True
        self.assertEqual(len(select_cases([unit],payload,tokenizer=ClassTokenizer(),tokens=tokens,selected_only=True)[0]['sequences']),1)
        tokens[2,0]=2
        with self.assertRaises(ValueError):select_cases([unit],payload,tokenizer=ClassTokenizer(),tokens=tokens)
        tokens[2,0]=3;entry['sequence']=1
        with self.assertRaises(ValueError):select_cases([unit],payload,tokenizer=ClassTokenizer(),tokens=tokens)

    def test_signed_terms_reconstruct(self):
        r=atom_terms([2,5,8],np.array([1.,2.,0.]),np.array([3.,-2.,9.]),.5)
        self.assertEqual((r['total'],r['positive_sum'],r['negative_sum'],r['nonzero_atoms']),(-.5,1.5,-2.,2))
        self.assertEqual([v['atom'] for v in r['terms']],[5,2])

    def test_selection_keeps_missing_and_rejects_changed_entry(self):
        entry=dict(condition='positive',sequence=3)
        unit=dict(source_seed=1,source_atom=2,sequences=[entry])
        choices=[dict(source_seed=1,source_atom=2,condition=c,entry=entry if c=='positive' else None,source_scope={'supported':True}) for c in ('positive','negative')]
        self.assertEqual(select_cases([unit],{'choices':choices})[0]['sequences'],[entry])
        choices[0]['entry']=dict(entry,sequence=4)
        with self.assertRaises(ValueError):select_cases([unit],{'choices':choices})

    def test_prefix_stops_at_boundary(self):
        r=context(Tokenizer(),[9,0,2,3,4],3)
        self.assertEqual((r['prefix'],r['token'],r['next_observed_token']),('2','3','4'))

    def test_logit_sign_and_shared_token_order(self):
        zero=np.zeros((1,1,3));source=np.array([[[2.,-1.,-1.]]]);candidate=-source
        r=effect_tokens(Tokenizer(),zero,candidate,source,0,1)
        v=r['source_top_increased'][0]
        self.assertEqual((v['token_id'],v['source_delta'],v['candidate_delta']),(0,2.,-2.))


if __name__=='__main__':unittest.main()
