import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from f4_case_details import atom_terms, select_cases, context, effect_tokens


class Tokenizer:
    def decode(self, ids): return ','.join(str(i) for i in ids)


class CaseDetailsTests(unittest.TestCase):
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
