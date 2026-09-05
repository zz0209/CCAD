import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_f4_agreement_source import make_prompts,swap_indices,margins,capped


class AgreementTests(unittest.TestCase):
    def test_factorial_and_swaps(self):
        cfg=json.loads((ROOT/'configs/f4_agreement_source_v1.json').read_text());rows=make_prompts(cfg['design'])
        self.assertEqual(len(rows),128);self.assertEqual(len({r['text'] for r in rows}),128)
        for split in ('development','reserved'):
            group=[r for r in rows if r['split']==split]
            for axis in ('subject','attractor'):
                ix=swap_indices(group,axis);np.testing.assert_array_equal(ix[ix],np.arange(64))
                for i,j in enumerate(ix):
                    self.assertEqual(group[i]['template'],group[j]['template'])
                    self.assertNotEqual(group[i][axis+'_number'],group[j][axis+'_number'])
                    other='attractor' if axis=='subject' else 'subject'
                    self.assertEqual(group[i][other+'_number'],group[j][other+'_number'])

    def test_sign_and_cap(self):
        out=margins(np.array([[-1.,-3.,-4.,-5.],[-1.,-3.,-4.,-5.]]),[{'subject_number':0},{'subject_number':1}])
        np.testing.assert_allclose(out[:,:2],[[2.,1.],[-2.,-1.]])
        delta,scale=capped(np.array([[3.,4.],[0.,0.]]),np.array([[6.,8.],[1.,0.]]),.1)
        np.testing.assert_allclose(scale,[.2,1.]);np.testing.assert_allclose(delta,[[.6,.8],[0.,0.]])


if __name__=='__main__':unittest.main()
