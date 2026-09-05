import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_f4_agreement_source import make_prompts,swap_indices,margins,capped,task_contrast_basis


class AgreementTests(unittest.TestCase):
    def test_task_contrast_balances_nuisance_and_order(self):
        rows=[dict(template=str(t),subject_number=s,attractor_number=a) for t in range(3) for s in (0,1) for a in (0,1)]
        codes=np.array([[r['subject_number'],r['attractor_number']+int(r['template'])] for r in rows],dtype=float)
        decoder=np.array([[3.,4.,0.],[0.,0.,20.]])
        b,y,norm=task_contrast_basis(codes,decoder,rows)
        np.testing.assert_allclose(b,[.6,.8,0.]);self.assertAlmostEqual(norm,5.)
        np.testing.assert_array_equal(y,codes@decoder)
        order=np.random.default_rng(91).permutation(len(rows))
        other,_,_=task_contrast_basis(codes[order],decoder,[rows[i] for i in order])
        np.testing.assert_allclose(b,other)
        with self.assertRaises(ValueError):task_contrast_basis(np.zeros_like(codes),decoder,rows)

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
