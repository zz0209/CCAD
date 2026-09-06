import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_f4_global_matching import full_assignment,pair_calibration


class MatchingTests(unittest.TestCase):
    def test_full_permuted_dictionary_preserves_group_operation_and_scale(self):
        source=np.diag([2.,3.,4.]);target=np.array([[0.,0.,-8.],[1.,0.,0.],[0.,6.,0.]])
        mapping,scale,stats=full_assignment(source,target)
        np.testing.assert_array_equal(mapping,[1,2,0]);np.testing.assert_allclose(scale,[.5,2.,-2.])
        zs=np.array([[1.,2.,3.],[4.,5.,6.]])
        zt=np.zeros_like(zs);zt[:,mapping]=zs/scale
        b=np.array([.2,-.1,.3]);source_weight=source@b
        np.testing.assert_allclose((zt[:,mapping]*scale)@source_weight,zs@source_weight)
        self.assertEqual(stats['mean_absolute_cosine'],1.)

    def test_all_dictionary_rows_compete_for_targets(self):
        source=np.array([[1.,0.],[.9,.1]]);target=np.array([[1.,0.],[0.,1.]])
        mapping,_,_=full_assignment(source,target)
        self.assertEqual(len(set(mapping)),2)
        self.assertEqual(mapping.tolist(),[0,1])

    def test_pair_calibration_replays_differences_and_ignores_offsets(self):
        x=np.array([[1.,2.,9.],[2.,4.,9.],[4.,3.,9.],[8.,7.,9.]])
        y=x*np.array([2.,-.5,0.])+np.array([7.,2.,5.]);w=np.array([1.,2.,3.,4.])
        slope,diag=pair_calibration(y,x,w,0.)
        np.testing.assert_allclose(slope,[2.,-.5,0.]);self.assertEqual(diag['zero_target_variance'],1)
        np.testing.assert_allclose((x[1:]-x[:-1])*slope,y[1:]-y[:-1])
        scaled,_=pair_calibration(y+100.,x-20.,w,.001)
        np.testing.assert_allclose(scaled,slope/1.001)

    def test_zero_decoder_has_finite_assignment_and_no_geometric_term(self):
        mapping,scale,stats=full_assignment(np.array([[1.,0.],[0.,0.]]),np.eye(2))
        self.assertTrue(np.isfinite(scale).all());self.assertEqual(scale[1],0.)
        self.assertEqual(stats['zero_norm_source'],1)


if __name__=='__main__':unittest.main()
