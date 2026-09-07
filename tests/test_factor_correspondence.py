import unittest
import numpy as np
from ccad.factor_correspondence import compact_source,assigned_readout,conditional_ot,group_support,ridge


class FactorCorrespondenceTests(unittest.TestCase):
    def test_compact_coordinates_preserve_intervention_not_just_mean(self):
        rng=np.random.default_rng(7);decoder=rng.normal(size=(12,9));dz=rng.normal(size=(40,12));train=np.arange(40)<20;sign=np.where(np.arange(40)%2,1.,-1.)
        source=compact_source(dz,decoder,train,sign,5);native=dz[:,source['support']]@decoder[source['support']]
        np.testing.assert_allclose(source['coordinates']@source['basis'].T,native,atol=1e-12)
        rotation,_=np.linalg.qr(rng.normal(size=(5,5)))
        np.testing.assert_allclose((source['coordinates']@rotation)@(source['basis']@rotation).T,native,atol=1e-12)

    def test_distinct_rescaled_atoms_are_recovered_by_assignment(self):
        rng=np.random.default_rng(8);xs=rng.normal(size=(200,4));xt=np.column_stack([xs[:,2]*3,xs[:,0]*.2,rng.normal(size=200),-xs[:,3]*2,xs[:,1]*4]);a=rng.normal(size=(4,3))
        w,diag=assigned_readout(xs,xt,a)
        np.testing.assert_allclose(xt@w,xs@a,atol=1e-11);self.assertEqual(len(set(diag['target_members'])),4)
        ot,_=conditional_ot(xs,xt,a,budget=4)
        self.assertLess(np.mean((xt@ot-xs@a)**2),1e-10)

    def test_group_fit_transfers_sparse_multidimensional_signal(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(240,30));w=np.zeros((30,3));w[[3,12,23]]=rng.normal(size=(3,3));y=x@w
        selected,_=group_support(x[:120],y[:120],3)
        self.assertEqual(set(selected),{3,12,23});beta=ridge(x[:120,selected],y[:120],1e-6)
        self.assertLess(np.mean((x[120:,selected]@beta-y[120:])**2),1e-8)

    def test_native_units_bound_rare_feature_extrapolation(self):
        rng=np.random.default_rng(10);a=rng.normal(size=100);x=np.column_stack([a,1e-5*a]);y=a[:,None]
        b=rng.normal(size=100);novel=np.column_stack([b,rng.normal(size=100)])
        physical=ridge(x,y,.01,'native_units');standard=ridge(x,y,.01,'feature_rms')
        self.assertLess(np.mean((novel@physical-b[:,None])**2),1e-3)
        self.assertGreater(np.mean((novel@standard-b[:,None])**2),1e6)


if __name__=='__main__':unittest.main()
