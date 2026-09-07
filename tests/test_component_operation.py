import tempfile,unittest
from pathlib import Path
import numpy as np
from ccad.component_operation import ComponentOperation


class ComponentOperationTests(unittest.TestCase):
    def test_partial_donor_and_removal_match_explicit_source_member_sum(self):
        rng=np.random.default_rng(5);h=rng.normal(size=(4,3));d=rng.normal(size=(3,7));z=rng.normal(size=(5,4));donor=rng.normal(size=(5,4));mask=np.array([1.,0.,-.5])
        op=ComponentOperation(np.array([2,8,4,9]),np.array([12,14,27]),h,d,{'hook':'example'})
        for kind,code in [('removal',-z),('contrast',donor-z)]:
            expected=sum(.7*mask[j]*(code@h[:,j])[:,None]*d[j] for j in range(3))
            np.testing.assert_allclose(op.apply(z,mask,kind,donor,.7),expected,atol=1e-14)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'map.npz';op.save(p);loaded=ComponentOperation.load(p)
            np.testing.assert_array_equal(loaded.source_members,op.source_members)
            np.testing.assert_allclose(loaded.apply(z,mask),op.apply(z,mask),atol=0)

    def test_required_donor_shape_and_finite_masks(self):
        op=ComponentOperation([0,1],[3,5],np.eye(2),np.eye(2),{})
        with self.assertRaises(ValueError):op.apply(np.ones((2,2)),consumer='contrast')
        with self.assertRaises(ValueError):op.apply(np.ones((2,2)),[np.nan,1])
        with self.assertRaises(ValueError):op.apply(np.ones((2,3)))
        with self.assertRaises(ValueError):ComponentOperation([0,0],[1,2],np.eye(2),np.eye(2),{})


if __name__=='__main__':unittest.main()
