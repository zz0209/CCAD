import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('causal',Path(__file__).resolve().parents[1]/'scripts/run_f4_source_reference_causal.py')
causal=importlib.util.module_from_spec(spec);spec.loader.exec_module(causal)


def test_difference_cancels_mean_and_matches_process_swap():
    rng=np.random.default_rng(71)
    recipient=rng.normal(size=(8,5));donor=rng.normal(size=(8,5));mean=rng.normal(size=5)
    p=[2,5];dp=[3,7];linear=rng.normal(size=(5,3))
    delta=causal.aligned_difference(recipient,donor,p,dp)
    np.testing.assert_allclose(delta,causal.aligned_difference(recipient-mean,donor-mean,p,dp),atol=1e-14)
    np.testing.assert_allclose(delta@linear,causal.aligned_difference(recipient@linear,donor@linear,p,dp),atol=1e-14)
    np.testing.assert_allclose((recipient-delta)[p],donor[dp],atol=1e-14)
    np.testing.assert_array_equal(delta[[0,1,3,4,6,7]],0)
    np.testing.assert_array_equal(causal.aligned_difference(recipient,recipient,[],[]),0)
    assert causal.compare(delta,np.zeros_like(delta))['normalized_error']==1.0


def test_source_dose_is_common_and_only_downscales():
    source=np.array([[3.0,4.0]]);hook=np.array([[6.0,8.0]]);candidate=source*2
    scale=causal.source_dose_scale(source,hook,0.1)
    assert scale==0.2
    np.testing.assert_allclose(np.linalg.norm(source*scale)/np.linalg.norm(hook),0.1)
    np.testing.assert_allclose(np.linalg.norm(candidate*scale)/np.linalg.norm(source*scale),2)
    assert causal.source_dose_scale(source,hook,0.8)==1
    assert causal.source_dose_scale(source,hook,None)==1
    assert causal.source_dose_scale(np.zeros_like(source),hook,0.1)==1
