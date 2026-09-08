import numpy as np
from ccad.native_operation import project_native, writable_support


def test_exact_writable_directions_and_dictionary_permutation():
    d = np.eye(4)
    signal = np.array([[0., 3., 0., 0.], [0., 0., 2., 0.]])
    members, _ = writable_support(d, signal, 2)
    assert set(members) == {1, 2}
    permutation = np.array([2, 0, 3, 1])
    permuted, _ = writable_support(d[permutation], signal, 2)
    assert set(permutation[permuted]) == {1, 2}
    z = np.full((2, 2), 4.)
    u, diagnostic = project_native(-signal, z, d[members])
    assert np.allclose(u @ d[members], -signal)
    assert diagnostic['minimum_final_state'] >= 0


def test_nonnegative_state_has_real_obstruction_and_coupled_refit_helps():
    d = np.array([[1., 0.], [1., 1.]])
    v = np.array([[-2., 2.]])
    z = np.array([[.1, .1]])
    u, diagnostic = project_native(v, z, d)
    assert np.min(u + z) >= -1e-12
    assert diagnostic['constrained_squared_error'] > 0
    assert diagnostic['constrained_squared_error'] < diagnostic['clipped_initial_squared_error']
    assert diagnostic['relative_projected_gradient'] < 1e-7
    gradient = (u @ d - v) @ d.T
    assert np.max(np.abs(gradient[u > -z + 1e-7])) < 1e-6
