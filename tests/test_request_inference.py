import unittest

import torch

from ccad.request_inference import request_second_moment, solve_request_columns


class TestRequestInference(unittest.TestCase):
    def test_request_moment_exact_uniform_quadrature(self):
        dtype = torch.float64
        groups = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=dtype)
        nodes = torch.tensor([0.5 - 1 / (12 ** 0.5), 0.5 + 1 / (12 ** 0.5)], dtype=dtype)
        requests = torch.cartesian_prod(nodes, nodes) @ groups.T
        expected = requests.T @ requests / len(requests)
        self.assertTrue(torch.allclose(request_second_moment(groups, 0), expected))

    def test_active_capacity_has_request_dependent_optimum(self):
        dtype = torch.float64
        gram = torch.ones((1, 1), dtype=dtype)
        rhs = torch.tensor([[-2., -1.]], dtype=dtype)
        capacity = torch.ones(1, dtype=dtype)
        initial = torch.zeros_like(rhs)
        q = torch.diag(torch.tensor([1., 4.], dtype=dtype))
        result = solve_request_columns(gram, rhs, capacity, initial, q, 512)
        self.assertTrue(torch.allclose(result, torch.tensor([[-0.4, -0.6]], dtype=dtype), atol=1e-9))
        identity = solve_request_columns(gram, rhs, capacity, initial, torch.eye(2, dtype=dtype), 512)
        self.assertTrue(torch.allclose(identity, torch.tensor([[-1., 0.]], dtype=dtype), atol=1e-9))
        vertices = torch.cartesian_prod(*[torch.tensor([0., 1.], dtype=dtype)] * 2)
        self.assertGreaterEqual(float((capacity[:, None] + result @ vertices.T).min()), -1e-12)

    def test_inactive_capacity_preserves_positive_definite_solution(self):
        dtype = torch.float64
        gram = torch.tensor([[2., 0.2], [0.2, 1.]], dtype=dtype)
        rhs = torch.tensor([[1., 0.4], [0.1, 0.8]], dtype=dtype)
        q = torch.tensor([[1., 0.3], [0.3, 2.]], dtype=dtype)
        result = solve_request_columns(gram, rhs, torch.ones(2, dtype=dtype), torch.zeros_like(rhs), q, 512)
        self.assertTrue(torch.allclose(result, torch.linalg.solve(gram, rhs), atol=1e-9))


if __name__ == '__main__':
    unittest.main()
