import itertools
import unittest

import torch

from ccad.request_capacity import part_request_columns


class RequestCapacityTests(unittest.TestCase):
    def test_group_cancellation(self):
        codes = torch.tensor([[1., 1.]], dtype=torch.float64)
        basis = torch.tensor([[2., -2.]], dtype=torch.float64, requires_grad=True)
        columns, _, _, _ = part_request_columns(codes, basis, torch.tensor([0, 0]),
            torch.tensor([[1.]], dtype=torch.float64), torch.ones(1), 1)
        self.assertEqual(columns.item(), 0.)
        columns.square().sum().backward()
        self.assertTrue(torch.isfinite(basis.grad).all())

    def test_all_request_vertices(self):
        generator = torch.Generator().manual_seed(9311)
        codes = torch.rand(7, 12, generator=generator, dtype=torch.float64)
        basis = torch.randn(9, 12, generator=generator, dtype=torch.float64, requires_grad=True)
        z = torch.rand(7, 9, generator=generator, dtype=torch.float64)
        z[:, 0] = 0
        columns, parts, _, _ = part_request_columns(codes, basis, torch.arange(12)//4, z, torch.ones(9), 5)
        self.assertEqual(len(parts), 3)
        for vertex in itertools.product([0., 1.], repeat=3):
            delta = columns@torch.tensor(vertex, dtype=torch.float64)
            self.assertGreaterEqual(float((z+delta).min().detach()), -1e-12)
            self.assertTrue(((delta != 0).sum(-1) <= 5).all())
        columns.square().sum().backward()
        self.assertTrue(torch.isfinite(basis.grad).all())

    def test_member_support_is_preserved(self):
        generator = torch.Generator().manual_seed(9312)
        codes = torch.rand(7, 12, generator=generator, dtype=torch.float64)
        basis = torch.randn(9, 12, generator=generator, dtype=torch.float64)
        z = torch.rand(7, 9, generator=generator, dtype=torch.float64)
        z[:, 0] = 0
        score = (codes@basis.abs().T)*(z > 0)
        allowed = torch.zeros_like(score).scatter(-1, score.topk(5, dim=-1).indices, 1.)
        columns, _, _, _ = part_request_columns(codes, basis, torch.arange(12)//4, z,
                                                torch.ones(9), 5, member_support=True)
        self.assertTrue((columns*(1-allowed.unsqueeze(-1)) == 0).all())
        for vertex in itertools.product([0., 1.], repeat=3):
            self.assertGreaterEqual(float((z+columns@torch.tensor(vertex, dtype=torch.float64)).min()), -1e-12)

    def test_coordinate_error_is_monotone(self):
        generator = torch.Generator().manual_seed(9313)
        codes = torch.rand(64, 12, generator=generator, dtype=torch.float64)
        basis = torch.randn(9, 12, generator=generator, dtype=torch.float64)
        z = torch.rand(64, 9, generator=generator, dtype=torch.float64)
        raw = codes[:, None, :]*basis[None, :, :]
        eta = (z/(-raw).clamp_min(0).sum(-1).clamp_min(1e-20)).clamp_max(1)
        members = raw.clamp_min(0)-(-raw).clamp_min(0)*eta.unsqueeze(-1)
        grouped = raw.reshape(64, 9, 3, 4).sum(-1)
        allocated = members.reshape(64, 9, 3, 4).sum(-1)
        parts, _, _, _ = part_request_columns(codes, basis, torch.arange(12)//4, z, torch.ones(9), 9)
        error, member_error = parts-grouped, allocated-grouped
        self.assertGreaterEqual(float(error.min()), -1e-12)
        self.assertLessEqual(float((error-member_error).max()), 1e-12)


if __name__ == '__main__':
    unittest.main()
