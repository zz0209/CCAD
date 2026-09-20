import unittest
from types import SimpleNamespace
import torch
from ccad.intervention_transport import pursuit_columns, project_capacity


class TestPursuitTransport(unittest.TestCase):
    def test_decoder_choice_recovers_feasible_direction(self):
        dtype = torch.float64
        target = SimpleNamespace(decoder=SimpleNamespace(weight=torch.tensor([[-1., 0., 1.], [0., 1., 0.]], dtype=dtype)),
                                 encode=lambda h: torch.zeros((*h.shape[:-1], 3), dtype=dtype))
        source = dict(center=torch.zeros(2, dtype=dtype), encoder=torch.tensor([[1., 0.]], dtype=dtype),
                      encoder_bias=torch.zeros(1, dtype=dtype), decoder=torch.tensor([[1., 0.]], dtype=dtype))
        columns = pursuit_columns(torch.tensor([[1., 0.], [0., 0.]], dtype=dtype), target, source, 1)
        self.assertTrue(torch.allclose(columns[0, :, 0], torch.tensor([1., 0., 0.], dtype=dtype)))
        self.assertEqual(float(columns[1].abs().sum()), 0.)

    def test_capacity_all_request_vertices(self):
        values = torch.tensor([[[-3., -2., 1.], [1., -4., -2.]]], dtype=torch.float64)
        capacity = torch.tensor([[2., 1.]], dtype=torch.float64)
        columns = project_capacity(values, capacity)
        requests = torch.cartesian_prod(*[torch.tensor([0., 1.], dtype=values.dtype)] * 3)
        self.assertGreaterEqual(float((capacity[..., None] + columns @ requests.T).min()), -1e-12)
        self.assertTrue(torch.allclose(columns[0, 0], torch.tensor([-1.5, -.5, 1.], dtype=values.dtype)))


if __name__ == '__main__':
    unittest.main()
