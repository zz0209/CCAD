"""Verify a last-block residual is compared before the final normalization."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'requires the pinned experiment runtime')
class FinalHookQualityTest(unittest.TestCase):
    def test_last_block_and_ordinary_block_oracles(self):
        import torch
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        from run_r006b_topk_capacity import evaluate, HookPointContract
        from types import SimpleNamespace
        torch.set_num_threads(2)
        torch.manual_seed(14)
        model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=16, attn_implementation='eager')).eval()

        class IdentityCoder:
            num_latents = 16
            def encode(self, value):
                return SimpleNamespace(top_acts=value,
                    top_indices=torch.arange(16).expand_as(value))
            def decode(self, acts, indices):
                return acts

        tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        observed = []
        handle = model.gpt_neox.layers[-1].register_forward_hook(
            lambda m, a, value: observed.append(value.detach().clone()))
        with torch.no_grad():
            output = model(tokens, output_hidden_states=True)
        handle.remove()
        self.assertGreater(float((observed[0] - output.hidden_states[-1]).abs().max()), .1)
        data = [{'input_ids': row} for row in tokens]
        for layer in [0, 1]:
            contract = HookPointContract(f'gpt_neox.layers.{layer}', layer, 'resid_post', 16)
            quality = evaluate(model, IdentityCoder(), model.gpt_neox.layers[layer],
                               contract, layer + 1, data, 2, 'cpu', torch)
            self.assertEqual(quality['hook_oracle_max_error'], 0)
            self.assertEqual(quality['capture_logit_max_error'], 0)
            self.assertEqual(quality['fve'], 1)
            self.assertEqual(quality['ce']['clean'], quality['ce']['reconstruction'])
            self.assertEqual(quality['hook_oracle_source'],
                'final_layer_norm_input' if layer == 1 else 'model_hidden_states')


if __name__ == '__main__':
    unittest.main()
