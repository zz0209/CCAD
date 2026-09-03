from __future__ import annotations

import unittest

import numpy as np

from ccad.activation_contract import (
    ActivationContractError,
    HookPointContract,
    assert_token_alignment,
    build_token_alignment_record,
    decoded_contribution,
    dynamic_group_ablation,
    dynamic_group_swap,
    extract_primary_hook_tensor,
    replace_primary_hook_tensor,
    sae_round_trip,
)


class ActivationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = HookPointContract(
            module_path="gpt_neox.layers.2",
            layer_index=2,
            tensor_kind="resid_post",
            hidden_size=4,
        )

    def test_tuple_hook_capture_and_replacement_preserve_auxiliary_output(self) -> None:
        hidden = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        cache = {"sentinel": 7}
        output = (hidden, cache)
        self.assertIs(extract_primary_hook_tensor(output, self.contract), hidden)
        replacement = hidden + 1
        replaced = replace_primary_hook_tensor(output, replacement, self.contract)
        self.assertIs(replaced[0], replacement)
        self.assertIs(replaced[1], cache)

    def test_hook_replacement_rejects_shape_and_dtype_drift(self) -> None:
        hidden = np.zeros((1, 3, 4), dtype=np.float32)
        with self.assertRaises(ActivationContractError):
            replace_primary_hook_tensor(hidden, np.zeros((1, 2, 4), dtype=np.float32), self.contract)
        with self.assertRaises(ActivationContractError):
            replace_primary_hook_tensor(hidden, np.zeros((1, 3, 4), dtype=np.float64), self.contract)

    def test_token_hash_detects_order_or_revision_drift(self) -> None:
        kwargs = dict(
            dataset_id="r004-fixture-v1", document_id="doc-0001",
            tokenizer_id="EleutherAI/pythia-70m-deduped", tokenizer_revision="main@resolved-commit",
            input_ids=[1, 7, 9, 2], attention_mask=[1, 1, 1, 1],
        )
        reference = build_token_alignment_record(**kwargs)
        assert_token_alignment(reference, build_token_alignment_record(**kwargs))
        kwargs["input_ids"] = [1, 9, 7, 2]
        with self.assertRaises(ActivationContractError):
            assert_token_alignment(reference, build_token_alignment_record(**kwargs))

    def test_round_trip_and_dynamic_interventions_share_hook_units(self) -> None:
        rng = np.random.default_rng(20260902)
        codes = rng.standard_normal((2, 3, 5))
        decoder = rng.standard_normal((4, 5))
        bias = rng.standard_normal(4)
        reconstruction = bias + decoded_contribution(decoder, codes)
        hook = reconstruction + 0.01 * rng.standard_normal(reconstruction.shape)
        observed, residual = sae_round_trip(hook, bias, decoder, codes)
        np.testing.assert_allclose(observed + residual, hook, rtol=0, atol=1e-12)
        group = decoded_contribution(decoder, codes, [1, 3])
        np.testing.assert_allclose(dynamic_group_ablation(hook, decoder, codes, [1, 3]), hook - group)
        np.testing.assert_allclose(
            dynamic_group_swap(hook, decoder, codes, [1, 3], decoder, codes, [1, 3]),
            hook,
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
