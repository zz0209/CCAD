import json
import unittest
from pathlib import Path

import numpy as np

from train_grammar_member_program import single_part_requests, source_measurement_key
from train_grammar_material_support import load_config


class GrammarSinglePartsScheduleTest(unittest.TestCase):
    def test_real_fit_panel_all_parts_and_exact_batch_reuse(self):
        config = load_config('configs/rg08_grammar_program_development.json')
        panel = json.loads(Path(config['panel']).read_text())
        fit = [row for task in config['tasks'] for row in
               [row for row in panel['rows'] if row['task'] == task and row['split'] == 'fit']
               [:config['fit_pairs_per_task']]]
        self.assertEqual(len(fit), 192)
        self.assertEqual(len({(row['task'], row['row_id']) for row in fit}), 192)
        requests = single_part_requests(config['steps'], len(fit), config['batch_pairs'])
        np.testing.assert_array_equal(requests.sum(1), np.ones(config['steps']))
        self.assertTrue(np.isin(requests, [0, 1]).all())
        rng = np.random.default_rng(config['training_seed'])
        rng.random((config['steps'], 3))
        order = rng.permutation(len(fit))
        batches = np.array([[order[(step*config['batch_pairs']+offset) % len(fit)]
                            for offset in range(config['batch_pairs'])] for step in range(config['steps'])])
        pairs = {(int(row), tuple(request)) for request, batch in zip(requests, batches) for row in batch}
        exact_batches = {(tuple(batch), tuple(request)) for request, batch in zip(requests, batches)}
        self.assertEqual(len(pairs), 576)
        self.assertEqual(len(exact_batches), 144)
        for row in range(len(fit)):
            observed = {request for index, request in pairs if index == row}
            self.assertEqual(observed, {tuple(vector) for vector in np.eye(3)})
        np.testing.assert_array_equal(requests[:144], requests[144:288])

    def test_cache_smoke_contains_an_exact_repeated_batch(self):
        requests = single_part_requests(26, 24, 4)
        np.testing.assert_array_equal(requests[0], requests[18])
        self.assertEqual((0*4) % 24, (18*4) % 24)

    def test_real_confirmation_reused_row_ids_keep_distinct_measurements(self):
        config = load_config('configs/cc23_single_parts_confirmation.json')
        original = json.loads(Path(config['panel']).read_text())
        confirmation = json.loads(Path(config['evaluation_panel']).read_text())
        fit = [row for row in original['rows'] if row['split'] == 'fit' and row['task'] in config['tasks']]
        measured = confirmation['rows']
        fit_ids = {(row['task'], row['row_id']) for row in fit}
        confirmation_ids = {(row['task'], row['row_id']) for row in measured}
        self.assertTrue(fit_ids & confirmation_ids)
        fit_keys = {source_measurement_key(1, row, [1, 0, 0]) for row in fit}
        confirmation_keys = {source_measurement_key(1, row, [1, 0, 0]) for row in measured}
        self.assertFalse(fit_keys & confirmation_keys)
        self.assertEqual(len(fit_keys | confirmation_keys), len(fit) + len(measured))


if __name__ == '__main__':
    unittest.main()
