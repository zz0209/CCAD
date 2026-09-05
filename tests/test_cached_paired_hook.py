import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_r008b_paired_codes import validate_cached_hook_config


class CachedHookTests(unittest.TestCase):
    def test_identity_and_split_restriction(self):
        config=dict(model_id='a',model_revision='r',hook_module_path='h',hook_hidden_size=2,
                    context_length=4,paired_corpus_run='c',token_manifest_sha256='s',attn_implementation='eager')
        rows={'splits':[{'split':'mean'},{'split':'discovery'},{'split':'calibration'}]}
        self.assertEqual(set(validate_cached_hook_config(config,config,rows,['mean','discovery'])),{'mean','discovery'})
        for bad in (dict(config,model_revision='new'),dict(config,hook_module_path='wrong')):
            with self.assertRaises(ValueError):validate_cached_hook_config(config,bad,rows,['mean'])
        with self.assertRaises(ValueError):validate_cached_hook_config(config,config,rows,['audit'])
        with self.assertRaises(ValueError):validate_cached_hook_config(dict(config,save_raw_hook=True),config,rows,['mean'])


if __name__=='__main__':unittest.main()
