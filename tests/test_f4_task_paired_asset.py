import copy
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_f4_task_paired_asset import generate


class TaskPairedTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/f4_task_paired_asset_v1.json').read_text())
        self.old=json.loads((ROOT/self.cfg['excluded_task_config']).read_text())

    def test_deterministic_whole_template_split_and_pairing(self):
        docs,rows=generate(self.cfg,self.old)
        self.assertEqual(len(docs),1024);self.assertEqual(len(rows),512)
        self.assertEqual(len({r['text'] for r in rows}),512)
        self.assertEqual(generate(self.cfg,self.old),(docs,rows))
        for d in docs:self.assertEqual({r['split'] for r in d['rows']},{d['split']})
        self.assertEqual({r['split'] for r in rows},{'discovery'})
        for a,b in zip(rows[::2],rows[1::2]):
            self.assertEqual(a['template'],b['template']);self.assertEqual(a['attractor_number'],b['attractor_number'])
            self.assertEqual((a['subject_number'],b['subject_number']),(0,1))

    def test_reserved_vocabulary_rejected(self):
        c=copy.deepcopy(self.cfg);c['subjects'][0]=['doctor','doctors']
        with self.assertRaises(ValueError):generate(c,self.old)


if __name__=='__main__':unittest.main()
