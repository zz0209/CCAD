import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_r008a_paired_corpus import read_document_exclusions


class ExclusionReaderTests(unittest.TestCase):
    def test_original_jsonl(self):
        row={'document_id':'a','text_sha256':'0'*64}
        with patch.object(Path,'read_text',return_value=json.dumps(row)+'\n'):
            self.assertEqual(read_document_exclusions(Path('old.jsonl')),[row])

    def test_long_training_json(self):
        rows=[{'document_id':'a','text_sha256':'0'*64,'split':'train'},
              {'document_id':'b','text_sha256':'1'*64,'split':'validation'}]
        with patch.object(Path,'read_text',return_value=json.dumps({'documents':rows})):
            self.assertEqual(read_document_exclusions(Path('long.json')),rows)

    def test_bad_hash_and_empty_ledger_rejected(self):
        for rows in ([],[{'document_id':'a','text_sha256':'bad'}]):
            with patch.object(Path,'read_text',return_value=json.dumps({'documents':rows})):
                with self.assertRaises(ValueError):read_document_exclusions(Path('bad.json'))


if __name__=='__main__':unittest.main()
