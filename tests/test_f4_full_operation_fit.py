import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_f4_full_operation_fit import operation_pairs


class OperationPairsTests(unittest.TestCase):
    def test_source_only_classes_documents_and_unique_budget(self):
        positive=np.array([0,1,2,3]);scores=np.array([0,0,0,0,10,9,8,7.]);coord=np.arange(8.)
        labels=['w','p','w','p','p','w','w','p'];docs={0:{'a'},1:{'b'},2:{'a'},3:{'c'}}
        rr,dd,missing,pool=operation_pairs(positive,scores,coord,labels.__getitem__,docs,2,count=4)
        self.assertEqual(len(set(rr.tolist()+dd.tolist())),2*len(rr))
        self.assertTrue(set(dd).issubset(set(pool)))
        for r,d in zip(rr,dd):
            self.assertEqual(labels[r],labels[d]);self.assertFalse(docs[r//2]&docs[d//2])
        self.assertEqual(set(rr)|set(missing),set(positive))

    def test_no_pool_expansion_when_class_unavailable(self):
        rr,dd,missing,pool=operation_pairs([0,1],np.array([0,0,10,1.]),np.arange(4.),lambda r:'w' if r<2 or r==3 else 'p',{i:{str(i)} for i in range(4)},1,count=1)
        self.assertEqual(rr.size,0);self.assertEqual(dd.size,0);self.assertEqual(missing,[0]);self.assertEqual(pool.tolist(),[2])


if __name__=='__main__':unittest.main()
