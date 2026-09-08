import random
from ccad.token_packing import sequence_document_ids


def test_linear_packing_preserves_reference_overlap_and_eos_boundaries():
    rng = random.Random(14014)
    for width in [1, 7, 128]:
        spans = []
        position = 1
        for i in range(200):
            length = rng.randrange(1, 700)
            spans.append(dict(document_id=str(i), start=position, end=position+length))
            position += length + rng.randrange(1, 4)
        count = position//width + 3
        expected = [[s['document_id'] for s in spans if s['start'] < (i+1)*width and s['end'] > i*width] for i in range(count)]
        assert list(sequence_document_ids(spans, count, width)) == expected
