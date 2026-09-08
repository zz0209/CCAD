"""Linear scan of ordered, nonoverlapping document spans in a token stream."""


def sequence_document_ids(spans, count, width):
    """Yield the exact overlap lists, preserving input order and EOS gaps."""
    if count < 0 or width <= 0:
        raise ValueError('Invalid sequence count or context width')
    previous_end = -1
    for span in spans:
        if span['start'] < previous_end or span['end'] < span['start']:
            raise ValueError('Packing requires ordered nonoverlapping spans')
        previous_end = span['end']
    cursor = 0
    for index in range(count):
        start, end = index*width, (index+1)*width
        while cursor < len(spans) and spans[cursor]['end'] <= start:
            cursor += 1
        j = cursor
        names = []
        while j < len(spans) and spans[j]['start'] < end:
            if spans[j]['end'] > start:
                names.append(spans[j]['document_id'])
            j += 1
        yield names
