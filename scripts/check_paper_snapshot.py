from pathlib import Path
import argparse
import json

import pypdfium2 as pdfium
from PIL import ImageChops

DESCRIPTION = ('Compare a rebuilt paper PDF with a reference PDF: page count, title metadata, the text of every page and '
               'the rendered pixels of every page; prints one JSON line.')


def pages(path):
    doc = pdfium.PdfDocument(str(path))
    texts, images = [], []
    for k in range(len(doc)):
        page = doc[k]
        tp = page.get_textpage()
        texts.append(tp.get_text_range())
        tp.close()
        images.append(page.render(scale=1.).to_pil().convert('RGB'))
        page.close()
    return doc.get_metadata_dict(), texts, images


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rebuilt', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    meta_a, text_a, img_a = pages(args.rebuilt) if not args.metadata_only else (pdfium.PdfDocument(str(args.rebuilt)).get_metadata_dict(), [], [])
    meta_b, text_b, img_b = pages(args.reference)
    if args.metadata_only:
        print(json.dumps(dict(rebuilt_title=meta_a.get('Title'), reference_title=meta_b.get('Title'))))
        return
    same_text = [a == b for a, b in zip(text_a, text_b)]
    diff_pixels = []
    for a, b in zip(img_a, img_b):
        if a.size != b.size:
            diff_pixels.append(-1)
            continue
        box = ImageChops.difference(a, b).getbbox()
        diff_pixels.append(0 if box is None else (box[2] - box[0]) * (box[3] - box[1]))
    print(json.dumps(dict(pages_rebuilt=len(text_a), pages_reference=len(text_b), title_rebuilt=meta_a.get('Title'),
                          title_reference=meta_b.get('Title'), pages_same_text=sum(same_text),
                          pages_text_differ=[k + 1 for k, s in enumerate(same_text) if not s],
                          pages_pixel_identical=sum(d == 0 for d in diff_pixels),
                          pages_pixel_differ=[k + 1 for k, d in enumerate(diff_pixels) if d != 0])))


if __name__ == '__main__':
    main()
