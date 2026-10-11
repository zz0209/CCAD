from pathlib import Path
import argparse

import pypdfium2 as pdfium

DESCRIPTION = 'Render pages of a PDF to PNG files for visual inspection.'


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--pages', default=None, help='first-last, one-based and inclusive')
    parser.add_argument('--scale', type=float, default=2.)
    args = parser.parse_args()
    doc = pdfium.PdfDocument(str(args.pdf))
    first, last = (1, len(doc)) if args.pages is None else map(int, args.pages.split('-'))
    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(first, last + 1):
        doc[k - 1].render(scale=args.scale).to_pil().save(args.out / f'page_{k:02d}.png')
    print(f'{len(doc)} pages, rendered {first} to {last}')


if __name__ == '__main__':
    main()
