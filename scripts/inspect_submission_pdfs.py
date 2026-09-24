import argparse
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in args.pdf:
        destination = args.output / path.stem
        destination.mkdir(exist_ok=True)
        document = pdfium.PdfDocument(path)
        reader = PdfReader(path)
        pages = []
        previews = []
        for i, page in enumerate(document):
            bitmap = page.render(scale=1.5)
            rendered = bitmap.to_pil()
            rendered.save(destination / f'page_{i + 1:02d}.png')
            thumb = rendered.copy()
            thumb.thumbnail((400, 560))
            previews.append(thumb)
            content = reader.pages[i].extract_text()
            (destination / f'page_{i + 1:02d}.txt').write_text(content, encoding='utf-8')
            pages.append({'page': i + 1, 'size_pt': list(page.get_size()), 'text_characters': len(content)})
            print(f'{path.name}: rendered page {i + 1}/{len(document)}', flush=True)
            page.close()
        for start in range(0, len(previews), 6):
            sheet = Image.new('RGB', (1230, 1180), 'white')
            draw = ImageDraw.Draw(sheet)
            for offset, preview in enumerate(previews[start:start + 6]):
                x, y = (offset % 3) * 410, (offset // 3) * 590
                draw.text((x + 5, y + 4), f'{path.name} — page {start + offset + 1}', fill='black')
                sheet.paste(preview, (x, y + 25))
            sheet.save(destination / f'overview_{start + 1:02d}.png')
        records.append({'pdf': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'pages': pages, 'metadata': dict(reader.metadata)})
        document.close()
    (args.output / 'PDF_INSPECTION.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
