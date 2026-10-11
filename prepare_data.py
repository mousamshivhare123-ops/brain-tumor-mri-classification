"""Normalize the uploaded dataset's split/class folder names.
Run: python prepare_data.py --zip /path/to/tumor.zip.zip --out data
"""
import argparse, zipfile, shutil
from pathlib import Path
from PIL import Image

ALIASES = {
    "glioma": "Glioma", "giloma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "no_tumor": "No_Tumor", "no tumor": "No_Tumor",
}
SPLIT_ALIASES = {"train": "train", "valid": "valid", "validation": "valid", "test": "test"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True, help='Path to uploaded ZIP dataset')
    ap.add_argument('--out', default='data', help='Output data directory')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists(): shutil.rmtree(out)
    counts = {}
    with zipfile.ZipFile(args.zip) as z:
        for name in z.namelist():
            p = Path(name)
            parts = p.parts
            if len(parts) < 4 or p.suffix.lower() not in EXTS:
                continue
            split = SPLIT_ALIASES.get(parts[-3].lower())
            label = ALIASES.get(parts[-2].lower())
            if not split or not label: continue
            dest_dir = out / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / p.name
            # Avoid collisions if two folders happen to contain the same filename.
            if dest.exists(): dest = dest_dir / f'{p.parent.name}_{p.name}'
            with z.open(name) as src, open(dest, 'wb') as dst: shutil.copyfileobj(src, dst)
            try:
                with Image.open(dest) as im: im.verify()
            except Exception:
                dest.unlink(missing_ok=True)
                continue
            counts[(split, label)] = counts.get((split, label), 0) + 1
    print('Prepared dataset at:', out.resolve())
    for split in ('train','valid','test'):
        total = sum(v for (s,_),v in counts.items() if s == split)
        print(f'{split}: {total} images')
        for cls in ('Glioma','Meningioma','No_Tumor','Pituitary'):
            print(f'  {cls}: {counts.get((split,cls),0)}')
    if not all((out/s).exists() for s in ('train','valid','test')):
        raise SystemExit('Expected train, valid and test folders. Check ZIP structure.')
    print('\nNote: review class counts before training; do not assume README totals match the ZIP contents.')

if __name__ == '__main__': main()
