"""Fetch a version-pinned public archive and record its checksum and metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('version', type=int)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = requests.get('https://www.kaggle.com/api/v1/datasets/view/' + args.dataset, timeout=30)
    metadata.raise_for_status()
    info = metadata.json()
    if not args.output.exists():
        url = f'https://www.kaggle.com/api/v1/datasets/download/{args.dataset}?datasetVersionNumber={args.version}'
        r = requests.get(url, stream=True, timeout=(25, 90))
        r.raise_for_status()
        if 'zip' not in r.headers.get('content-type', ''):
            raise RuntimeError('Expected a public ZIP download')
        partial = args.output.with_suffix('.partial')
        total, last = 0, time.monotonic()
        with partial.open('wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
                total += len(chunk)
                if time.monotonic() - last > 10:
                    print(f'Downloaded {total / 1024**2:.1f} MiB', flush=True)
                    last = time.monotonic()
        partial.replace(args.output)
    digest = hashlib.sha256()
    with args.output.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            digest.update(block)
    record = dict(dataset=args.dataset, version=args.version, sha256=digest.hexdigest(),
                  bytes=args.output.stat().st_size, license=info.get('licenseName'), metadata=info)
    args.output.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Ready:', args.output, digest.hexdigest(), flush=True)


if __name__ == '__main__':
    main()
