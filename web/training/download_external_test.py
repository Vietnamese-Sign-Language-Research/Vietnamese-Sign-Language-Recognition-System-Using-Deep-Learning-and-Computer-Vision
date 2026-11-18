"""Download independent-source evaluation images, never training inputs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import requests

ROOT = Path('artifacts/alphabet-audit/external')
REPO = 'TuanKiet1774/SignLanguage_Vietnamese'
SHA = '2e44477b2939f14be2fbfd222191af9e617448f4'
LABELS = set('ABCDEGHIKLMNOPQRSTUVXY') | {'Dd'}


def fetch(item):
    path, label = item
    target = ROOT / 'images' / label / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        r = requests.get(f'https://raw.githubusercontent.com/{REPO}/{SHA}/{path}', timeout=25)
        r.raise_for_status()
        target.write_bytes(r.content)
    return {'path': str(target), 'label': 'Đ' if label == 'Dd' else label,
            'source_path': path}


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    r = requests.get(f'https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1', timeout=30)
    r.raise_for_status()
    files = r.json()['tree']
    selected = []
    for label in sorted(LABELS):
        paths = sorted(x['path'] for x in files if x['path'].startswith(f'data/{label}/')
                       and x['path'].lower().endswith(('.jpg', '.jpeg', '.png')))
        if not paths:
            continue
        indices = sorted(set(round(i * (len(paths) - 1) / 19) for i in range(20)))
        selected.extend((paths[i], label) for i in indices)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = []
        for i, result in enumerate(pool.map(fetch, selected), 1):
            records.append(result)
            if i % 50 == 0:
                print(f'Downloaded {i}/{len(selected)} independent images', flush=True)
    (ROOT/'manifest.json').write_text(json.dumps({'repo':REPO,'commit':SHA,
       'purpose':'evaluation only; adjacent frames are correlated; signer independence from training is unverified',
       'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print('External test set ready:',len(records),flush=True)
