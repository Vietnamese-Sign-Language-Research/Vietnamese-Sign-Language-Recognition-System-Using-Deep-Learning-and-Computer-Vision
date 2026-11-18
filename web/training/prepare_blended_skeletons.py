"""Historical mixed-form diagnostic; NOT the final national-chart dataset.

Later review found the supplemental H incompatible with the national chart.
prepare_national_skeletons removes all those H records before final training.

Do not import the known ASL-style K/R labels or disputed H from the old corpus.
No images from the separate-source diagnostic collection enter training.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import zipfile
import tempfile

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letter_skeleton import LETTERS, render_hand

ROOT=Path('artifacts/alphabet-audit')


def process_clip(item):
    label,person,name,split,local=item
    import mediapipe as mp
    cv2.setNumThreads(1)
    result=[]
    with tempfile.TemporaryDirectory(dir=ROOT/'vsl-corrected') as folder:
        if local:path=Path(name)
        else:
            path=Path(folder)/Path(name).name
            with zipfile.ZipFile(ROOT/'vsl-corrected/hauuto-v2.zip') as z:path.write_bytes(z.read(name))
        capture=cv2.VideoCapture(str(path));count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        selected=np.linspace(round(count*.3),round(count*.7),12).astype(int)
        with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1,model_complexity=1,min_detection_confidence=.5) as h:
            for frame_number in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES,int(frame_number));ok,image=capture.read()
                if not ok:continue
                tracked=h.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                if not tracked.multi_hand_landmarks:continue
                try:skeleton=render_hand(tracked.multi_hand_landmarks[0].landmark,image.shape[1],image.shape[0])
                except ValueError:continue
                result.append((skeleton,dict(label=label,person=person,path=name,frame=int(frame_number),split=split,source='ngannguyen65' if local else 'hauuto')))
        capture.release()
    return result


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');out=ROOT/'vsl-corrected/skeleton-v2';out.mkdir(exist_ok=True)
    base=ROOT/'vsl-corrected/skeleton';b=np.load(base/'skeletons.npz')
    images=list(b['images']);labels=list(b['labels']);splits=list(b['splits'])
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'));records=manifest['records']
    old=ROOT/'prepared';o=np.load(old/'features.npz');old_manifest=json.loads((old/'manifest.json').read_text(encoding='utf-8'))
    for i,r in enumerate(old_manifest['records']):
        if r['split']!='train' or r['label'] in ['H','K','R']:continue
        points=[SimpleNamespace(x=float(p[0])*100,y=float(p[1])*100) for p in o['points'][i]]
        images.append(render_hand(points,1,1));labels.append(LETTERS.index(r['label']));splits.append('train')
        records.append(dict(r,source='mcphngnga/dataset-vsl@2',kind='non-HKR training poses'))
    jobs=[]
    with zipfile.ZipFile(ROOT/'vsl-corrected/hauuto-v2.zip') as z:
        for name in z.namelist():
            if not name.startswith('raw/') or not name.endswith('.mp4'):continue
            person=Path(name).parts[-2];label=Path(name).stem.split('_')[0].upper()
            if label not in ['H','K']:continue
            if person in ['hau','tai']:split='train'
            elif person=='vy':split='test'
            elif person=='khoi' and label=='K':split='val'
            else:continue  # Khoi H displays a conflicting form and is excluded.
            jobs.append((label,person,name,split,False))
    for r in json.loads((ROOT/'vsl-corrected/native-photos/sources.json').read_text(encoding='utf-8')):
        if r['label'] not in ['H','K']:continue  # Crossed-finger R is not the pictured national form.
        ix=int(Path(r['local']).stem.split('-')[-1]);split='train' if ix in [0,40] else 'test'
        jobs.append((r['label'],'ngan',r['local'],split,True))
    supplementary=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n,rows in enumerate(pool.map(process_clip,jobs),1):
            supplementary.extend(rows)
            print('Reviewed clip extracted',n,'/',len(jobs),flush=True)
    for image,record in supplementary:
        repeats=5 if record['split']=='train' else 1
        for _ in range(repeats):
            images.append(image);labels.append(LETTERS.index(record['label']));splits.append(record['split']);records.append(record)
    np.savez_compressed(out/'skeletons.npz',images=np.array(images),labels=np.array(labels),splits=np.array(splits))
    manifest.update(records=records,supplemental_review='H/K middle-third raw recordings; Hau/Tai train, Vy test, Khoi K validation; conflicting Khoi H and crossed-finger R excluded.',
                    warning='Source-block split; supplemental people separated within that collection. Identity overlap with the original skeleton archive is unknown.')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Blended frames',len(images),'supplemental raw frames',len(supplementary),flush=True)
