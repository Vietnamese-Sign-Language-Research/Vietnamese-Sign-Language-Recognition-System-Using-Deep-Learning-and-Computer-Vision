"""Keep national-chart H/K/R labels; supplement K with reviewed raw views.

The supplemental H clips are intentionally excluded: their three-raised-finger
form does not match the national-chart H used by the skeleton source.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letter_skeleton import LETTERS,render_hand
from letter_landmarks import normalized_points
from prepare_blended_skeletons import ROOT,process_clip


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    source=ROOT/'vsl-corrected/skeleton-v2'
    target=ROOT/'vsl-corrected/skeleton-v4';target.mkdir(exist_ok=True)
    data=np.load(source/'skeletons.npz',allow_pickle=False)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    keep=np.array([not (r['label']=='H' and r.get('source') in ['hauuto','ngannguyen65']) for r in manifest['records']])
    records=[r for r,k in zip(manifest['records'],keep) if k]
    images=list(data['images'][keep]);labels=list(data['labels'][keep]);splits=list(data['splits'][keep])
    # Retain the fixed split. Perspective augmentation is generated exclusively
    # from training clips; no held-out raw image or pose enters training.
    import cv2,mediapipe as mp,tempfile,zipfile
    rng=np.random.default_rng(20261120)
    jobs={r['path']:r for r in records if r.get('source')=='hauuto' and r['label']=='K' and r['split']=='train'}
    with tempfile.TemporaryDirectory(dir=target) as folder,zipfile.ZipFile(ROOT/'vsl-corrected/hauuto-v2.zip') as archive,mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1,model_complexity=1,min_detection_confidence=.3) as tracker:
        for name,record in jobs.items():
            p=Path(folder)/Path(name).name;p.write_bytes(archive.read(name))
            cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame in np.linspace(round(n*.3),round(n*.7),24).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,im=cap.read()
                if not ok:continue
                tracked=tracker.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                if not tracked.multi_hand_landmarks:continue
                points=normalized_points(tracked.multi_hand_landmarks[0].landmark,im.shape[1],im.shape[0])
                for variant in range(12):
                    changed=points.copy()
                    for first,second,amount in [(0,2,.65),(1,2,.4),(0,1,.55)]:
                        a=rng.uniform(-amount,amount);x,y=changed[:,first].copy(),changed[:,second].copy()
                        changed[:,first]=np.cos(a)*x-np.sin(a)*y;changed[:,second]=np.sin(a)*x+np.cos(a)*y
                    rendered=render_hand([SimpleNamespace(x=float(p[0])*100,y=float(p[1])*100) for p in changed],1,1)
                    images.append(rendered);labels.append(LETTERS.index('K'));splits.append('train')
                    records.append(dict(record,frame=int(frame),augmentation='3D camera rotation',variant=variant))
            cap.release();print('Augmented reviewed training K',name,flush=True)
    np.savez_compressed(target/'skeletons.npz',images=np.array(images),labels=np.array(labels),splits=np.array(splits))
    manifest.update(records=records,supplemental_review='Only reviewed K raw clips. ALL supplemental H/R and old-corpus H/K/R excluded. H/R use national-chart skeleton corpus.',warning='Source-block split. Signer overlap between sources unknown. Raw H in other collections is incompatible with this standard, so no raw H success claim is made.')
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared national-standard candidate',len(images),flush=True)
