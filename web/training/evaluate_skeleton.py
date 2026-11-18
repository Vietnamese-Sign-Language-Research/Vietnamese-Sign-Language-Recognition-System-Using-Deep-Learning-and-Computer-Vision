"""Release checks against held-out skeletons, raw photos, and existing models."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letter_skeleton import LETTERS, SkeletonNet, SkeletonRecognizer
from train_letters import metrics


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--approve',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    root=Path('artifacts/alphabet-audit');state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=SkeletonNet();model.load_state_dict(state['state_dict']);model.eval()
    d=np.load(root/'vsl-corrected/external-skeletons.npz')
    x=torch.from_numpy(d['images'].astype('float32')[:,None]/255)
    with torch.inference_mode():
        probabilities=torch.cat([model(b) for b in x.split(128)]).softmax(1).numpy()
    predictions=probabilities.argmax(1)
    prior=json.loads((root/'prepared/external_evaluation.json').read_text(encoding='utf-8'))
    report={'source_block_test':state['evaluation']['test'],
            'separate_source_diagnostic':metrics(d['labels'],predictions),
            'note':'The separate-source collection contains disputed H variants/labels; report it unchanged, not as certified VSL ground truth.'}
    common=d['labels']!=LETTERS.index('Đ')
    old=np.array([LETTERS.index(r['legacy']) if r['legacy'] in LETTERS else -1 for r in prior['predictions']])
    mlp=np.array([LETTERS.index(r['candidate']) for r in prior['predictions']])
    report['paired_common_22']={'old_yolo':float(np.mean(old[common]==d['labels'][common])),
                               'landmark_candidate':float(np.mean(mlp[common]==d['labels'][common])),
                               'skeleton_candidate':float(np.mean(predictions[common]==d['labels'][common]))}
    # These teaching/sample photos are not in the skeleton training archive.
    references=[('Y',Path('assets/samples/sample-1.jpg')),
                ('X',Path('assets/samples/sample-2.jpg')),
                ('H',root/'vsl-corrected/inspection/hanoi-teaching-H.jpg')]
    recognizer=SkeletonRecognizer(args.checkpoint,require_approved=False)
    raw=[]
    for expected,path in references:
        image=cv2.imread(str(path));result=recognizer.image_result(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        raw.append(dict(source=str(path),expected=expected,label=result.get('label'),status=result['status'],details=result.get('details')))
    manifest=json.loads((root/'external/prepared/manifest.json').read_text(encoding='utf-8'))
    records=manifest['records']
    with zipfile.ZipFile(root/'external/source.zip') as archive:
        for label in ['K','R']:
            chosen=sorted((r for r in records if r['label']==label),key=lambda r:r['path'])
            for r in chosen:
                image=cv2.imdecode(np.frombuffer(archive.read(r['path']),np.uint8),cv2.IMREAD_COLOR)
                result=recognizer.image_result(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                raw.append(dict(source=r['path'],expected=label,label=result.get('label'),status=result['status'],details=result.get('details')))
    blank=recognizer.image_result(np.zeros((480,640,3),dtype='uint8'));recognizer.close()
    report['raw_photo_checks']=raw
    report['blank_image_check']=blank
    test=state['evaluation']['test']
    checks=dict(held_out_accuracy=test['accuracy']>=.95,
                held_out_hkr=all(test['per_class'][l]['recall']>=.95 for l in ['H','K','R']),
                h_teaching_reference=all(r['label']==r['expected'] for r in raw if r['expected']=='H'),
                ordinary_photo_samples=all(r['label']==r['expected'] for r in raw if r['expected'] in ['Y','X']),
                raw_k_r=all(sum(r['label']==l for r in raw if r['expected']==l)/20>=.9 for l in ['K','R']),
                blank_rejected=blank['label'] is None,
                beats_original=report['paired_common_22']['skeleton_candidate']>report['paired_common_22']['old_yolo'])
    report['release_checks']=checks
    target=args.checkpoint.parent/'release_evaluation.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Comparisons',report['paired_common_22'],flush=True)
    print('Raw photo outcomes',[(r['expected'],r['label']) for r in raw[:3]],flush=True)
    print('Raw K/R counts',{l:dict(Counter(r['label'] for r in raw if r['expected']==l)) for l in ['K','R']},flush=True)
    print('Release checks',checks,flush=True)
    if args.approve:
        if not all(checks.values()):raise RuntimeError('Release checks failed; production checkpoint not approved')
        state['approved_for_production']=True;state['release_checks']=checks
        torch.save(state,args.checkpoint)
        print('Checkpoint approved after measured release checks.',flush=True)


if __name__=='__main__':main()
