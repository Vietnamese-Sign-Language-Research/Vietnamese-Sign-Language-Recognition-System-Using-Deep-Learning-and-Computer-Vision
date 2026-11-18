"""Measure the complete candidate, including raw held-out recordings, before release."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letter_skeleton import LETTERS,SkeletonRecognizer
from recognition import ModelRegistry

ROOT=Path('artifacts/alphabet-audit')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--approve',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    # The older geometry model supplements non-conflicting letters only.
    # H/K/R are excluded because their labels do not match the national chart.
    donor=torch.load(ROOT/'prepared/letter_candidate.pt',map_location='cpu',weights_only=True)
    state['geometry_head']={k:donor[k] for k in ['state_dict','labels','mean','sd','format','source']}
    state['geometry_head']['excluded_outputs']=['H','K','R']
    specialist=ROOT/'vsl-corrected/k-specialist/k_specialist.pt'
    if specialist.is_file():state['k_specialist']=torch.load(specialist,map_location='cpu',weights_only=True)
    state['approved_for_production']=False
    state['engine']='vsl-corrected-ensemble-v1'
    state['source']='national-chart VSL skeletons + curated raw K + non-HKR geometry/photo fallback'
    state['curation_manifest_sha256']=hashlib.sha256((args.checkpoint.parent/'manifest.json').read_bytes()).hexdigest()
    state['target_reference']='Circular 17/2020/TT-BGDDT, national fingerspelling chart, gazette page 51'
    torch.save(state,args.checkpoint)
    recognizer=SkeletonRecognizer(args.checkpoint,require_approved=False)
    registry=ModelRegistry();registry.letters=recognizer
    from ultralytics import YOLO
    registry.detectors['letters']=YOLO('weights/alphabet_model.pt')
    def predict(image):return registry.image_result(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),'letters')
    ordinary=[]
    for name,label in [('sample-1.jpg','Y'),('sample-2.jpg','X')]:
        r=predict(cv2.imread('assets/samples/'+name))
        ordinary.append(dict(path=name,expected=label,label=r.get('label'),status=r['status']))
    reference_photos=[]
    reference_path=ROOT/'vsl-corrected/user-reference/h-reference.jpg'
    if reference_path.is_file():
        r=predict(cv2.imread(str(reference_path)))
        reference_photos.append(dict(path=str(reference_path),expected='H',label=r.get('label'),status=r['status'],
            provenance='User-supplied H reference, reserved for testing; original teaching source not yet supplied'))
    # Preserve the original separate-source diagnostic labels without pretending
    # that its disputed regional H labels are a certified national benchmark.
    prior=json.loads((ROOT/'prepared/external_evaluation.json').read_text(encoding='utf-8'))
    paired=[]
    with zipfile.ZipFile(ROOT/'external/source.zip') as z:
        for i,record in enumerate(prior['predictions']):
            image=cv2.imdecode(np.frombuffer(z.read(record['path']),np.uint8),cv2.IMREAD_COLOR)
            r=predict(image)
            paired.append(dict(path=record['path'],expected=record['actual'],label=r.get('label'),
                               status=r['status'],legacy=record['legacy']))
            if i%100==0:print('Paired raw-photo check',i,'/',len(prior['predictions']),flush=True)
    # Native recordings were never used by the training split for these people
    # or clips. Score every planned middle-third frame, including tracking loss.
    manifest=json.loads((args.checkpoint.parent/'manifest.json').read_text(encoding='utf-8'))
    native_sources={}
    for r in manifest['records']:
        if r.get('source') in ['hauuto','ngannguyen65'] and r['split']=='test':
            native_sources[(r['source'],r['path'])]=(r['label'],r['person'])
    native=[]
    with tempfile.TemporaryDirectory(dir=ROOT/'vsl-corrected') as folder,zipfile.ZipFile(ROOT/'vsl-corrected/hauuto-v2.zip') as z:
        for (source,name),(label,person) in native_sources.items():
            path=Path(name) if source=='ngannguyen65' else Path(folder)/Path(name).name
            if source=='hauuto':path.write_bytes(z.read(name))
            cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_number in np.linspace(round(n*.3),round(n*.7),12).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_number));ok,image=cap.read()
                if not ok:raise RuntimeError('Failed to decode held-out recording')
                r=predict(image)
                native.append(dict(source=source,path=name,frame=int(frame_number),person=person,
                                    expected=label,label=r.get('label'),status=r['status']))
            cap.release()
    blank=predict(np.zeros((480,640,3),np.uint8));registry.close()
    common=[r for r in paired if r['expected']!='Đ']
    by_class={l:dict(count=sum(r['expected']==l for r in paired),
                    correct=sum(r['expected']==l and r['label']==l for r in paired)) for l in LETTERS}
    native_rates={l:sum(r['label']==l for r in native if r['expected']==l)/sum(r['expected']==l for r in native) for l in ['H','K'] if any(r['expected']==l for r in native)}
    native_recognized_rates={l:sum(r['label']==l and r['status']=='recognized' for r in native if r['expected']==l)/sum(r['expected']==l for r in native) for l in native_rates}
    r_rate=by_class['R']['correct']/by_class['R']['count']
    checks=dict(held_out_overall=state['evaluation']['test']['accuracy']>=.95,
                held_out_hkr=all(state['evaluation']['test']['per_class'][l]['recall']>=.95 for l in ['H','K','R']),
                # Absence of verified raw H must fail, not become a vacuous pass.
                raw_h=native_rates.get('H',0)>=.9,raw_k=native_rates.get('K',0)>=.9,
                external_raw_k=by_class['K']['correct']/by_class['K']['count']>=.9,raw_r=r_rate>=.9,
                ordinary_samples=all(r['expected']==r['label'] for r in ordinary),blank_rejected=blank['label'] is None,
                paired_beats_original=sum(r['label']==r['expected'] for r in common)>sum(r['legacy']==r['expected'] for r in common))
    if reference_photos:checks['user_h_reference']=all(r['expected']==r['label'] and r['status']=='recognized' for r in reference_photos)
    if state.get('k_specialist'):
        ke=state['k_specialist']['evaluation']['test']
        kv=state['k_specialist']['evaluation']['val']
        checks.update(k_specialist_recall=min(ke['k_recall'],kv['k_recall'])>=.9,
                      k_specialist_false_positives=max(ke['false_positive_rate'],kv['false_positive_rate'])<=.01)
    report=dict(checkpoint=str(args.checkpoint),checks=checks,
        held_out_component=state['evaluation']['test'],native_recording_rates=native_rates,
        native_fully_recognized_rates=native_recognized_rates,
        missing_evaluation=['Real-person national-chart H (conflicting supplemental H clips excluded)'] if 'H' not in native_rates else [],
        k_specialist_evaluation=state.get('k_specialist',{}).get('evaluation'),
        native_recording_frames=native,raw_r_rate=r_rate,ordinary_samples=ordinary,user_reference_photos=reference_photos,
        paired_common_22=dict(candidate=sum(r['label']==r['expected'] for r in common)/len(common),
                              original=sum(r['legacy']==r['expected'] for r in common)/len(common),count=len(common)),
        paired_per_class=by_class,paired_photos=paired,
        limitations='Small correlated video-frame samples; signer overlap with the large skeleton corpus is unknown. The external H labels do not match the selected national-chart H and are preserved unchanged, so aggregate accuracy is only a mixed-label diagnostic. These diagnostics informed revisions and are not an untouched test or a population accuracy guarantee. Label-correct rates include uncertain suggestions; fully recognized rates are reported separately.')
    (args.checkpoint.parent/'final_release_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Ordinary samples',ordinary,flush=True)
    print('Native H/K',native_rates,'raw R',r_rate,flush=True)
    print('Paired common 22',report['paired_common_22'],flush=True)
    print('Release checks',checks,flush=True)
    if args.approve:
        if not all(checks.values()):raise RuntimeError('Candidate failed release checks; website must remain unchanged')
        state['approved_for_production']=True;state['release_checks']=checks
        torch.save(state,args.checkpoint)
        print('Approved checkpoint SHA256',hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':main()
