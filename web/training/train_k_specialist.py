"""Train a national-form K specialist from reviewed raw recordings only.

Other poses, including the incompatible old K form, are negative examples.
The external diagnostic photographs are never used for training or selection.
"""
from pathlib import Path
import copy
import json
import sys
import tempfile
import zipfile

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letter_landmarks import LETTERS,features,normalized_points
from letter_skeleton import KSpecialist

ROOT=Path('artifacts/alphabet-audit')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.manual_seed(20261121)
    out=ROOT/'vsl-corrected/k-specialist';out.mkdir(exist_ok=True)
    old=np.load(ROOT/'prepared/features.npz',allow_pickle=False)
    # None of the old corpus's K images uses the target national form.
    points=list(old['points']);truth=[0]*len(points);splits=list(old['splits'])
    manifest=json.loads((ROOT/'vsl-corrected/skeleton-v4/manifest.json').read_text(encoding='utf-8'))
    jobs={(r['source'],r['path']):r for r in manifest['records'] if r.get('source') in ['hauuto','ngannguyen65'] and r['label']=='K'}
    rows=[]
    with tempfile.TemporaryDirectory(dir=out) as folder,zipfile.ZipFile(ROOT/'vsl-corrected/hauuto-v2.zip') as z,mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1,model_complexity=1,min_detection_confidence=.3) as tracker:
        for (source,name),r in jobs.items():
            p=Path(name) if source=='ngannguyen65' else Path(folder)/Path(name).name
            if source=='hauuto':p.write_bytes(z.read(name))
            cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame in np.linspace(round(n*.3),round(n*.7),12).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,image=cap.read()
                if not ok:continue
                h=tracker.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                if not h.multi_hand_landmarks:continue
                points.append(normalized_points(h.multi_hand_landmarks[0].landmark,image.shape[1],image.shape[0]));truth.append(1);splits.append(r['split'])
                rows.append(dict(r,frame=int(frame)))
            cap.release()
    points=np.array(points);truth=np.array(truth);splits=np.array(splits)
    rng=np.random.default_rng(20261121);xx=[];yy=[]
    for label,copies in [(0,2),(1,30)]:
        base=points[(splits=='train')&(truth==label)]
        for _ in range(copies):
            changed=base.copy();changed[:,:,0]*=rng.choice([-1,1],size=(len(changed),1))
            for first,second,amount in [(0,2,.55),(1,2,.4),(0,1,.5)]:
                a=rng.uniform(-amount,amount,size=(len(changed),1));x,y=changed[:,:,first].copy(),changed[:,:,second].copy()
                changed[:,:,first]=np.cos(a)*x-np.sin(a)*y;changed[:,:,second]=np.sin(a)*x+np.cos(a)*y
            changed+=rng.normal(0,.006,changed.shape)
            xx.extend(features(p) for p in changed);yy.extend([label]*len(changed))
    xx=np.array(xx);yy=np.array(yy);mean=xx.mean(0);sd=np.maximum(xx.std(0),.05)
    x=torch.from_numpy(((xx-mean)/sd).astype('float32'));y=torch.from_numpy(yy.astype('int64'))
    allx=torch.from_numpy(((np.array([features(p) for p in points])-mean)/sd).astype('float32'))
    vx=allx[splits=='val'];vy=torch.from_numpy(truth[splits=='val'])
    model=KSpecialist();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.005)
    best=None;loss_best=float('inf');stale=0
    for epoch in range(100):
        model.train()
        for batch in torch.randperm(len(x)).split(256):
            opt.zero_grad();loss=nn.functional.cross_entropy(model(x[batch]),y[batch],label_smoothing=.02);loss.backward();opt.step()
        model.eval()
        with torch.inference_mode():
            logits=model(vx);loss=float(nn.functional.cross_entropy(logits,vy))
        if loss<loss_best-.0002:best=copy.deepcopy(model.state_dict());loss_best=loss;stale=0
        else:stale+=1
        if epoch%10==0:print('Epoch',epoch+1,'validation loss',loss,flush=True)
        if stale>=18:break
    model.load_state_dict(best);model.eval()
    # Select the operating point on validation negatives, not on the external
    # photos or held-out test people. Softmax scores are not calibrated accuracy.
    with torch.inference_mode():vp=model(vx).softmax(1)[:,1].numpy()
    actual=truth[splits=='val'];options=[]
    for threshold in np.linspace(.5,.995,496):
        if np.mean(vp[actual==0]>=threshold)<=.005:
            options.append((float(np.mean(vp[actual==1]>=threshold)),float(threshold)))
    if not options:raise RuntimeError('No acceptable K specialist validation operating point')
    _,threshold=max(options)
    report={}
    with torch.inference_mode():
        for split in ['val','test']:
            mask=splits==split;s=model(allx[mask]).softmax(1)[:,1].numpy();actual=truth[mask]
            report[split]=dict(positive_count=int(sum(actual==1)),negative_count=int(sum(actual==0)),
                k_recall=float(np.mean(s[actual==1]>=threshold)),false_positive_rate=float(np.mean(s[actual==0]>=threshold)))
    state=dict(state_dict=best,mean=mean.tolist(),sd=sd.tolist(),threshold=threshold,threshold_selection='Maximize validation K recall subject to <=0.5% validation false positives; choose highest threshold among ties.',source='Reviewed Hauuto/Ngan national K clips; other old poses are negatives',evaluation=report)
    torch.save(state,out/'k_specialist.pt')
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'positive_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('K specialist threshold',threshold,'evaluation',report,flush=True)
