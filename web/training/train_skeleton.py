"""Train only on reviewed VSL-v2 skeletons; hold out complete filename blocks."""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import zipfile

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letter_skeleton import LETTERS, SkeletonNet, canonical_skeleton
from train_letters import metrics


def prepare(archive, output):
    rng = random.Random(20261119)
    by_class = defaultdict(list)
    images, records, seen = [], [], {}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if Path(name).suffix.lower() in {'.jpg','.jpeg','.png'}:
                label = Path(name).parts[-2]
                if label in LETTERS:
                    by_class[label].append(name)
        for label in LETTERS:
            groups = defaultdict(list)
            for name in sorted(by_class[label]):
                number = int(re.findall(r'\d+',Path(name).stem)[-1])
                groups[number//250].append(name)
            keys = sorted(groups)
            rng.shuffle(keys)
            if len(keys)<5:raise ValueError('Insufficient filename groups for '+label)
            allocation = {'test':keys[:2], 'val':keys[2:4], 'train':keys[4:]}
            for split, chosen in allocation.items():
                names = [n for k in chosen for n in groups[k]]
                rng.shuffle(names)
                limit = 500 if split=='train' else 120
                for name in names:
                    im=cv2.imdecode(np.frombuffer(z.read(name),np.uint8),cv2.IMREAD_GRAYSCALE)
                    im=canonical_skeleton(im)
                    digest=hashlib.sha256(im.tobytes()).hexdigest()
                    if digest in seen:
                        if seen[digest]!=label:raise ValueError('Cross-label duplicate '+label)
                        continue
                    seen[digest]=label
                    number=int(re.findall(r'\d+',Path(name).stem)[-1])
                    records.append(dict(path=name,label=label,split=split,group=f'{label}/{number//250}',sha256=digest))
                    images.append(im)
                    limit-=1
                    if limit==0:break
            print('Prepared',label,flush=True)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'skeletons.npz',images=np.array(images),
                         labels=np.array([LETTERS.index(r['label']) for r in records]),
                         splits=np.array([r['split'] for r in records]))
    manifest=dict(source='https://www.kaggle.com/datasets/cuongnk9104/vsl-v2',version=1,
                  warning='Filename-block split; signer IDs are not supplied. Not a signer-independent benchmark.',
                  source_license='Unspecified by publisher',records=records)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared counts',dict(Counter(r['split'] for r in records)),flush=True)


def augment(x,rotation=.28):
    n=len(x)
    angle=(torch.rand(n)-.5)*rotation
    scale=torch.rand(n)*.15+.9
    flip=torch.where(torch.rand(n)>.5,1.,-1.)
    theta=torch.zeros(n,2,3)
    theta[:,0,0]=angle.cos()*scale*flip
    theta[:,0,1]=-angle.sin()*scale
    theta[:,1,0]=angle.sin()*scale*flip
    theta[:,1,1]=angle.cos()*scale
    theta[:,:,2]=(torch.rand(n,2)-.5)*.08
    grid=F.affine_grid(theta,x.size(),align_corners=False)
    return F.grid_sample(x,grid,align_corners=False)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--init',type=Path)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--rotation',type=float,default=.28)
    p.add_argument('--focus-hkr',action='store_true')
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(20261119)
    if not (args.output/'skeletons.npz').exists():prepare(args.archive,args.output)
    d=np.load(args.output/'skeletons.npz',allow_pickle=False)
    xs=torch.from_numpy(d['images'].astype('float32')[:,None]/255)
    ys=torch.from_numpy(d['labels'].astype('int64'))
    tx,ty=xs[d['splits']=='train'],ys[d['splits']=='train']
    vx,vy=xs[d['splits']=='val'],ys[d['splits']=='val']
    model=SkeletonNet()
    if args.init:
        initial=torch.load(args.init,map_location='cpu',weights_only=True)
        model.load_state_dict(initial['state_dict'],strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.002)
    class_weight=torch.ones(len(LETTERS))
    if args.focus_hkr:
        for label,weight in [('H',1.5),('K',3.),('R',1.5)]:class_weight[LETTERS.index(label)]=weight
    best,best_loss,stale=None,float('inf'),0
    start=time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(tx)).split(192):
            opt.zero_grad();loss=F.cross_entropy(model(augment(tx[indices],args.rotation)),ty[indices],weight=class_weight,label_smoothing=.02)
            loss.backward();opt.step()
        model.eval()
        with torch.inference_mode():
            logits=torch.cat([model(b) for b in vx.split(256)])
            loss=float(F.cross_entropy(logits,vy,weight=class_weight));acc=float((logits.argmax(1)==vy).float().mean())
        print(f'Epoch {epoch+1}: val_accuracy={acc:.4f}, loss={loss:.4f}, elapsed={time.monotonic()-start:.0f}s',flush=True)
        if loss<best_loss-.001:best,best_loss,stale=copy.deepcopy(model.state_dict()),loss,0
        else:stale+=1
        if stale>=8:break
    model.load_state_dict(best);model.eval();report={}
    with torch.inference_mode():
        for split in ['val','test']:
            mask=d['splits']==split
            predictions=torch.cat([model(b) for b in xs[mask].split(256)]).argmax(1).numpy()
            report[split]=metrics(d['labels'][mask],predictions)
    manifest=json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    sources=sorted({'cuongnk9104/vsl-v2@1'} | {r['source'] for r in manifest['records'] if r.get('source')})
    state=dict(format='vsl-skeleton-v1',labels=LETTERS,state_dict=best,
               source=' + '.join(sources),source_license='See source manifest; not all publishers specify a license',
               training_parameters=dict(seed=20261119,epochs_requested=args.epochs,learning_rate=args.learning_rate,rotation=args.rotation,focus_hkr=args.focus_hkr),
               approved_for_production=False,evaluation=report)
    torch.save(state,args.output/'letter_skeleton.pt')
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Test accuracy',report['test']['accuracy'],flush=True)
    print('H K R recalls',{k:report['test']['per_class'][k] for k in ['H','K','R']},flush=True)


if __name__=='__main__':main()
