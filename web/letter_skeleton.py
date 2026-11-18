"""Canonical MediaPipe skeleton input and a small Vietnamese-letter CNN."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from letter_landmarks import LETTERS, LetterNet, features, normalized_points


def merge_predictions(cnn, geometry):
    """Use the reviewed VSL head for strong H/K/R; supplement uncertain poses.

    The older geometry head is never permitted to supply H, K, or R: its
    training corpus contains incompatible handshapes identified in the audit.
    """
    result=cnn.copy()
    for row in range(len(cnn)):
        ci=int(cnn[row].argmax());gi=int(geometry[row].argmax())
        if LETTERS[ci] in ['H','K','R'] and cnn[row,ci]>=.8:
            continue
        if LETTERS[ci] in ['H','K','R']:
            continue
        order=np.argsort(geometry[row])[::-1]
        if LETTERS[gi] not in ['H','K','R'] and geometry[row,gi]>=.9 and geometry[row,gi]-geometry[row,order[1]]>=.25:
            if ci==gi or cnn[row,ci]<.9:
                result[row]=geometry[row]
    return result


def merge_k_specialist(probabilities,k_scores,threshold=.95):
    result=probabilities.copy()
    for row,score in enumerate(k_scores):
        index=int(result[row].argmax())
        if LETTERS[index] in ['H','R'] and result[row,index]>=.8:
            continue
        if score>=threshold:
            result[row]*=(1-score)
            result[row,LETTERS.index('K')]+=score
    return result

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
               (13,17),(17,18),(18,19),(19,20),(0,17)]


def canonical_skeleton(image):
    """Remove drawing-style differences while retaining finger topology/orientation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = (gray > 25).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        raise ValueError('No skeleton found')
    crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    result = np.zeros((64,64), dtype=np.uint8)
    result[4:60,4:60] = cv2.resize(crop, (56,56), interpolation=cv2.INTER_AREA)
    return result


def render_hand(points, width, height):
    """Render the same 21 hand joints used by the reviewed training dataset."""
    xy = np.array([[p.x * width, p.y * height] for p in points], dtype=np.float32)
    if not np.isfinite(xy).all():
        raise ValueError('Invalid landmarks')
    span = np.ptp(xy, axis=0)
    if min(span) < 3:
        raise ValueError('Degenerate hand geometry')
    # Normalize the hand's bounding rectangle; canonical_skeleton does likewise.
    xy = np.round((xy - xy.min(0)) / span * 190 + 16).astype(int)
    image = np.zeros((224,224), dtype=np.uint8)
    for a,b in CONNECTIONS:
        cv2.line(image, tuple(xy[a]), tuple(xy[b]), 255, 2)
    for p in xy:
        cv2.circle(image, tuple(p), 4, 100, -1)
    return canonical_skeleton(image)


class SkeletonNet(nn.Module):
    def __init__(self, classes=23):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,12,3,padding=1), nn.BatchNorm2d(12), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12,24,3,padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48*8*8,128), nn.ReLU(),
                                  nn.Dropout(.2), nn.Linear(128,classes))

    def forward(self, value):
        return self.head(self.conv(value))


class KSpecialist(nn.Module):
    """Learn the national K geometry from reviewed positive/negative examples."""
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(498,96),nn.LayerNorm(96),nn.GELU(),nn.Dropout(.15),
                                  nn.Linear(96,32),nn.GELU(),nn.Linear(32,2))

    def forward(self,value):
        return self.layers(value)


class SkeletonRecognizer:
    """Photo adapter. The shared registry serializes calls to the hand tracker."""
    def __init__(self, path, require_approved=True, allow_demo=False):
        import mediapipe as mp
        state = torch.load(path,map_location='cpu',weights_only=True)
        if state.get('format') != 'vsl-skeleton-v1' or state.get('labels') != LETTERS:
            raise ValueError('Unrecognized letter checkpoint')
        demo_approved=allow_demo and state.get('approved_for_demo') is True and state.get('release_channel')=='demo'
        if require_approved and state.get('approved_for_production') is not True and not demo_approved:
            raise ValueError('Letter checkpoint has not passed release checks')
        self.model=SkeletonNet(len(LETTERS))
        self.model.load_state_dict(state['state_dict'],strict=True)
        self.model.eval()
        self.engine=state.get('engine','vsl-skeleton-v1')
        self.release_channel=state.get('release_channel','production' if state.get('approved_for_production') else 'candidate')
        self.experimental_labels=state.get('experimental_labels',[])
        self.geometry=None
        self.k_specialist=None
        if state.get('geometry_head'):
            head=state['geometry_head']
            self.geometry=LetterNet(len(LETTERS));self.geometry.load_state_dict(head['state_dict'],strict=True);self.geometry.eval()
            self.geometry_mean=np.asarray(head['mean'],dtype='float32');self.geometry_sd=np.asarray(head['sd'],dtype='float32')
        if state.get('k_specialist'):
            head=state['k_specialist']
            self.k_specialist=KSpecialist();self.k_specialist.load_state_dict(head['state_dict'],strict=True);self.k_specialist.eval()
            self.k_mean=np.asarray(head['mean'],dtype='float32');self.k_sd=np.asarray(head['sd'],dtype='float32')
            self.k_threshold=float(head['threshold'])
        self.labels=LETTERS
        self.tracker=mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,
                     model_complexity=1,min_detection_confidence=.25)

    def close(self):
        self.tracker.close()

    def image_result(self,rgb):
        height,width=rgb.shape[:2]
        # Tiny source thumbnails need the same working resolution as webcam
        # inputs. Upsampling changes tracking coverage, not source detail.
        scale=640/max(height,width)
        frame=cv2.resize(rgb,(round(width*scale),round(height*scale)))
        tracked=self.tracker.process(np.ascontiguousarray(frame))
        if not tracked.multi_hand_landmarks:
            return {'status':'no_sign','label':None,
                    'message':'No clear hand found. Keep your whole signing hand in view.'}
        hands=[]
        for landmarks in tracked.multi_hand_landmarks:
            points=landmarks.landmark
            try:skeleton=render_hand(points,frame.shape[1],frame.shape[0])
            except ValueError:continue
            xy=np.array([[p.x,p.y] for p in points]);bounds=np.clip([*xy.min(0),*xy.max(0)],0,1)
            area=float(np.prod(bounds[2:]-bounds[:2]))
            try:geometry=features(normalized_points(points,frame.shape[1],frame.shape[0]))
            except ValueError:continue
            hands.append((area,bounds,skeleton,geometry))
        if not hands:
            return {'status':'no_sign','label':None,'message':'The hand is partly hidden. Show the complete hand and try again.'}
        hands.sort(key=lambda h:h[0],reverse=True)
        tensor=torch.from_numpy(np.array([h[2] for h in hands],dtype='float32')[:,None]/255)
        with torch.inference_mode():
            probabilities=self.model(tensor).softmax(1).numpy()
            if self.geometry is not None:
                gx=(np.array([h[3] for h in hands])-self.geometry_mean)/self.geometry_sd
                gp=self.geometry(torch.from_numpy(gx)).softmax(1).numpy()
                probabilities=merge_predictions(probabilities,gp)
            if self.k_specialist is not None:
                kx=(np.array([h[3] for h in hands])-self.k_mean)/self.k_sd
                kp=self.k_specialist(torch.from_numpy(kx)).softmax(1)[:,1].numpy()
                probabilities=merge_k_specialist(probabilities,kp,self.k_threshold)
        indices=probabilities.argmax(1)
        if len(hands)>1 and hands[1][0]>.65*hands[0][0] and indices[0]!=indices[1]:
            return {'status':'no_sign','label':None,'message':'Show one signing hand so the letter is unambiguous.'}
        index=int(indices[0]);scores=probabilities[0];score=float(scores[index]);order=np.argsort(scores)[::-1]
        label=self.labels[index]
        details={'model':'VSL corrected letter ensemble','score':round(score,4),
                 'boxes':[{'label':label,'score':round(score,4),'bounds':hands[0][1].tolist()}],
                 'alternatives':[{'label':self.labels[int(i)],'score':round(float(scores[i]),4)} for i in order[:3]],
                 'note':'Matches the trained Vietnamese fingerspelling vocabulary. Scores are not calibrated accuracy.'}
        if score<.5:
            return {'status':'no_sign','label':None,'message':'No reliable supported letter found. Match the guide and try again.','details':details}
        uncertain=score<.75 or score-float(scores[order[1]])<.2
        if label in self.experimental_labels:
            return {'status':'uncertain','label':label,
                    'message':'Experimental letter match. H, K and R are still being refined.','details':details}
        return {'status':'uncertain' if uncertain else 'recognized','label':label,
                'message':'Possible match. Check the signing guide to confirm.' if uncertain else 'Sign recognized.',
                'details':details}
