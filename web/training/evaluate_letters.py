"""Compare a candidate with the existing detector on a separate source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from letter_landmarks import LETTERS, LetterNet, features
from train_letters import metrics


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('external', type=Path)
    parser.add_argument('--legacy', type=Path, default=Path('weights/alphabet_model.pt'))
    parser.add_argument('--baseline-report', type=Path, help='Reuse predictions for this exact ordered external image set')
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = torch.load(args.candidate, map_location='cpu', weights_only=True)
    assert state['labels'] == LETTERS
    model = LetterNet(len(LETTERS))
    model.load_state_dict(state['state_dict'])
    model.eval()
    data = np.load(args.external/'features.npz', allow_pickle=False)
    raw = np.asarray([features(p) for p in data['points']])
    normalized = (raw - state['mean']) / state['sd']
    x = torch.tensor(normalized, dtype=torch.float32)
    with torch.inference_mode():
        probabilities = model(x).softmax(1).numpy()
    predictions = probabilities.argmax(1)
    report = dict(candidate=metrics(data['labels'], predictions))
    manifest = json.loads((args.external/'manifest.json').read_text(encoding='utf-8'))
    records = manifest['records']
    report['source'] = manifest['source']
    report['warning'] = 'Separate source, but signer independence is unverified; adjacent video frames are correlated.'
    report['hand_tracking'] = dict(usable=len(records), failed=sum(manifest['skipped_no_hand'].values()),
                                   failed_by_class=manifest['skipped_no_hand'])
    legacy_predictions = []
    if args.baseline_report:
        cached = json.loads(args.baseline_report.read_text(encoding='utf-8'))['predictions']
        assert [(r['path'],r['label']) for r in records] == [(r['path'],r['actual']) for r in cached]
        legacy_predictions = [LETTERS.index(r['legacy']) if r['legacy'] in LETTERS else -1 for r in cached]
    else:
        os.environ['YOLO_CONFIG_DIR'] = str(Path('.cache/ultralytics').resolve())
        from ultralytics import YOLO
        detector = YOLO(str(args.legacy))
        with zipfile.ZipFile(args.external.parent/'source.zip') as archive:
            for i, record in enumerate(records):
                image = cv2.imdecode(np.frombuffer(archive.read(record['path']), np.uint8), cv2.IMREAD_COLOR)
                result = detector.predict(image, imgsz=640, conf=.35, verbose=False, device='cpu')[0]
                name = None
                if result.boxes is not None and len(result.boxes):
                    best = int(result.boxes.conf.argmax())
                    name = result.names[int(result.boxes.cls[best])]
                legacy_predictions.append(LETTERS.index(name) if name in LETTERS else -1)
                if i % 100 == 0:
                    print(f'Baseline evaluation {i}/{len(records)}', flush=True)
    legacy_predictions = np.asarray(legacy_predictions)
    report['legacy'] = {'accuracy':float(np.mean(legacy_predictions == data['labels'])),
                        'recognized_fraction':float(np.mean(legacy_predictions >= 0)),
                        'h_recall':float(np.mean(legacy_predictions[data['labels']==LETTERS.index('H')] == LETTERS.index('H')))}
    common = data['labels'] != LETTERS.index('Đ')
    report['common_22_letters'] = {'candidate_accuracy':float(np.mean(predictions[common] == data['labels'][common])),
                                    'legacy_accuracy':float(np.mean(legacy_predictions[common] == data['labels'][common]))}
    report['predictions'] = [dict(path=r['path'], actual=r['label'],
                            candidate=LETTERS[int(p)], confidence=float(probabilities[i,p]),
                            legacy=LETTERS[int(l)] if l>=0 else None)
                            for i,(r,p,l) in enumerate(zip(records,predictions,legacy_predictions))]
    target = args.candidate.parent/'external_evaluation.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'candidate_accuracy':report['candidate']['accuracy'],
                     'candidate_h':report['candidate']['per_class']['H'],
                     'legacy':report['legacy'],'hand_tracking':report['hand_tracking']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
