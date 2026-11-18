"""Train a candidate only. Promotion requires separate-source evaluation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from letter_landmarks import LETTERS, LetterNet, features


def metrics(truth, predicted):
    confusion = np.zeros((len(LETTERS), len(LETTERS)), dtype=int)
    np.add.at(confusion, (truth, predicted), 1)
    rows = {}
    for i, label in enumerate(LETTERS):
        tp, actual, called = confusion[i, i], confusion[i].sum(), confusion[:, i].sum()
        recall, precision = float(tp / max(actual, 1)), float(tp / max(called, 1))
        rows[label] = {"support": int(actual), "recall": recall, "precision": precision,
                       "f1": 2 * precision * recall / max(precision + recall, 1e-9)}
    return {"accuracy": float(np.mean(truth == predicted)),
            "macro_f1": float(np.mean([r['f1'] for r in rows.values() if r['support']])),
            "per_class": rows, "confusion": confusion.tolist()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--epochs", type=int, default=180)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261118)
    rng = np.random.default_rng(20261118)
    data = np.load(args.prepared / "features.npz", allow_pickle=False)
    points, labels, splits = data['points'], data['labels'], data['splits']
    original_features = np.asarray([features(p) for p in points])
    train_mask = splits == 'train'
    train_points = points[train_mask]
    augmented = [original_features[train_mask]]
    for _ in range(11):
        changed = train_points.copy()
        changed[:, :, 0] *= rng.choice([-1, 1], size=(len(changed), 1))
        angle = rng.uniform(-.3, .3, size=(len(changed), 1))
        x, y = changed[:, :, 0].copy(), changed[:, :, 1].copy()
        changed[:, :, 0] = np.cos(angle) * x - np.sin(angle) * y
        changed[:, :, 1] = np.sin(angle) * x + np.cos(angle) * y
        for first, second in [(0, 2), (1, 2)]:
            angle = rng.uniform(-.25, .25, size=(len(changed), 1))
            x, y = changed[:, :, first].copy(), changed[:, :, second].copy()
            changed[:, :, first] = np.cos(angle) * x - np.sin(angle) * y
            changed[:, :, second] = np.sin(angle) * x + np.cos(angle) * y
        changed += rng.normal(0, .008, size=changed.shape)
        augmented.append(np.asarray([features(p) for p in changed]))
    x_train = np.concatenate(augmented)
    y_train = np.tile(labels[train_mask], len(augmented))
    mu, sd = x_train.mean(0), np.maximum(x_train.std(0), .05)
    x = torch.from_numpy(((x_train - mu) / sd).astype('float32'))
    y = torch.from_numpy(y_train.astype('int64'))
    vx = torch.from_numpy(((original_features[splits == 'val'] - mu) / sd).astype('float32'))
    vy = torch.from_numpy(labels[splits == 'val'].astype('int64'))
    model = LetterNet(len(LETTERS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.003)
    counts = torch.bincount(y, minlength=len(LETTERS)).clamp_min(1)
    weight = (counts.float().mean() / counts).clamp(.5, 3)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=.025)
    best_loss, best, stale = float('inf'), None, 0
    start = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        for batch in order.split(256):
            optimizer.zero_grad()
            loss = criterion(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(vx)
            val_loss = float(nn.functional.cross_entropy(logits, vy))
            accuracy = float((logits.argmax(1) == vy).float().mean())
        if val_loss < best_loss - .0005:
            best_loss, best, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}: val_accuracy={accuracy:.3f}, val_loss={val_loss:.3f}", flush=True)
        if stale >= 25:
            break
    model.load_state_dict(best)
    model.eval()
    evaluation = {}
    with torch.inference_mode():
        for split in ['val', 'test']:
            mask = splits == split
            sx = torch.from_numpy(((original_features[mask] - mu) / sd).astype('float32'))
            prediction = model(sx).argmax(1).numpy()
            evaluation[split] = metrics(labels[mask], prediction)
    checkpoint = dict(format="vsl-landmarks-v2", input_features=498, state_dict=best, labels=LETTERS,
                      mean=mu.tolist(), sd=sd.tolist(),
                      source="mcphngnga/dataset-vsl@2", source_license="Apache-2.0",
                      evaluation=evaluation, approved_for_production=False)
    target = args.prepared / 'letter_candidate.pt'
    torch.save(checkpoint, target)
    evaluation['training_seconds'] = time.monotonic() - start
    evaluation['production_status'] = 'candidate; external-source evaluation and signing-reference review required'
    (args.prepared / 'evaluation.json').write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({s: {k:v for k,v in evaluation[s].items() if k in ['accuracy','macro_f1']} for s in ['val','test']}), flush=True)
    print('Candidate saved:', target, flush=True)


if __name__ == '__main__':
    main()
