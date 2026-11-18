"""Extract a reproducible, deduplicated sample from the public DATASET_VSL zip.

The filename block split is NOT signer-independent: signer IDs are not supplied.
An external-source evaluation is required before promoting any checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from letter_landmarks import LETTERS, normalized_points

ARCHIVE = None
TRACKER = None


def initialize(path):
    global ARCHIVE, TRACKER
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(1)
    ARCHIVE = zipfile.ZipFile(path)
    TRACKER = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                      model_complexity=1, min_detection_confidence=.5)


def extract(item):
    import cv2
    name, label, split, group = item
    content = ARCHIVE.read(name)
    digest = hashlib.sha256(content).hexdigest()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return name, label, split, group, digest, None
    height, width = image.shape[:2]
    ratio = min(1, 640 / max(height, width))
    image = cv2.resize(image, (round(width * ratio), round(height * ratio)))
    results = TRACKER.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return name, label, split, group, digest, None
    try:
        points = normalized_points(results.multi_hand_landmarks[0].landmark,
                                   image.shape[1], image.shape[0])
    except ValueError:
        points = None
    return name, label, split, group, digest, points


def assign_group_splits(records):
    """Every class gets validation/test groups; a group cannot cross splits."""
    groups = defaultdict(set)
    for record in records:
        groups[record['label']].add(record['group'])
    assignments = {}
    rng = random.Random(20261118)
    for label, keys in sorted(groups.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        if len(keys) < 3:
            raise ValueError(f"{label} needs at least three distinct filename groups")
        count = max(1, round(len(keys) * .15))
        for i, group in enumerate(keys):
            assignments[group] = 'val' if i < count else 'test' if i < count * 2 else 'train'
    for record in records:
        record['split'] = assignments[record['group']]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--external", action="store_true", help="Evaluation-only archive; never use for training")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261118)
    groups = defaultdict(list)
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            raw_label = Path(name).parts[-2]
            label = "Đ" if raw_label.lower() == "dd" else raw_label
            if label in LETTERS:
                groups[label].append(name)
    items = []
    for label in LETTERS:
        names = sorted(groups[label])
        rng.shuffle(names)
        for name in names[:args.per_class]:
            stem = Path(name).stem
            prefix = re.sub(r"[\d\W_]+", "", stem) or "numeric"
            numbers = re.findall(r"\d+", stem)
            block = int(numbers[-1]) // 50 if numbers else 0
            group = f"{label}/{prefix}/{block}"
            n = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
            split = "test" if args.external or n == 0 else "val" if n == 1 else "train"
            items.append((name, label, split, group))
    rng.shuffle(items)
    records, points, seen = [], [], {}
    skipped, conflicts = Counter(), []
    with ProcessPoolExecutor(args.workers, initializer=initialize,
                             initargs=(str(args.archive.resolve()),)) as pool:
        for index, result in enumerate(pool.map(extract, items, chunksize=20), 1):
            name, label, split, group, digest, coordinates = result
            if coordinates is None:
                skipped[label] += 1
            elif digest in seen:
                if seen[digest] != label:
                    conflicts.append({"sha256": digest, "labels": [seen[digest], label]})
            else:
                seen[digest] = label
                records.append(dict(path=name, label=label, split=split, group=group,
                                    sha256=digest))
                points.append(coordinates)
            if index % 200 == 0:
                print(f"Extracted {index}/{len(items)}; usable={len(points)}", flush=True)
    if conflicts:
        raise RuntimeError(f"Conflicting labels for identical files: {conflicts[:3]}")
    if not args.external:
        assign_group_splits(records)
    np.savez_compressed(args.output / "features.npz", points=np.asarray(points),
                        labels=np.asarray([LETTERS.index(r['label']) for r in records]),
                        splits=np.asarray([r['split'] for r in records]))
    metadata = dict(source="https://github.com/TuanKiet1774/SignLanguage_Vietnamese" if args.external else "https://www.kaggle.com/datasets/mcphngnga/dataset-vsl",
                    version="2e44477b2939f14be2fbfd222191af9e617448f4" if args.external else 2,
                    license="not specified; evaluation only" if args.external else "Apache-2.0", labels=LETTERS, records=records,
                    skipped_no_hand=dict(skipped),
                    split_warning="Filename blocks, not signer-independent. External evaluation required.")
    (args.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"usable": len(points), "split_counts": dict(Counter(r['split'] for r in records)),
                      "skipped_no_hand": dict(skipped)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
