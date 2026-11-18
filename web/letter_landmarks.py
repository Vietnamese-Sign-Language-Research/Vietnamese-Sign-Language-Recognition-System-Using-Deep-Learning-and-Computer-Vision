"""Small CPU classifier for Vietnamese fingerspelling hand landmarks."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

LETTERS = ["A", "B", "C", "D", "Đ", "E", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y"]


def normalized_points(landmarks, width, height):
    points = np.asarray([[p.x * width, p.y * height, p.z * width]
                         for p in landmarks], dtype=np.float32)
    points -= points[0]
    scale = np.linalg.norm(points[9])
    if not np.isfinite(points).all() or scale < 1e-5:
        raise ValueError("Invalid hand geometry")
    return points / scale


def features(points):
    """Retain orientation and depth, alongside scale-independent finger geometry."""
    points = np.asarray(points, dtype=np.float32).reshape(21, 3)
    # Finger-to-finger geometry is less sensitive to camera angle than raw xyz.
    differences = points[:, None, :] - points[None, :, :]
    indices = np.triu_indices(21, 1)
    distances = np.linalg.norm(differences, axis=-1)[indices]
    image_distances = np.linalg.norm(differences[:, :, :2], axis=-1)[indices]
    angles = []
    for chain in [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                  (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]:
        for a, b, c in zip(chain, chain[1:], chain[2:]):
            u, v = points[a] - points[b], points[c] - points[b]
            angles.append(float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8)))
    return np.concatenate([points.flatten(), distances, image_distances, np.asarray(angles)]).astype(np.float32)


class LetterNet(nn.Module):
    def __init__(self, classes=23):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(498, 192), nn.LayerNorm(192), nn.GELU(),
                                    nn.Dropout(.2), nn.Linear(192, 96), nn.GELU(),
                                    nn.Dropout(.1), nn.Linear(96, classes))

    def forward(self, value):
        return self.layers(value)
