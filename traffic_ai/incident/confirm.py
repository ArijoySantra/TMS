"""Stage 2: confirm a candidate with a video model to cut false alarms.

VideoMAEConfirmer uses a VideoMAE (V2) classifier fine-tuned on accident / normal clips
(see notebooks/02_train_accident_videomae.ipynb). Without a checkpoint, the heuristic
confirmer checks that the scene stays disturbed after the event.
"""
from __future__ import annotations

import numpy as np


class VideoMAEConfirmer:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, VideoMAEForVideoClassification
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = VideoMAEForVideoClassification.from_pretrained(checkpoint).to(device).eval()
        self.device = device
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.pos = labels.get("accident", 1)
        self.n_frames = self.model.config.num_frames

    def __call__(self, frames: list[np.ndarray], candidate=None) -> float:
        """frames: list of RGB frames around the event. Returns P(accident)."""
        idx = np.linspace(0, len(frames) - 1, self.n_frames).round().astype(int)
        inputs = self.proc([frames[i] for i in idx], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        return float(logits.softmax(-1)[0, self.pos])


class HeuristicConfirmer:
    """Confirms when the flagged vehicles stay (nearly) stationary after the event,
    which separates real collisions from brief occlusions and box jitter."""

    def __init__(self, estimator, settle_s: float = 3.0):
        self.est, self.settle = estimator, settle_s

    def __call__(self, frames, candidate) -> float:
        states = [self.est.tracks.get(i) for i in candidate.track_ids]
        states = [s for s in states if s is not None]
        if not states:
            return 0.3  # vehicles vanished (e.g. occluded); leave it to the operator
        still = np.mean([s.speed < 1.0 for s in states])
        return float(0.4 + 0.5 * still + 0.1 * min(1.0, len(candidate.cues) / 2))
