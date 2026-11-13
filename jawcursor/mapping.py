"""Gaze features -> screen position, learned from a short calibration.

Ridge regression, with the amount of regularisation and the choice between a
linear and a slightly curved model picked by leaving out one calibration dot
at a time. That same leave-one-out error is the accuracy estimate shown to
the user, so it is measured on dots the model did not see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gaze import FEATURE_FLOORS, FEATURE_NAMES, N_EYE_FEATURES

LAMBDAS = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
KINDS = ("linear", "quadratic")


def _expand(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "quadratic":
        return np.concatenate([z, z[..., :N_EYE_FEATURES] ** 2], axis=-1)
    return z


def _ridge(z: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    z_mean, y_mean = z.mean(axis=0), y.mean(axis=0)
    zc, yc = z - z_mean, y - y_mean
    n, d = zc.shape
    weights = np.linalg.solve(zc.T @ zc / n + lam * np.eye(d), zc.T @ yc / n)
    return weights, y_mean - z_mean @ weights


@dataclass
class FitReport:
    kind: str
    lam: float
    median_error: float     # fraction of screen width, on left-out dots
    p90_error: float
    points: int
    samples: int

    def pixels(self, screen_width: int) -> tuple[int, int]:
        return round(self.median_error * screen_width), round(self.p90_error * screen_width)


class GazeMapper:
    def __init__(self, mu, sd, weights, bias, kind: str, aspect: float):
        self.mu, self.sd = np.asarray(mu, float), np.asarray(sd, float)
        self.weights, self.bias = np.asarray(weights, float), np.asarray(bias, float)
        self.kind, self.aspect = kind, aspect

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        z = _expand((np.asarray(features, float) - self.mu) / self.sd, self.kind)
        x, y = z @ self.weights + self.bias
        return float(min(max(x, 0.0), 1.0)), float(min(max(y, 0.0), 1.0))

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, groups: np.ndarray,
            aspect: float = 16 / 9) -> tuple["GazeMapper", FitReport]:
        """`targets` are (x, y) in 0..1; `groups` says which dot each row came from."""
        x = np.asarray(features, float)
        y = np.asarray(targets, float)
        groups = np.asarray(groups)
        ids = np.unique(groups)
        if len(ids) < 5:
            raise ValueError("calibration needs at least 5 dots with usable samples")
        mu = x.mean(axis=0)
        sd = np.maximum(x.std(axis=0), np.asarray(FEATURE_FLOORS))
        z = (x - mu) / sd

        def errors(kind: str, lam: float) -> np.ndarray:
            ze = _expand(z, kind)
            out = np.empty(len(y))
            for g in ids:
                held = groups == g
                w, b = _ridge(ze[~held], y[~held], lam)
                diff = ze[held] @ w + b - y[held]
                out[held] = np.hypot(diff[:, 0], diff[:, 1] / aspect)   # in screen widths
            return out

        scored = [(float(np.median(errors(kind, lam))), kind, lam)
                  for kind in KINDS for lam in LAMBDAS]
        _, kind, lam = min(scored)
        err = errors(kind, lam)
        weights, bias = _ridge(_expand(z, kind), y, lam)
        report = FitReport(kind=kind, lam=lam, median_error=float(np.median(err)),
                           p90_error=float(np.percentile(err, 90)),
                           points=len(ids), samples=len(y))
        return cls(mu, sd, weights, bias, kind, aspect), report

    # -- persistence -------------------------------------------------------
    def save(self, path: Path, report: FitReport | None = None, screen=None) -> None:
        path.write_text(json.dumps({
            "features": FEATURE_NAMES, "kind": self.kind, "aspect": self.aspect,
            "mu": self.mu.tolist(), "sd": self.sd.tolist(),
            "weights": self.weights.tolist(), "bias": self.bias.tolist(),
            "report": report.__dict__ if report else None,
            "screen": list(screen) if screen else None,
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GazeMapper":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("features") != FEATURE_NAMES:
            raise ValueError("gaze calibration is from another version; run "
                             "`jawcursor calibrate gaze` again")
        return cls(data["mu"], data["sd"], data["weights"], data["bias"],
                   data["kind"], data["aspect"])


def clean_dot_samples(samples: np.ndarray, limit: float = 4.0) -> np.ndarray:
    """Drops frames taken mid-saccade or mid-blink from one dot's samples."""
    if len(samples) < 5:
        return samples
    eye = samples[:, :N_EYE_FEATURES]
    median = np.median(eye, axis=0)
    spread = 1.4826 * np.median(np.abs(eye - median), axis=0)
    spread = np.maximum(spread, np.asarray(FEATURE_FLOORS[:N_EYE_FEATURES]) / 4.0)
    keep = (np.abs(eye - median) / spread).max(axis=1) <= limit
    return samples[keep]
