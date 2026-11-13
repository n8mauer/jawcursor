"""Webcam -> face landmarks -> a small feature vector that follows the gaze."""

from __future__ import annotations

import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CameraParams, home_dir

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# MediaPipe face mesh indices. "Right" and "left" are the subject's own.
R_CORNERS, R_LIDS, R_IRIS = (33, 133), (159, 145), 468
L_CORNERS, L_LIDS, L_IRIS = (362, 263), (386, 374), 473

FEATURE_NAMES = [
    "r_iris_h", "r_iris_v", "l_iris_h", "l_iris_v",     # iris position in the eye
    "look_h", "look_v",                                 # model's eye-look scores
    "r_open", "l_open",                                 # lid opening (tracks looking down)
    "yaw", "pitch", "roll",                             # head rotation, rad
    "tx", "ty", "tz",                                   # head position, decimetres
]
# Smallest spread each feature is assumed to have. Stops a feature that barely
# moved during calibration from being blown up into noise.
FEATURE_FLOORS = [0.01] * 4 + [0.03] * 2 + [0.02] * 2 + [0.02] * 3 + [0.05] * 3
N_EYE_FEATURES = 6          # the leading features that get squared terms


@dataclass
class GazeFeatures:
    vector: np.ndarray                      # len(FEATURE_NAMES)
    blink: float                            # 0 open .. 1 closed
    iris_px: tuple[tuple[float, float], tuple[float, float]]


def _eye(points: np.ndarray, corners, lids, iris) -> tuple[float, float, float]:
    a, b = points[corners[0]], points[corners[1]]
    width = float(np.linalg.norm(b - a))
    if width < 1e-6:
        return 0.0, 0.0, 0.0
    along = (b - a) / width                             # image-left -> image-right
    down = np.array([-along[1], along[0]])
    offset = points[iris] - (a + b) / 2.0
    opening = float(np.linalg.norm(points[lids[0]] - points[lids[1]])) / width
    return float(offset @ along) / width, float(offset @ down) / width, opening


def extract_features(landmarks, blendshapes: dict[str, float], matrix,
                     width: int, height: int) -> GazeFeatures:
    """Landmarks are MediaPipe's normalised ones; `matrix` is the 4x4 face pose."""
    points = np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float64)
    r_h, r_v, r_open = _eye(points, R_CORNERS, R_LIDS, R_IRIS)
    l_h, l_v, l_open = _eye(points, L_CORNERS, L_LIDS, L_IRIS)

    bs = blendshapes.get
    # Positive = toward the subject's left / upward, averaged over both eyes.
    look_h = 0.5 * ((bs("eyeLookOutLeft", 0.0) + bs("eyeLookInRight", 0.0))
                    - (bs("eyeLookInLeft", 0.0) + bs("eyeLookOutRight", 0.0)))
    look_v = 0.5 * ((bs("eyeLookUpLeft", 0.0) + bs("eyeLookUpRight", 0.0))
                    - (bs("eyeLookDownLeft", 0.0) + bs("eyeLookDownRight", 0.0)))
    blink = 0.5 * (bs("eyeBlinkLeft", 0.0) + bs("eyeBlinkRight", 0.0))

    if matrix is not None:
        m = np.asarray(matrix, dtype=np.float64)
        yaw = math.atan2(m[0, 2], m[2, 2])
        pitch = math.asin(max(-1.0, min(1.0, -m[1, 2])))
        roll = math.atan2(m[1, 0], m[1, 1])
        tx, ty, tz = m[0, 3] / 10.0, m[1, 3] / 10.0, m[2, 3] / 10.0
    else:
        yaw = pitch = roll = tx = ty = tz = 0.0

    vector = np.array([r_h, r_v, l_h, l_v, look_h, look_v, r_open, l_open,
                       yaw, pitch, roll, tx, ty, tz], dtype=np.float64)
    return GazeFeatures(vector=vector, blink=float(blink),
                        iris_px=(tuple(points[R_IRIS]), tuple(points[L_IRIS])))


def ensure_model() -> Path:
    """Path to the face landmark model, downloading it once (3.7 MB) if needed."""
    path = home_dir() / "face_landmarker.task"
    if not path.exists() or path.stat().st_size < 1_000_000:
        print(f"Downloading the face model to {path} ...", file=sys.stderr)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


class FaceTracker:
    def __init__(self, model_path: Path | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path or ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> GazeFeatures | None:
        """One BGR frame -> features, or None when no face is visible."""
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._last_ms = max(self._last_ms + 1, int(t * 1000.0))
        result = self._landmarker.detect_for_video(image, self._last_ms)
        if not result.face_landmarks:
            return None
        blend = {}
        if result.face_blendshapes:
            blend = {c.category_name: c.score for c in result.face_blendshapes[0]}
        matrix = (result.facial_transformation_matrixes[0]
                  if result.facial_transformation_matrixes else None)
        height, width = frame_bgr.shape[:2]
        return extract_features(result.face_landmarks[0], blend, matrix, width, height)

    def close(self) -> None:
        self._landmarker.close()


DARK_LEVEL = 6.0        # mean pixel value (0..255) below which the lens is covered


def is_dark(frame: np.ndarray) -> bool:
    """True when the image is black: privacy shutter closed or lens covered."""
    return float(frame[::8, ::8].mean()) < DARK_LEVEL


class Camera:
    def __init__(self, params: CameraParams):
        import cv2

        wanted = [cv2.CAP_PROP_FRAME_WIDTH, params.width,
                  cv2.CAP_PROP_FRAME_HEIGHT, params.height,
                  cv2.CAP_PROP_FPS, params.fps]
        # Asking for the format while opening is much faster on Windows than
        # changing it afterwards. Media Foundation gave 30 fps where
        # DirectShow gave 16 on the webcam this was developed with.
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_ANY]
        self._cap = None
        for backend in backends:
            cap = cv2.VideoCapture(params.index, backend, wanted)
            if cap.isOpened() and cap.read()[0]:
                self._cap = cap
                break
            cap.release()
        if self._cap is None:                   # last resort: defaults, then adjust
            cap = cv2.VideoCapture(params.index)
            if cap.isOpened():
                for prop, value in zip(wanted[::2], wanted[1::2]):
                    cap.set(prop, value)
                self._cap = cap
        if self._cap is None:
            raise RuntimeError(
                f"Camera {params.index} did not open. Close other apps using it, "
                "or pick another with --camera N.")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def close(self) -> None:
        self._cap.release()
