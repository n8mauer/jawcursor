"""Settings, with defaults that work before any calibration has been run."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


def home_dir() -> Path:
    """Folder for the config, calibration results and the face model."""
    path = Path(os.environ.get("JAWCURSOR_HOME") or Path.home() / ".jawcursor")
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class ClenchParams:
    # Separating a clench from deliberate head motion
    slow_tau: float = 0.40          # s. Slower than this counts as head motion.
    baseline_window: float = 6.0    # s of history used for the noise floor
    warmup: float = 1.0             # s of data needed before anything can fire
    # Trigger thresholds, as multiples of the noise floor
    k_on: float = 6.0
    k_off: float = 2.5
    gyro_weight: float = 1.0        # 0 switches the gyro channel off
    accel_weight: float = 1.0       # 0 switches the accelerometer channel off
    gyro_floor: float = 0.006       # rad/s. Noise floor never drops below this.
    accel_floor: float = 0.003      # in the sender's acceleration unit
    # Rejecting things that are not a clench
    max_head_rate: float = 0.5      # rad/s of slow rotation. Above it, ignore.
    max_duration: float = 0.45      # s. Longer bursts are head motion or chewing.
    max_angle: float = 0.06         # rad the earbud may turn during one burst
    refractory: float = 0.25        # s of deafness after a burst ends
    # A held clench makes a second burst on release, pointing the other way
    suppress_release: bool = True
    release_window: float = 1.5     # s after a clench in which a release can come
    max_gap: float = 1.0            # s between samples before state is reset


@dataclass
class GestureParams:
    # "double_click": every clench is a left click, two quick ones double-click.
    # "right_click":  one clench = left click, two quick ones = right click
    #                 (a single click then waits for double_window to pass).
    double_action: str = "double_click"
    double_window: float = 0.6


@dataclass
class PointerParams:
    min_cutoff: float = 0.5         # Hz. Lower = steadier when fixating.
    beta: float = 1.0               # Higher = snappier on large eye movements.
    d_cutoff: float = 1.0
    deadzone: float = 0.012         # fraction of screen width the gaze may wander
    click_lead: float = 0.12        # s before the clench where the click lands
    freeze_after_click: float = 0.35
    blink_threshold: float = 0.5
    blink_hold: float = 0.12        # s to hold still after a blink
    yield_seconds: float = 2.0      # hand control back when the real mouse moves
    yield_tolerance_px: int = 4


@dataclass
class CameraParams:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class ImuParams:
    host: str = "0.0.0.0"
    port: int = 7355                # UDP for JawBridge, HTTP POST for Sensor Logger


@dataclass
class Config:
    clench: ClenchParams = field(default_factory=ClenchParams)
    gesture: GestureParams = field(default_factory=GestureParams)
    pointer: PointerParams = field(default_factory=PointerParams)
    camera: CameraParams = field(default_factory=CameraParams)
    imu: ImuParams = field(default_factory=ImuParams)


def _merge(obj, data: dict) -> None:
    for f in fields(obj):
        if f.name not in data:
            continue
        current = getattr(obj, f.name)
        value = data[f.name]
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, bool):
            setattr(obj, f.name, bool(value))
        elif isinstance(current, (int, float)) and isinstance(value, (int, float)):
            setattr(obj, f.name, type(current)(value))
        elif isinstance(current, str) and isinstance(value, str):
            setattr(obj, f.name, value)


def config_path() -> Path:
    return home_dir() / "config.json"


def load_config(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or config_path()
    if path.exists():
        _merge(cfg, json.loads(path.read_text(encoding="utf-8")))
    return cfg


def save_config(cfg: Config, path: Path | None = None) -> Path:
    path = path or config_path()
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return path
