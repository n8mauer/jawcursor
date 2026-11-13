"""The two calibrations: where you look (gaze) and what your clench looks like."""

from __future__ import annotations

import csv
import random
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from .clench import ClenchRecording, fit_clench_params
from .config import ClenchParams, Config
from .gaze import is_dark
from .imu import ImuSample
from .mapping import FitReport, GazeMapper, clean_dot_samples

# --------------------------------------------------------------------------
# Gaze
# --------------------------------------------------------------------------

MIN_SAMPLES_PER_DOT = 8


def dot_sequence(seed: int = 7) -> list[tuple[float, float]]:
    """13 dots (3x3 grid plus 4 inner), centre first, the rest shuffled."""
    edge = 0.07
    grid = [(x, y) for y in (edge, 0.5, 1 - edge) for x in (edge, 0.5, 1 - edge)]
    inner = [(0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)]
    rest = [p for p in grid if p != (0.5, 0.5)] + inner
    random.Random(seed).shuffle(rest)
    return [(0.5, 0.5)] + rest


def fit_gaze(per_dot: list[np.ndarray], dots: list[tuple[float, float]],
             aspect: float) -> tuple[GazeMapper, FitReport]:
    """Fits the mapper from the feature rows gathered at each dot."""
    rows, targets, groups = [], [], []
    for index, (samples, dot) in enumerate(zip(per_dot, dots)):
        samples = clean_dot_samples(np.asarray(samples, float)) if len(samples) else samples
        if len(samples) < MIN_SAMPLES_PER_DOT:
            continue
        rows.append(samples)
        targets.append(np.tile(dot, (len(samples), 1)))
        groups.append(np.full(len(samples), index))
    if len(rows) < 5:
        raise ValueError("too few dots had a clear view of your eyes")
    return GazeMapper.fit(np.vstack(rows), np.vstack(targets), np.concatenate(groups), aspect)


def run_gaze_calibration(camera, tracker, screen: tuple[int, int], cfg: Config,
                         glide: float = 0.45, settle: float = 0.6, collect: float = 1.0,
                         countdown: float = 3.0, result_seconds: float = 2.5,
                         fullscreen: bool = True,
                         on_dot: Callable[[int, tuple[float, float]], None] | None = None
                         ) -> tuple[GazeMapper, FitReport] | None:
    """Full-screen dot-following routine. Returns None if cancelled with Esc.

    `on_dot(index, position)` is called as each dot starts (used by tests).
    """
    import cv2

    width, height = screen
    window = "jawcursor calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window, width, height)
    canvas = np.zeros((height, width, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, width / 1900.0)

    def text(message: str, y: float, colour=(210, 210, 210)) -> None:
        size = cv2.getTextSize(message, font, scale, 2)[0]
        cv2.putText(canvas, message, ((width - size[0]) // 2, int(y * height)),
                    font, scale, colour, 2, cv2.LINE_AA)

    def show() -> bool:
        cv2.imshow(window, canvas)
        return (cv2.waitKey(1) & 0xFF) != 27

    dark = False

    def features(now: float):
        nonlocal dark
        frame = camera.read()
        if frame is None:
            return None
        dark = is_dark(frame)
        return tracker.process(frame, now)

    try:
        # Wait for a face, then count down. No key press needed.
        seen_since = None
        while True:
            now = time.monotonic()
            found = features(now) is not None
            seen_since = (seen_since or now) if found else None
            remaining = countdown - (now - seen_since) if seen_since else None
            canvas[:] = 0
            text("Follow the dot with your eyes. Keep your head fairly still.", 0.40)
            text("Sit as you normally do. Esc cancels.", 0.46)
            if dark:
                text("The camera image is black. Open its privacy shutter or lens cover.",
                     0.58, (80, 160, 255))
            elif remaining is None:
                text("No face in view. Check the camera and the lighting.", 0.58, (80, 160, 255))
            else:
                text(f"Starting in {max(1, int(remaining) + 1)}", 0.58, (120, 220, 120))
            if not show():
                return None
            if remaining is not None and remaining <= 0:
                break

        dots = dot_sequence()
        per_dot: list[list[np.ndarray]] = [[] for _ in dots]
        previous = dots[0]
        for index, dot in enumerate(dots):
            if on_dot is not None:
                on_dot(index, dot)
            begin = time.monotonic()
            while True:
                now = time.monotonic()
                elapsed = now - begin
                if elapsed >= glide + settle + collect:
                    break
                feats = features(now)
                move = min(1.0, elapsed / glide)
                ease = move * move * (3 - 2 * move)
                x = int((previous[0] + (dot[0] - previous[0]) * ease) * width)
                y = int((previous[1] + (dot[1] - previous[1]) * ease) * height)
                collecting = elapsed >= glide + settle
                if collecting and feats is not None and feats.blink < cfg.pointer.blink_threshold:
                    per_dot[index].append(feats.vector)
                progress = max(0.0, (elapsed - glide) / (settle + collect))
                radius = int((30 - 20 * progress) * scale)
                canvas[:] = 0
                cv2.circle(canvas, (x, y), radius, (90, 200, 90) if collecting else (60, 150, 255),
                           -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), max(2, int(3 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, f"{index + 1}/{len(dots)}", (20, height - 20), font,
                            scale * 0.7, (90, 90, 90), 1, cv2.LINE_AA)
                if not show():
                    return None
            previous = dot

        try:
            mapper, report = fit_gaze([np.array(s) for s in per_dot], dots, width / height)
        except ValueError as error:
            canvas[:] = 0
            text(f"Calibration failed: {error}.", 0.48, (80, 160, 255))
            end = time.monotonic() + 3.0
            while time.monotonic() < end and show():
                pass
            return None

        median_px, p90_px = report.pixels(width)
        canvas[:] = 0
        text(f"Done. Typical error about {median_px} px (9 in 10 within {p90_px} px).", 0.48,
             (120, 220, 120))
        end = time.monotonic() + result_seconds
        while time.monotonic() < end and show():
            pass
        return mapper, report
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)


def gaze_verdict(report: FitReport) -> str:
    error = report.median_error
    if error <= 0.035:
        return "good: small buttons should be reachable"
    if error <= 0.07:
        return "usable for large targets; zoom the page or enlarge UI elements"
    return ("poor: try brighter, even light on your face, the camera at eye level, "
            "sitting closer, and run it again")


# --------------------------------------------------------------------------
# Clench
# --------------------------------------------------------------------------

class SampleFeed:
    """Collects (sample, local_t) pairs from the receiver thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: list[tuple[ImuSample, float]] = []

    def on_sample(self, sample: ImuSample, local_t: float) -> None:
        with self._lock:
            self._items.append((sample, local_t))

    def snapshot(self) -> list[tuple[ImuSample, float]]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def run_clench_calibration(feed: SampleFeed, base: ClenchParams | None = None,
                           say: Callable[[str], None] = print,
                           clock: Callable[[], float] = time.monotonic,
                           sleep: Callable[[float], None] = time.sleep,
                           rest: float = 8.0, move: float = 12.0, clenches: int = 10,
                           interval: float = 2.5, wait_for_data: float = 30.0
                           ) -> tuple[ClenchParams, dict, ClenchRecording] | None:
    """Guided recording of rest, movement and cued clenches, then a fit.

    Returns None if no motion data arrives.
    """
    say("Waiting for AirPods motion data ...")
    deadline = clock() + wait_for_data
    while len(feed) < 10:
        if clock() > deadline:
            say("No motion data arrived. Is the relay app streaming to this machine?")
            return None
        sleep(0.2)

    def countdown(seconds: float) -> None:
        end = clock() + seconds
        while clock() < end:
            sleep(min(0.25, max(0.0, end - clock())))

    say(f"\n1/3  Sit still with your jaw relaxed for {rest:.0f} seconds.")
    countdown(1.5)
    rest_start = clock()
    countdown(rest)
    rest_end = clock()

    say(f"\n2/3  Now move normally for {move:.0f} seconds: turn your head, nod, talk, "
        "swallow.\n     Do NOT clench.")
    countdown(1.0)
    move_start = clock()
    countdown(move)
    move_end = clock()

    say(f"\n3/3  Each time you see CLENCH, bite down quickly and firmly, then let go.\n"
        f"     {clenches} times. Keep your head still.")
    countdown(2.0)
    cues: list[float] = []
    for number in range(1, clenches + 1):
        cues.append(clock())
        say(f"     CLENCH  ({number}/{clenches})\a")
        countdown(interval)
    countdown(0.5)

    # Put the samples on the local clock so they line up with the cue times.
    samples = [replace(sample, t=local_t) for sample, local_t in feed.snapshot()]
    samples.sort(key=lambda s: s.t)
    recording = ClenchRecording(samples=samples, cues=cues,
                                negative_spans=[(rest_start, rest_end), (move_start, move_end)])
    params, report = fit_clench_params(recording, base)
    return params, report, recording


def describe_clench_fit(report: dict) -> str:
    if report.get("found_in_probe", 0) < 0.5 * max(report.get("cues", 1), 1):
        return ("Your clenches barely show up in the earbud motion "
                f"({report.get('found_in_probe', 0)} of {report.get('cues', 0)} visible). "
                "Try a sharper bite on the back teeth, reseat the AirPods snugly, or use "
                "ear tips one size up, then calibrate again.")
    lines = [
        f"Detected {report['hits']} of {report['cues']} clenches, with "
        f"{report['false_alarms']} false trigger(s) while you were moving and talking "
        f"({report['false_per_minute']:.1f} per minute).",
        f"Using {report['channels']}, threshold {report['k_on']} x noise floor.",
    ]
    if not report["ok"]:
        lines.append("That is not reliable enough to save. Try a firmer, quicker bite and "
                     "calibrate again.")
    elif report["false_alarms"]:
        lines.append("Expect an occasional stray click when you talk or move.")
    return "\n".join(lines)


def save_recording_csv(path: Path, samples: list[ImuSample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "gx", "gy", "gz", "ax", "ay", "az", "qw", "qx", "qy", "qz", "loc"])
        for s in samples:
            quat = s.quat or ("", "", "", "")
            writer.writerow([f"{s.t:.4f}", *s.gyro, *s.accel, *quat, s.location or ""])


def load_recording_csv(path: Path) -> list[ImuSample]:
    samples: list[ImuSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            quat = None
            if row.get("qw"):
                quat = tuple(float(row[k]) for k in ("qw", "qx", "qy", "qz"))
            samples.append(ImuSample(
                t=float(row["t"]),
                gyro=(float(row["gx"]), float(row["gy"]), float(row["gz"])),
                accel=(float(row["ax"]), float(row["ay"]), float(row["az"])),
                quat=quat, location=row.get("loc") or None))
    return samples
