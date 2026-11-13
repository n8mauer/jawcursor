"""Jaw-clench detection from earbud motion, and turning clenches into clicks.

Clenching bulges the jaw muscles next to the ear canal, which nudges the
earbud: a short, small burst in rotation rate and acceleration. Deliberate
head motion is larger and slower. The detector therefore

1. removes the slow part of the signal (head motion) with a running average,
2. measures what is left against a running noise floor,
3. accepts a burst only if the head was otherwise still, the burst was short,
   and the earbud ended up pointing almost where it started.

It emits when the burst ends and reports when it began, so the caller can
click where the cursor was before the clench disturbed anything.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, replace

from .config import ClenchParams, GestureParams
from .imu import ImuSample, Vec3


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _lerp(a: Vec3, b: Vec3, k: float) -> Vec3:
    return a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1]), a[2] + k * (b[2] - a[2])


@dataclass(frozen=True)
class ClenchEvent:
    t_onset: float      # sender clock, first sample over threshold
    t_end: float
    peak: float         # peak score, in multiples of the noise floor
    duration: float
    angle: float        # net earbud rotation over the burst, rad


class ClenchDetector:
    IDLE, BURST, WAIT = "idle", "burst", "wait"

    def __init__(self, params: ClenchParams | None = None):
        self.p = params or ClenchParams()
        self.reset()

    def reset(self) -> None:
        self._t_prev: float | None = None
        self._t_first: float | None = None
        self._w_slow: Vec3 = (0.0, 0.0, 0.0)
        self._a_slow: Vec3 = (0.0, 0.0, 0.0)
        self._hist: deque[tuple[float, float, float]] = deque()   # t, |gyro|, |accel|
        self._noise_w = self.p.gyro_floor
        self._noise_a = self.p.accel_floor
        self.state = self.IDLE
        self._blocked_until = -math.inf
        self._last_dir: Vec3 | None = None      # onset direction of the last clench
        self._last_end = -math.inf
        # exposed for the monitor
        self.score = 0.0
        self.head_rate = 0.0
        self.rejected: str | None = None        # why the last candidate was dropped

    # -- noise floor -------------------------------------------------------
    def _learn(self, t: float, bw: float, ba: float) -> None:
        self._hist.append((t, bw, ba))
        while self._hist and t - self._hist[0][0] > self.p.baseline_window:
            self._hist.popleft()
        if len(self._hist) >= 5:
            self._noise_w = max(statistics.median(h[1] for h in self._hist), self.p.gyro_floor)
            self._noise_a = max(statistics.median(h[2] for h in self._hist), self.p.accel_floor)

    # -- main entry --------------------------------------------------------
    def update(self, s: ImuSample) -> ClenchEvent | None:
        p = self.p
        if self._t_prev is not None and -p.max_gap < s.t - self._t_prev <= 0.0:
            return None                         # duplicate or out-of-order packet
        if self._t_prev is None or not (0.0 < s.t - self._t_prev <= p.max_gap):
            keep_floor = self._t_prev is not None and s.t > self._t_prev
            noise = (self._noise_w, self._noise_a)
            self.reset()
            if keep_floor:
                self._noise_w, self._noise_a = noise
            self._t_prev = self._t_first = s.t
            self._w_slow, self._a_slow = s.gyro, s.accel
            return None

        dt = s.t - self._t_prev
        self._t_prev = s.t
        k = 1.0 - math.exp(-dt / p.slow_tau)
        self._w_slow = _lerp(self._w_slow, s.gyro, k)
        self._a_slow = _lerp(self._a_slow, s.accel, k)
        w_fast = _sub(s.gyro, self._w_slow)
        bw = _norm(w_fast)
        ba = _norm(_sub(s.accel, self._a_slow))
        self.head_rate = _norm(self._w_slow)
        self.score = max(p.gyro_weight * bw / self._noise_w,
                         p.accel_weight * ba / self._noise_a)
        still = self.head_rate <= p.max_head_rate

        if self.state == self.IDLE:
            warm = self._t_first is not None and s.t - self._t_first >= p.warmup
            if warm and still and self.score >= p.k_on and s.t >= self._blocked_until:
                self.state = self.BURST
                self._t0 = s.t
                self._peak = self.score
                self._turn: Vec3 = (s.gyro[0] * dt, s.gyro[1] * dt, s.gyro[2] * dt)
                self._dir: Vec3 = w_fast
                self._dir_n = 1
                self._dir_strength = bw / self._noise_w
            else:
                self._learn(s.t, bw, ba)
            return None

        if self.state == self.WAIT:
            # A rejected long burst: keep learning so the floor can rise to it.
            self._learn(s.t, bw, ba)
            if self.score < p.k_off:
                self.state = self.IDLE
                self._blocked_until = s.t + p.refractory
            return None

        # state == BURST
        self._peak = max(self._peak, self.score)
        self._turn = (self._turn[0] + s.gyro[0] * dt,
                      self._turn[1] + s.gyro[1] * dt,
                      self._turn[2] + s.gyro[2] * dt)
        if self._dir_n < 2:
            self._dir = (self._dir[0] + w_fast[0], self._dir[1] + w_fast[1], self._dir[2] + w_fast[2])
            self._dir_n += 1
            self._dir_strength = max(self._dir_strength, bw / self._noise_w)

        if not still:
            return self._reject("head moving")
        if s.t - self._t0 > p.max_duration:
            return self._reject("too long")
        if self.score >= p.k_off:
            return None

        # Burst is over. Judge it.
        self.state = self.IDLE
        self._blocked_until = s.t + p.refractory
        angle = _norm(self._turn)
        if angle > p.max_angle:
            self.rejected = "head turned"
            return None

        direction = self._dir
        informative = self._dir_strength >= 3.0 and _norm(direction) > 0
        if (p.suppress_release and informative and self._last_dir is not None
                and self._t0 - self._last_end <= p.release_window):
            a, b = direction, self._last_dir
            cosine = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (_norm(a) * _norm(b))
            if cosine < -0.7:
                self._last_dir = None           # a release answers one clench only
                self.rejected = "release"
                return None

        self._last_dir = direction if informative else None
        self._last_end = s.t
        self.rejected = None
        return ClenchEvent(t_onset=self._t0, t_end=s.t, peak=self._peak,
                           duration=s.t - self._t0, angle=angle)

    def _reject(self, reason: str) -> None:
        self.state = self.WAIT
        self.rejected = reason
        return None


# --------------------------------------------------------------------------
# Clenches -> clicks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    button: str         # "left" or "right"
    t_event: float      # when the triggering clench began (local clock)


class GestureResolver:
    """Decides what each clench means. Pure logic; the caller supplies time."""

    def __init__(self, params: GestureParams | None = None):
        self.p = params or GestureParams()
        self._pending: float | None = None

    def on_clench(self, t: float) -> list[Action]:
        if self.p.double_action != "right_click":
            return [Action("left", t)]
        if self._pending is not None and t - self._pending <= self.p.double_window:
            first, self._pending = self._pending, None
            return [Action("right", first)]
        actions = self.poll(t)
        self._pending = t
        return actions

    def poll(self, now: float) -> list[Action]:
        """Releases a lone clench as a left click once the window has passed."""
        if self._pending is not None and now - self._pending > self.p.double_window:
            first, self._pending = self._pending, None
            return [Action("left", first)]
        return []


# --------------------------------------------------------------------------
# Fitting thresholds to one person's clench
# --------------------------------------------------------------------------

@dataclass
class ClenchRecording:
    samples: list[ImuSample]                    # on one continuous clock
    cues: list[float]                           # times the user was told to clench
    negative_spans: list[tuple[float, float]]   # spans with no clenching at all


CUE_WINDOW = (-0.15, 1.5)       # s around a cue in which the clench may start


def detect_all(samples: list[ImuSample], params: ClenchParams) -> list[ClenchEvent]:
    detector = ClenchDetector(params)
    return [ev for ev in map(detector.update, samples) if ev is not None]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _score_events(events: list[ClenchEvent], rec: ClenchRecording) -> tuple[int, int]:
    hits = sum(
        any(cue + CUE_WINDOW[0] <= ev.t_onset <= cue + CUE_WINDOW[1] for ev in events)
        for cue in rec.cues)
    false_alarms = sum(
        any(start <= ev.t_onset <= end for start, end in rec.negative_spans)
        for ev in events)
    return hits, false_alarms


def fit_clench_params(rec: ClenchRecording, base: ClenchParams | None = None
                      ) -> tuple[ClenchParams, dict]:
    """Picks thresholds from a recording of rest, movement and cued clenches.

    Returns the fitted parameters and a report. If the clenches cannot be
    found in the signal at all, the base parameters come back unchanged and
    report["ok"] is False.
    """
    base = base or ClenchParams()
    negative_minutes = max(sum(e - s for s, e in rec.negative_spans) / 60.0, 1e-6)
    report: dict = {"ok": False, "cues": len(rec.cues)}

    # Pass 1: look permissively to learn what this person's clench looks like.
    probe = replace(base, k_on=3.0, k_off=1.5, max_duration=1.0, max_angle=math.inf,
                    refractory=0.15, suppress_release=False)
    found: list[ClenchEvent] = []
    for cue in rec.cues:
        near = [ev for ev in detect_all(rec.samples, probe)
                if cue + CUE_WINDOW[0] <= ev.t_onset <= cue + CUE_WINDOW[1]]
        if near:
            found.append(max(near, key=lambda ev: ev.peak))
    report["found_in_probe"] = len(found)
    if not rec.cues or len(found) < 0.5 * len(rec.cues):
        report["reason"] = "clenches are not visible in the motion signal"
        return base, report

    max_duration = min(max(1.6 * _percentile([ev.duration for ev in found], 0.9), 0.25), 0.8)
    max_angle = min(max(3.0 * _percentile([ev.angle for ev in found], 0.9), 0.03), 0.25)

    # Pass 2: try each channel choice and threshold, replaying the recording.
    grid = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 26.0, 33.0, 42.0]
    channels = {"gyro+accel": (1.0, 1.0), "gyro": (1.0, 0.0), "accel": (0.0, 1.0)}
    best = None
    for name, (gw, aw) in channels.items():
        rows = []
        for k in grid:
            params = replace(base, k_on=k, k_off=max(1.5, k / 2.5), gyro_weight=gw,
                             accel_weight=aw, max_duration=max_duration, max_angle=max_angle)
            hits, false_alarms = _score_events(detect_all(rec.samples, params), rec)
            recall = hits / len(rec.cues)
            per_minute = false_alarms / negative_minutes
            rows.append((recall - 0.15 * per_minute, k, recall, per_minute, hits, false_alarms))
        top = max(row[0] for row in rows)
        # Take the middle of the stretch of thresholds that all do about as well.
        start = next(i for i, row in enumerate(rows) if row[0] >= top - 0.02)
        end = start
        while end + 1 < len(rows) and rows[end + 1][0] >= top - 0.02:
            end += 1
        k_pick = math.sqrt(rows[start][1] * rows[end][1])
        chosen = min(rows[start:end + 1], key=lambda row: abs(math.log(row[1] / k_pick)))
        candidate = (top, end - start, name, gw, aw, k_pick, chosen)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    _, plateau, name, gw, aw, k_pick, chosen = best
    fitted = replace(base, k_on=round(k_pick, 2), k_off=round(max(1.5, k_pick / 2.5), 2),
                     gyro_weight=gw, accel_weight=aw,
                     max_duration=round(max_duration, 3), max_angle=round(max_angle, 4))
    hits, false_alarms = _score_events(detect_all(rec.samples, fitted), rec)
    report.update(
        ok=hits >= 0.7 * len(rec.cues),
        channels=name, k_on=fitted.k_on, hits=hits, false_alarms=false_alarms,
        recall=hits / len(rec.cues), false_per_minute=false_alarms / negative_minutes,
        max_duration=fitted.max_duration, max_angle=fitted.max_angle,
        threshold_margin_steps=plateau)
    if not report["ok"]:
        report["reason"] = "too few cued clenches were detected"
    return fitted, report
