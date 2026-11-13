"""Synthetic earbud motion, for testing the pipeline without AirPods.

The clench shape here is an assumption (a small two-lobed rotation pulse plus
an acceleration blip), not a measurement. It exercises the code paths; it does
not prove the detector works on a real jaw. Use `jawcursor record` for that.
"""

from __future__ import annotations

import json
import math
import random
import socket
import time
from dataclasses import dataclass

from .imu import ImuSample


def _lobe(tau: float, width: float) -> float:
    return math.sin(math.pi * tau / width) ** 2 if 0.0 <= tau <= width else 0.0


@dataclass
class _Clench:
    t: float
    hold: float
    strength: float


@dataclass
class _Turn:
    t: float
    duration: float
    peak: float


class SyntheticImu:
    CLENCH_AXIS = (0.80, 0.20, 0.56)        # arbitrary but fixed unit vector
    ACCEL_AXIS = (0.10, 0.70, 0.70)
    LOBE = 0.12                             # s per rotation lobe

    def __init__(self, seed: int = 0, gyro_noise: float = 0.004, accel_noise: float = 0.002,
                 clench_gyro: float = 0.16, clench_accel: float = 0.03):
        self._rng = random.Random(seed)
        self.gyro_noise, self.accel_noise = gyro_noise, accel_noise
        self.clench_gyro, self.clench_accel = clench_gyro, clench_accel
        self._clenches: list[_Clench] = []
        self._turns: list[_Turn] = []

    def clench(self, t: float, hold: float = 0.10, strength: float = 1.0) -> None:
        """A bite at `t`, released `hold` seconds later (0.10 = quick tap)."""
        self._clenches.append(_Clench(t, hold, strength))

    def head_turn(self, t: float, duration: float = 0.8, peak: float = 1.2) -> None:
        self._turns.append(_Turn(t, duration, peak))

    def sample(self, t: float) -> ImuSample:
        g = [self._rng.gauss(0.0, self.gyro_noise) for _ in range(3)]
        a = [self._rng.gauss(0.0, self.accel_noise) for _ in range(3)]
        for c in self._clenches:
            bite = _lobe(t - c.t, self.LOBE)
            release = _lobe(t - c.t - c.hold, self.LOBE)
            rotation = self.clench_gyro * c.strength * (bite - release)
            shake = self.clench_accel * c.strength * (bite + release)
            for i in range(3):
                g[i] += rotation * self.CLENCH_AXIS[i]
                a[i] += shake * self.ACCEL_AXIS[i]
        for turn in self._turns:
            rate = turn.peak * _lobe(t - turn.t, turn.duration)
            g[2] += rate                                    # yaw
            a[0] += 0.02 * rate
        return ImuSample(t=t, gyro=(g[0], g[1], g[2]), accel=(a[0], a[1], a[2]),
                         quat=(1.0, 0.0, 0.0, 0.0), location="left")

    def run(self, duration: float, rate: float = 25.0, start: float = 0.0) -> list[ImuSample]:
        return [self.sample(start + i / rate) for i in range(int(duration * rate))]


def bridge_packet(s: ImuSample) -> bytes:
    """Encodes a sample exactly as the JawBridge app does."""
    return json.dumps({
        "v": 1, "t": round(s.t, 4),
        "rr": [round(v, 5) for v in s.gyro],
        "ua": [round(v, 5) for v in s.accel],
        "q": list(s.quat or (1.0, 0.0, 0.0, 0.0)),
        "loc": s.location or "default",
    }, separators=(",", ":")).encode()


def stream(host: str, port: int, seconds: float, clench_every: float = 3.0,
           rate: float = 25.0, turn_every: float = 0.0) -> int:
    """Sends synthetic motion over UDP in real time. Returns packets sent."""
    imu = SyntheticImu(seed=1)
    t = 2.0
    while t < seconds:
        imu.clench(t)
        t += clench_every
    t = 3.5
    while turn_every and t < seconds:
        imu.head_turn(t)
        t += turn_every
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    begin = time.monotonic()
    sent = 0
    try:
        while True:
            elapsed = sent / rate
            if elapsed >= seconds:
                break
            delay = begin + elapsed - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            sock.sendto(bridge_packet(imu.sample(elapsed)), (host, port))
            sent += 1
    finally:
        sock.close()
    return sent
