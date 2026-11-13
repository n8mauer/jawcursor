"""AirPods motion samples: wire formats, clock mapping and the receiver.

Two senders are understood on the same port number:

* JawBridge (the Swift app in apple/): one JSON object per UDP datagram.
* Sensor Logger (App Store): HTTP POST of batched JSON to /data.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterable

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ImuSample:
    t: float                      # seconds on the sender's clock
    gyro: Vec3                    # rotation rate, rad/s
    accel: Vec3                   # user acceleration (gravity removed)
    quat: tuple[float, float, float, float] | None = None   # w, x, y, z
    location: str | None = None   # "left", "right" or "default"


def _vec3(value) -> Vec3:
    x, y, z = value
    return float(x), float(y), float(z)


def parse_bridge_packet(data: bytes) -> ImuSample | None:
    """One JawBridge datagram -> sample, or None if it is not one."""
    try:
        obj = json.loads(data)
        quat = obj.get("q")
        return ImuSample(
            t=float(obj["t"]),
            gyro=_vec3(obj["rr"]),
            accel=_vec3(obj["ua"]),
            quat=tuple(float(v) for v in quat) if quat and len(quat) == 4 else None,
            location=obj.get("loc"),
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def parse_sensor_logger(body: dict) -> list[ImuSample]:
    """A Sensor Logger HTTP push body -> the headphone samples inside it."""
    samples: list[ImuSample] = []
    payload = body.get("payload") if isinstance(body, dict) else None
    for item in payload or []:
        if not isinstance(item, dict) or item.get("name") != "headphone":
            continue
        # Fields sit under "values" in current releases; accept flat items too.
        values = item.get("values") if isinstance(item.get("values"), dict) else item
        try:
            quat = None
            if "quaternionW" in values:
                quat = tuple(float(values[f"quaternion{axis}"]) for axis in "WXYZ")
            samples.append(ImuSample(
                t=float(item["time"]) / 1e9,          # epoch nanoseconds
                gyro=_vec3(values[f"rotationRate{axis}"] for axis in "XYZ"),
                accel=_vec3(values[f"acceleration{axis}"] for axis in "XYZ"),
                quat=quat,
                location=values.get("devicelocation"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    samples.sort(key=lambda s: s.t)
    return samples


class ClockMapper:
    """Maps sender timestamps onto the local monotonic clock.

    The offset is the smallest (arrival - sender time) seen recently, so
    network jitter and batching delay do not shift where a sample lands.
    """

    def __init__(self, window: float = 20.0, reset_jump: float = 5.0):
        self._window = window
        self._reset_jump = reset_jump
        self._history: deque[tuple[float, float]] = deque()
        self._offset: float | None = None

    def to_local(self, t_sender: float, t_recv: float) -> float:
        return self.batch_to_local([t_sender], t_recv)[0]

    def batch_to_local(self, sender_times: list[float], t_recv: float) -> list[float]:
        """Maps samples that arrived together. The newest one sets the offset
        for all of them, so a batch keeps its internal spacing."""
        if not sender_times:
            return []
        delta = t_recv - max(sender_times)
        if self._offset is not None and abs(delta - self._offset) > self._reset_jump:
            self._history.clear()            # sender restarted or changed clocks
        self._history.append((t_recv, delta))
        while self._history and t_recv - self._history[0][0] > self._window:
            self._history.popleft()
        self._offset = min(d for _, d in self._history)
        return [t + self._offset for t in sender_times]


SampleHandler = Callable[[ImuSample, float], None]


class ImuServer:
    """Listens for both senders and hands every sample to `on_sample`.

    `on_sample(sample, local_t)` runs on a receiver thread; `local_t` is when
    the sample was measured, on this machine's monotonic clock.
    """

    def __init__(self, host: str, port: int, on_sample: SampleHandler,
                 clock: Callable[[], float] = time.monotonic):
        self.host, self.port = host, port
        self._on_sample = on_sample
        self._clock = clock
        self._lock = threading.Lock()
        self._mapper = ClockMapper()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._udp: socket.socket | None = None
        self._http: ThreadingHTTPServer | None = None
        self._arrivals: deque[float] = deque(maxlen=400)
        self.count = 0
        self.source: str | None = None
        self.last_recv: float | None = None
        self.last_location: str | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):                           # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, OSError):
                    self.send_response(400)
                    self.end_headers()
                    return
                server.deliver(parse_sensor_logger(body), "sensor-logger")
                self._reply(b"{}")

            def do_GET(self):                            # noqa: N802
                self._reply(json.dumps(
                    {"jawcursor": "imu endpoint", "samples": server.count}).encode())

            def _reply(self, payload: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):                # keep the console quiet
                pass

        # UDP and HTTP share one port number. With port 0 the system picks the
        # UDP port, and the same TCP number may be reserved (Windows keeps
        # blocks of TCP ports for Hyper-V and WSL), so try again if it is.
        error: OSError | None = None
        for _ in range(25 if self.port == 0 else 1):
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                udp.bind((self.host, self.port))
                self._http = ThreadingHTTPServer((self.host, udp.getsockname()[1]), Handler)
            except OSError as failure:
                udp.close()
                error = failure
                continue
            self._udp = udp
            break
        else:
            raise OSError(f"Port {self.port} is not available for both UDP and HTTP "
                          f"({error}). Pick another with --port.")
        self._udp.settimeout(0.2)
        self.port = self._udp.getsockname()[1]
        self._http.daemon_threads = True
        for target in (self._udp_loop, self._http.serve_forever):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._udp is not None:
            self._udp.close()

    # -- ingestion ---------------------------------------------------------
    def _udp_loop(self) -> None:
        assert self._udp is not None
        while not self._stop.is_set():
            try:
                data, _ = self._udp.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            sample = parse_bridge_packet(data)
            if sample is not None:
                self.deliver([sample], "jawbridge")

    def deliver(self, samples: Iterable[ImuSample], source: str) -> None:
        now = self._clock()
        samples = list(samples)
        with self._lock:
            local_times = self._mapper.batch_to_local([s.t for s in samples], now)
            for sample, local_t in zip(samples, local_times):
                self.count += 1
                self.source = source
                self.last_recv = now
                self.last_location = sample.location
                self._arrivals.append(local_t)
                self._on_sample(sample, local_t)

    # -- status ------------------------------------------------------------
    def rate(self) -> float:
        """Samples per second over the last couple of seconds."""
        with self._lock:
            times = list(self._arrivals)
        if len(times) < 2:
            return 0.0
        recent = [t for t in times if times[-1] - t <= 2.0]
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def is_live(self, max_age: float = 1.5) -> bool:
        return self.last_recv is not None and self._clock() - self.last_recv <= max_age


def lan_address() -> str:
    """This machine's address on the local network (no packet is sent)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()
