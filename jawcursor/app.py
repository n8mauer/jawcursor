"""The running program: camera loop, motion receiver, clicks."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .clench import Action, ClenchDetector, ClenchEvent, GestureResolver
from .config import Config, home_dir
from .imu import ImuSample, ImuServer, lan_address
from .pointer import DryRunMouse, PointerController, PynputMouse
from .screen import screen_size


def gaze_model_path() -> Path:
    return home_dir() / "gaze_model.json"


class ClenchPipeline:
    """Samples in, click actions out. Safe to feed from the receiver thread."""

    def __init__(self, cfg: Config, on_action: Callable[[Action], None],
                 on_event: Callable[[ClenchEvent, float], None] | None = None):
        self.detector = ClenchDetector(cfg.clench)
        self.resolver = GestureResolver(cfg.gesture)
        self._on_action = on_action
        self._on_event = on_event
        self._lock = threading.Lock()
        self.events = 0
        self._peak = 0.0

    def on_sample(self, sample: ImuSample, local_t: float) -> None:
        with self._lock:
            event = self.detector.update(sample)
            self._peak = max(self._peak, self.detector.score)
            actions: list[Action] = []
            onset_local = 0.0
            if event is not None:
                self.events += 1
                onset_local = local_t - (sample.t - event.t_onset)
                actions = self.resolver.on_clench(onset_local)
        if event is not None and self._on_event:
            self._on_event(event, onset_local)
        for action in actions:
            self._on_action(action)

    def poll(self, now: float) -> None:
        """Call regularly; releases clicks that were waiting on a second clench."""
        with self._lock:
            actions = self.resolver.poll(now)
        for action in actions:
            self._on_action(action)

    def take_peak(self) -> float:
        """Highest score since the last call (for the live monitor)."""
        with self._lock:
            peak, self._peak = self._peak, 0.0
            return peak


def connection_help(port: int) -> str:
    address = lan_address()
    lines = [f"Listening for AirPods motion on port {port}."]
    if sys.platform == "darwin":
        lines.append(f"  JawBridge on this Mac:      host 127.0.0.1, port {port}")
    lines += [f"  JawBridge on an iPhone:     host {address}, port {port}",
              f"  Sensor Logger HTTP push:    http://{address}:{port}/data"]
    return "\n".join(lines)


def _start_hotkeys(on_pause: Callable[[], None], on_quit: Callable[[], None]):
    try:
        from pynput import keyboard

        listener = keyboard.GlobalHotKeys({"<ctrl>+<alt>+p": on_pause, "<ctrl>+<alt>+q": on_quit})
        listener.daemon = True
        listener.start()
        return listener
    except Exception as error:      # missing permissions on macOS, no display, ...
        print(f"Hotkeys unavailable ({error}). Use Ctrl+C in this window to stop.")
        return None


def run(cfg: Config, *, dry_run: bool = False, use_gaze: bool = True, preview: bool = False,
        duration: float | None = None, hotkeys: bool = True) -> int:
    screen = screen_size()
    mouse = DryRunMouse(start=(screen[0] // 2, screen[1] // 2), verbose=True) if dry_run \
        else PynputMouse()
    pointer = PointerController(mouse, screen, cfg.pointer, yield_to_mouse=use_gaze)

    clicks = 0

    def on_action(action: Action) -> None:
        nonlocal clicks
        if pointer.click(action.t_event, action.button) is not None:
            clicks += 1

    pipeline = ClenchPipeline(cfg, on_action)
    server = ImuServer(cfg.imu.host, cfg.imu.port, pipeline.on_sample)

    mapper = camera = tracker = None
    if use_gaze:
        from .gaze import Camera, FaceTracker, is_dark
        from .mapping import GazeMapper

        if not gaze_model_path().exists():
            print("No gaze calibration yet. Run:  jawcursor calibrate gaze\n"
                  "(or start with --no-gaze to try clench-clicking with your normal mouse)")
            return 2
        mapper = GazeMapper.load(gaze_model_path())

    server.start()                  # before the camera, which can take a few seconds
    if use_gaze:
        print("Opening the camera ...")
        try:
            camera = Camera(cfg.camera)
            tracker = FaceTracker()
        except Exception:
            server.stop()
            if camera is not None:
                camera.close()
            raise
    stop = threading.Event()

    def toggle_pause() -> None:
        print("Paused." if pointer.toggle_pause() else "Resumed.")

    listener = _start_hotkeys(toggle_pause, stop.set) if hotkeys else None
    print(connection_help(server.port))
    print(f"Screen {screen[0]}x{screen[1]}. "
          + ("DRY RUN: the real cursor is not touched. " if dry_run else "")
          + ("Ctrl+Alt+P pauses, Ctrl+Alt+Q quits. " if listener is not None
             else "Ctrl+C in this window stops. ")
          + ("Moving the real mouse takes over for a moment." if use_gaze else ""))

    cv2 = None
    if preview and use_gaze:
        import cv2 as cv2       # noqa: PLC0414

    started = last_status = time.monotonic()
    frames = 0
    blink_until = 0.0
    seeing = "starting"             # what the camera has at the moment
    try:
        while not stop.is_set():
            now = time.monotonic()
            if duration is not None and now - started >= duration:
                break
            pipeline.poll(now)

            if use_gaze:
                frame = camera.read()
                now = time.monotonic()
                feats = tracker.process(frame, now) if frame is not None else None
                frames += 1
                seeing = ("eyes tracked" if feats is not None
                          else "no frames" if frame is None
                          else "image is black (shutter closed?)" if is_dark(frame)
                          else "no face in view")
                if feats is not None:
                    if feats.blink >= cfg.pointer.blink_threshold:
                        blink_until = now + cfg.pointer.blink_hold
                    elif now >= blink_until:
                        pointer.update_gaze(*mapper.predict(feats.vector), now)
                if cv2 is not None and frame is not None:
                    small = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
                    ratio = 640 / frame.shape[1]
                    if feats is not None:
                        for x, y in feats.iris_px:
                            cv2.circle(small, (int(x * ratio), int(y * ratio)), 3, (0, 255, 0), -1)
                    label = f"AirPods {server.rate():.0f} Hz  clicks {clicks}"
                    cv2.putText(small, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                    cv2.imshow("jawcursor", small)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
            else:
                time.sleep(0.02)

            if now - last_status >= 3.0:
                fps = frames / (now - last_status)
                frames, last_status = 0, now
                motion = (f"AirPods {server.rate():.0f} Hz via {server.source}"
                          if server.is_live() else "AirPods: no data")
                state = "paused" if pointer.paused else "yielding to mouse" \
                    if pointer.yielding() else "active"
                gaze = f"camera {fps:.0f} fps, {seeing} | " if use_gaze else ""
                print(f"  {gaze}{motion} | clicks {clicks} | {state}")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if listener is not None:
            listener.stop()
        if tracker is not None:
            tracker.close()
        if camera is not None:
            camera.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    print(f"Stopped. {clicks} click(s), {pipeline.events} clench(es) detected.")
    return 0
