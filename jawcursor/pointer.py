"""Turns gaze estimates and click requests into mouse movement."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable

from .config import PointerParams
from .filters import OneEuro2D

Point = tuple[int, int]


class PynputMouse:
    def __init__(self):
        from pynput.mouse import Button, Controller

        self._controller = Controller()
        self._buttons = {"left": Button.left, "right": Button.right}

    def position(self) -> Point:
        x, y = self._controller.position
        return int(x), int(y)

    def move(self, x: int, y: int) -> None:
        self._controller.position = (int(x), int(y))

    def click(self, button: str = "left", count: int = 1) -> None:
        self._controller.click(self._buttons[button], count)


class DryRunMouse:
    """Records what would have happened and touches nothing."""

    def __init__(self, start: Point = (0, 0), verbose: bool = False):
        self._pos = start
        self.moves: list[Point] = []
        self.clicks: list[tuple[str, Point]] = []
        self.verbose = verbose

    def position(self) -> Point:
        return self._pos

    def move(self, x: int, y: int) -> None:
        self._pos = (int(x), int(y))
        self.moves.append(self._pos)

    def click(self, button: str = "left", count: int = 1) -> None:
        for _ in range(count):
            self.clicks.append((button, self._pos))
        if self.verbose:
            print(f"[dry run] {button} click at {self._pos}")


class PointerController:
    """Smooths the gaze point, moves the cursor, and places clicks.

    * Small wanderings of the gaze estimate are absorbed (the cursor is on a
      short leash), so the pointer holds still while you look at a target.
    * A click lands where the cursor was just before the clench began, and the
      cursor stays there briefly, because clenching disturbs the head.
    * Moving the physical mouse takes control back for a moment.
    """

    def __init__(self, mouse, screen: tuple[int, int], params: PointerParams | None = None,
                 clock: Callable[[], float] = time.monotonic, yield_to_mouse: bool = True):
        self.mouse = mouse
        self.width, self.height = screen
        self.p = params or PointerParams()
        self._clock = clock
        self._yield_to_mouse = yield_to_mouse   # off when the user steers by hand
        self._lock = threading.RLock()
        self._filter = OneEuro2D(self.p.min_cutoff, self.p.beta, self.p.d_cutoff)
        self._cursor: tuple[float, float] | None = None
        self._history: deque[tuple[float, int, int]] = deque(maxlen=240)
        self._last_set: Point | None = None
        self._frozen_until = -math.inf
        self._yield_until = -math.inf
        self._last_click: tuple[float, Point] | None = None
        self.paused = False

    # -- state -------------------------------------------------------------
    def toggle_pause(self) -> bool:
        with self._lock:
            self.paused = not self.paused
            self._resync()
            return self.paused

    def hold(self, seconds: float, t: float | None = None) -> None:
        """Keeps the cursor where it is, e.g. across a blink."""
        with self._lock:
            now = self._clock() if t is None else t
            self._frozen_until = max(self._frozen_until, now + seconds)

    def yielding(self, t: float | None = None) -> bool:
        return (self._clock() if t is None else t) < self._yield_until

    def _resync(self) -> None:
        self._filter.reset()
        self._cursor = None
        self._last_set = None

    def _check_manual(self, now: float) -> None:
        """Notices the physical mouse being moved and steps back."""
        if self._last_set is None or not self._yield_to_mouse:
            return
        actual = self.mouse.position()
        if math.dist(actual, self._last_set) > self.p.yield_tolerance_px:
            self._yield_until = now + self.p.yield_seconds
            self._resync()

    def _place(self, x: float, y: float, now: float) -> Point:
        pos = (int(round(min(max(x, 0), self.width - 1))),
               int(round(min(max(y, 0), self.height - 1))))
        self.mouse.move(*pos)
        self._last_set = pos
        self._history.append((now, pos[0], pos[1]))
        return pos

    # -- gaze --------------------------------------------------------------
    def update_gaze(self, nx: float, ny: float, t: float | None = None) -> Point | None:
        """Feeds one gaze estimate (0..1 each). Returns the cursor if it was set."""
        with self._lock:
            now = self._clock() if t is None else t
            if self.paused:
                return None
            self._check_manual(now)
            if now < self._yield_until:
                return None
            fx, fy = self._filter(nx, ny, now)
            target = (fx * self.width, fy * self.height)
            if self._cursor is None:
                self._cursor = target
            if now < self._frozen_until:
                return None
            leash = self.p.deadzone * self.width
            dx, dy = target[0] - self._cursor[0], target[1] - self._cursor[1]
            distance = math.hypot(dx, dy)
            if distance > leash:
                pull = (distance - leash) / distance
                self._cursor = (self._cursor[0] + dx * pull, self._cursor[1] + dy * pull)
            return self._place(self._cursor[0], self._cursor[1], now)

    # -- clicks ------------------------------------------------------------
    def _position_at(self, t: float) -> Point | None:
        chosen = None
        for stamp, x, y in self._history:
            if stamp > t:
                break
            chosen = (x, y)
        if chosen is None and self._history:
            chosen = self._history[0][1:]
        return chosen

    def click(self, t_event: float, button: str = "left", t: float | None = None) -> Point | None:
        """Clicks where the cursor was shortly before `t_event`."""
        with self._lock:
            now = self._clock() if t is None else t
            if self.paused:
                return None
            self._check_manual(now)
            if now < self._yield_until:
                return None
            pos = self._position_at(t_event - self.p.click_lead) or self.mouse.position()
            if (self._yield_to_mouse and self._last_click
                    and now - self._last_click[0] <= 0.8):
                pos = self._last_click[1]       # keep a double click on one spot
            self.mouse.move(*pos)
            self.mouse.click(button, 1)
            self._last_set = pos
            self._cursor = (float(pos[0]), float(pos[1]))
            self._frozen_until = max(self._frozen_until, now + self.p.freeze_after_click)
            self._last_click = (now, pos)
            return pos
