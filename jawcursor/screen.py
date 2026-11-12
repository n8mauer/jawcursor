"""Primary screen size in the same units the mouse is moved in."""

from __future__ import annotations

import sys


def enable_dpi_awareness() -> None:
    """On Windows, makes coordinates mean physical pixels. Call once, first."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)      # per-monitor
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def screen_size() -> tuple[int, int]:
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    if sys.platform == "darwin":
        try:
            import Quartz       # installed with pynput

            bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            return int(bounds.size.width), int(bounds.size.height)
        except Exception:
            pass
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    try:
        return int(root.winfo_screenwidth()), int(root.winfo_screenheight())
    finally:
        root.destroy()
