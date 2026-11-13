"""Command line: run, calibrate, monitor, record, replay, simulate, doctor."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import Config, config_path, home_dir, load_config, save_config
from .screen import enable_dpi_awareness, screen_size


def _config(args) -> Config:
    cfg = load_config(Path(args.config) if args.config else None)
    if getattr(args, "port", None):
        cfg.imu.port = args.port
    if getattr(args, "host", None):
        cfg.imu.host = args.host
    if getattr(args, "camera", None) is not None:
        cfg.camera.index = args.camera
    return cfg


# -- run ---------------------------------------------------------------------

def cmd_run(args) -> int:
    from .app import run

    cfg = _config(args)
    if args.right_click:
        cfg.gesture.double_action = "right_click"
    return run(cfg, dry_run=args.dry_run, use_gaze=not args.no_gaze, preview=args.preview,
               duration=args.duration, hotkeys=not args.no_hotkeys)


# -- calibrate ---------------------------------------------------------------

def cmd_calibrate_gaze(args) -> int:
    from .app import gaze_model_path
    from .calibrate import gaze_verdict, run_gaze_calibration
    from .gaze import Camera, FaceTracker

    cfg = _config(args)
    screen = screen_size()
    camera = Camera(cfg.camera)
    tracker = FaceTracker()
    try:
        result = run_gaze_calibration(camera, tracker, screen, cfg)
    finally:
        tracker.close()
        camera.close()
    if result is None:
        print("Gaze calibration cancelled or failed; nothing was saved.")
        return 1
    mapper, report = result
    mapper.save(gaze_model_path(), report, screen)
    median_px, p90_px = report.pixels(screen[0])
    print(f"Saved {gaze_model_path()}")
    print(f"Typical error {median_px} px, 9 in 10 within {p90_px} px, measured on dots held "
          f"out of the fit ({report.points} dots, {report.samples} frames, {report.kind} model).")
    print("Verdict: " + gaze_verdict(report))
    return 0


def cmd_calibrate_clench(args) -> int:
    from .app import connection_help
    from .calibrate import (SampleFeed, describe_clench_fit, run_clench_calibration,
                            save_recording_csv)
    from .imu import ImuServer

    cfg = _config(args)
    feed = SampleFeed()
    server = ImuServer(cfg.imu.host, cfg.imu.port, feed.on_sample)
    server.start()
    print(connection_help(server.port))
    try:
        result = run_clench_calibration(feed, cfg.clench, clenches=args.clenches)
    except KeyboardInterrupt:
        print("\nCancelled; nothing was saved.")
        return 1
    finally:
        server.stop()
    if result is None:
        return 1
    params, report, recording = result
    csv_path = home_dir() / "clench_calibration.csv"
    save_recording_csv(csv_path, recording.samples)
    print("\n" + describe_clench_fit(report))
    print(f"Raw recording kept at {csv_path}")
    if not report["ok"]:
        return 1
    cfg.clench = params
    print(f"Saved {save_config(cfg, Path(args.config) if args.config else None)}")
    return 0


# -- monitor / record / replay -----------------------------------------------

def cmd_monitor(args) -> int:
    from .app import ClenchPipeline, connection_help
    from .imu import ImuServer

    cfg = _config(args)

    def on_event(event, _onset) -> None:
        print(f"\r  CLENCH  peak {event.peak:5.1f}x floor, {event.duration * 1000:3.0f} ms, "
              f"turned {event.angle * 57.3:.1f} deg" + " " * 20)

    pipeline = ClenchPipeline(cfg, on_action=lambda action: None, on_event=on_event)
    server = ImuServer(cfg.imu.host, cfg.imu.port, pipeline.on_sample)
    server.start()
    print(connection_help(server.port))
    print(f"Threshold {cfg.clench.k_on} x noise floor. Ctrl+C stops.\n")
    end = time.monotonic() + args.seconds if args.seconds else None
    try:
        while end is None or time.monotonic() < end:
            time.sleep(0.25)
            if not server.is_live():
                line = "waiting for motion data ..."
            else:
                peak = pipeline.take_peak()
                bar = "#" * min(40, int(peak * 2))
                line = (f"{server.rate():5.1f} Hz {server.last_location or '':>7} | head "
                        f"{pipeline.detector.head_rate:4.2f} rad/s | score {peak:5.1f} |{bar}")
            print("\r" + line.ljust(100), end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    print(f"\n{server.count} samples, {pipeline.events} clench(es).")
    return 0


def cmd_record(args) -> int:
    from .app import connection_help
    from .calibrate import SampleFeed, save_recording_csv
    from .imu import ImuServer

    cfg = _config(args)
    feed = SampleFeed()
    server = ImuServer(cfg.imu.host, cfg.imu.port, feed.on_sample)
    server.start()
    print(connection_help(server.port))
    print(f"Recording {args.seconds:.0f} s to {args.file} ...")
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    samples = sorted((replace(s, t=local_t) for s, local_t in feed.snapshot()),
                     key=lambda s: s.t)
    save_recording_csv(Path(args.file), samples)
    print(f"Wrote {len(samples)} samples.")
    return 0 if samples else 1


def cmd_replay(args) -> int:
    from .calibrate import load_recording_csv
    from .clench import detect_all

    cfg = _config(args)
    samples = load_recording_csv(Path(args.file))
    if len(samples) < 2:
        print("Recording is empty.")
        return 1
    events = detect_all(samples, cfg.clench)
    span = samples[-1].t - samples[0].t
    print(f"{len(samples)} samples over {span:.1f} s ({(len(samples) - 1) / span:.1f} Hz), "
          f"threshold {cfg.clench.k_on}x")
    for event in events:
        print(f"  {event.t_onset - samples[0].t:7.2f} s  peak {event.peak:5.1f}x  "
              f"{event.duration * 1000:3.0f} ms  turned {event.angle * 57.3:.1f} deg")
    print(f"{len(events)} clench(es).")
    return 0


def cmd_simulate(args) -> int:
    from .simulate import stream

    print(f"Sending synthetic motion to {args.host}:{args.port} for {args.seconds:g} s "
          f"(a clench every {args.clench_every:g} s). This is made-up data.")
    sent = stream(args.host, args.port, args.seconds, args.clench_every,
                  turn_every=args.turn_every)
    print(f"Sent {sent} packets.")
    return 0


# -- doctor ------------------------------------------------------------------

def cmd_doctor(args) -> int:
    from .app import gaze_model_path
    from .calibrate import SampleFeed
    from .imu import ImuServer, lan_address

    cfg = _config(args)
    problems = 0

    def check(label: str, ok: bool, detail: str = "", optional: bool = False) -> None:
        nonlocal problems
        problems += 0 if ok or optional else 1
        mark = "ok" if ok else "--" if optional else "!!"
        print(f"  [{mark}] {label}" + (f": {detail}" if detail else ""))

    print(f"jawcursor {__version__}, Python {sys.version.split()[0]} on {sys.platform}")
    try:
        import cv2
        import mediapipe
        import numpy
        import pynput  # noqa: F401

        check("packages", True, f"mediapipe {mediapipe.__version__}, opencv {cv2.__version__}, "
                                f"numpy {numpy.__version__}")
    except ImportError as error:
        check("packages", False, str(error))
        return 1

    width, height = screen_size()
    check("screen", width > 0 and height > 0, f"{width}x{height}")

    try:
        from .gaze import Camera, FaceTracker

        camera = Camera(cfg.camera)
        try:
            frame = None
            for _ in range(15):
                frame = camera.read()
            check("camera", frame is not None,
                  f"index {cfg.camera.index}, {frame.shape[1]}x{frame.shape[0]}"
                  if frame is not None else "opened but returned no frames")
            if frame is not None:
                from .gaze import is_dark

                check("camera image", not is_dark(frame),
                      "" if not is_dark(frame) else
                      "all black: open the privacy shutter or lens cover")
                tracker = FaceTracker()
                try:
                    feats = None
                    for _ in range(10):
                        shot = camera.read()
                        if shot is not None:
                            feats = tracker.process(shot, time.monotonic()) or feats
                    check("face in view", feats is not None,
                          "eyes tracked" if feats is not None else
                          "no face found (fine if nobody is at the desk)", optional=True)
                finally:
                    tracker.close()
        finally:
            camera.close()
    except Exception as error:
        check("camera", False, str(error))

    check("gaze calibration", gaze_model_path().exists(),
          str(gaze_model_path()) if gaze_model_path().exists() else "run: jawcursor calibrate gaze")
    check("clench calibration", config_path().exists(),
          str(config_path()) if config_path().exists()
          else "using defaults; run: jawcursor calibrate clench", optional=True)

    feed = SampleFeed()
    try:
        server = ImuServer(cfg.imu.host, cfg.imu.port, feed.on_sample)
        server.start()
        time.sleep(args.wait)
        rate = server.rate()
        server.stop()
        check("AirPods motion", len(feed) > 0,
              f"{rate:.0f} Hz via {server.source}" if len(feed)
              else f"nothing on port {cfg.imu.port} in {args.wait:.0f} s; point the relay at "
                   f"{lan_address()}:{cfg.imu.port}")
    except OSError as error:
        check("AirPods motion", False, f"could not listen on port {cfg.imu.port}: {error}")

    print("All good." if not problems else f"{problems} item(s) need attention.")
    return 0 if not problems else 1


# -- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jawcursor",
        description="Move the cursor with your eyes and click with a jaw clench.")
    parser.add_argument("--version", action="version", version=__version__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="config file (default: ~/.jawcursor/config.json)")
    net = argparse.ArgumentParser(add_help=False)
    net.add_argument("--port", type=int, help="motion port (default 7355)")
    net.add_argument("--host", help="address to listen on (default 0.0.0.0)")
    cam = argparse.ArgumentParser(add_help=False)
    cam.add_argument("--camera", type=int, help="camera index (default 0)")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common, net, cam], help="start controlling the cursor")
    run.add_argument("--dry-run", action="store_true", help="print clicks, leave the cursor alone")
    run.add_argument("--no-gaze", action="store_true",
                     help="clench clicks only; steer with your normal mouse")
    run.add_argument("--preview", action="store_true", help="show the camera with eye markers")
    run.add_argument("--right-click", action="store_true",
                     help="two quick clenches = right click (single clicks wait 0.6 s)")
    run.add_argument("--duration", type=float, help="stop after this many seconds")
    run.add_argument("--no-hotkeys", action="store_true", help=argparse.SUPPRESS)
    run.set_defaults(func=cmd_run)

    calibrate = sub.add_parser("calibrate", help="teach it your eyes or your clench")
    kinds = calibrate.add_subparsers(dest="what", required=True)
    gaze = kinds.add_parser("gaze", parents=[common, cam], help="follow dots on screen (30 s)")
    gaze.set_defaults(func=cmd_calibrate_gaze)
    clench = kinds.add_parser("clench", parents=[common, net], help="guided clench recording (1 min)")
    clench.add_argument("--clenches", type=int, default=10)
    clench.set_defaults(func=cmd_calibrate_clench)

    monitor = sub.add_parser("monitor", parents=[common, net],
                             help="live view of the motion signal and detections")
    monitor.add_argument("--seconds", type=float, help="stop after this long")
    monitor.set_defaults(func=cmd_monitor)

    record = sub.add_parser("record", parents=[common, net], help="save raw motion to a CSV")
    record.add_argument("file")
    record.add_argument("--seconds", type=float, default=30.0)
    record.set_defaults(func=cmd_record)

    replay = sub.add_parser("replay", parents=[common], help="run the detector over a CSV")
    replay.add_argument("file")
    replay.set_defaults(func=cmd_replay)

    simulate = sub.add_parser("simulate", help="send made-up motion data (no AirPods needed)")
    simulate.add_argument("--host", default="127.0.0.1")
    simulate.add_argument("--port", type=int, default=7355)
    simulate.add_argument("--seconds", type=float, default=20.0)
    simulate.add_argument("--clench-every", type=float, default=3.0)
    simulate.add_argument("--turn-every", type=float, default=0.0, help="add head turns")
    simulate.set_defaults(func=cmd_simulate)

    doctor = sub.add_parser("doctor", parents=[common, net, cam], help="check the setup")
    doctor.add_argument("--wait", type=float, default=3.0, help="seconds to wait for motion data")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    enable_dpi_awareness()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
