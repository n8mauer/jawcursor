import socket
import threading

from jawcursor.app import ClenchPipeline, run
from jawcursor.calibrate import (SampleFeed, describe_clench_fit, load_recording_csv,
                                 run_clench_calibration, save_recording_csv)
from jawcursor.clench import detect_all
from jawcursor.config import Config, load_config, save_config
from jawcursor.simulate import SyntheticImu, stream


def test_pipeline_turns_clenches_into_clicks_with_local_onset_times():
    imu = SyntheticImu(seed=30)
    for t in (3.0, 6.0):
        imu.clench(t)
    actions = []
    pipeline = ClenchPipeline(Config(), actions.append)
    offset = 5000.0                                     # local clock = sender clock + 5000
    for sample in imu.run(9.0):
        pipeline.on_sample(sample, sample.t + offset)
    assert [a.button for a in actions] == ["left", "left"]
    assert abs(actions[0].t_event - (offset + 3.0)) < 0.15
    assert abs(actions[1].t_event - (offset + 6.0)) < 0.15
    assert pipeline.events == 2
    assert pipeline.take_peak() > 6.0 and pipeline.take_peak() == 0.0


def test_pipeline_right_click_mode_needs_poll_for_single_clicks():
    cfg = Config()
    cfg.gesture.double_action = "right_click"
    imu = SyntheticImu(seed=31)
    imu.clench(3.0)                                     # single
    imu.clench(8.0)
    imu.clench(8.45)                                    # pair
    actions = []
    pipeline = ClenchPipeline(cfg, actions.append)
    for sample in imu.run(12.0):
        pipeline.on_sample(sample, sample.t)
        pipeline.poll(sample.t)
    assert [a.button for a in actions] == ["left", "right"]


def test_config_round_trip_and_unknown_keys(tmp_path):
    cfg = Config()
    cfg.clench.k_on = 11.5
    cfg.gesture.double_action = "right_click"
    cfg.camera.index = 2
    path = save_config(cfg, tmp_path / "config.json")
    loaded = load_config(path)
    assert loaded.clench.k_on == 11.5
    assert loaded.gesture.double_action == "right_click"
    assert loaded.camera.index == 2
    path.write_text('{"clench": {"k_on": 9, "bogus": 1}, "nonsense": {}}', encoding="utf-8")
    assert load_config(path).clench.k_on == 9.0
    assert load_config(tmp_path / "missing.json").clench.k_on == Config().clench.k_on


def test_recording_csv_round_trip(tmp_path):
    samples = SyntheticImu(seed=32).run(2.0)
    path = tmp_path / "rec.csv"
    save_recording_csv(path, samples)
    loaded = load_recording_csv(path)
    assert len(loaded) == len(samples)
    assert loaded[7].gyro == samples[7].gyro and loaded[7].location == "left"
    assert abs(loaded[7].t - samples[7].t) < 1e-4


class FakeTime:
    """A clock that only advances when the routine sleeps, feeding samples as it goes."""

    def __init__(self, imu: SyntheticImu, feed: SampleFeed, rate: float = 25.0):
        self.now = 0.0
        self._imu, self._feed, self._step = imu, feed, 1.0 / rate
        self._next = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 1e-3)
        while self._next <= self.now:
            # Sender clock differs from the local one by a fixed amount.
            sample = self._imu.sample(self._next)
            shifted = type(sample)(t=sample.t + 777.0, gyro=sample.gyro, accel=sample.accel,
                                   quat=sample.quat, location=sample.location)
            self._feed.on_sample(shifted, self._next)
            self._next += self._step


def test_guided_clench_calibration_end_to_end():
    said = []
    # First pass with an empty world just to learn when the cues fall.
    probe_feed = SampleFeed()
    probe_time = FakeTime(SyntheticImu(seed=40), probe_feed)
    result = run_clench_calibration(probe_feed, say=said.append,
                                    clock=probe_time.clock, sleep=probe_time.sleep)
    _, probe_report, probe_recording = result
    assert not probe_report["ok"]                       # nobody clenched
    assert "barely show up" in describe_clench_fit(probe_report)
    cues = probe_recording.cues
    assert len(cues) == 10

    # Second pass: a person who moves during phase 2 and clenches after each cue.
    imu = SyntheticImu(seed=41)
    (move_start, _move_end) = probe_recording.negative_spans[1]
    for i in range(5):
        imu.head_turn(move_start + 1.0 + 2.0 * i, duration=0.7, peak=1.1)
    for i, cue in enumerate(cues):
        imu.clench(cue + 0.3 + 0.07 * (i % 3))
    feed = SampleFeed()
    fake = FakeTime(imu, feed)
    params, report, recording = run_clench_calibration(feed, say=said.append,
                                                       clock=fake.clock, sleep=fake.sleep)
    assert recording.cues == cues                       # same timing both passes
    assert report["ok"], report
    assert report["hits"] >= 9 and report["false_alarms"] == 0
    assert len(detect_all(recording.samples, params)) >= 9
    assert "Detected" in describe_clench_fit(report)
    assert any("CLENCH" in line for line in said)


def test_calibration_gives_up_when_no_data_arrives():
    feed = SampleFeed()
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    said = []
    assert run_clench_calibration(feed, say=said.append, clock=lambda: now[0], sleep=sleep) is None
    assert "No motion data" in said[-1]


def free_port() -> int:
    """A port number that is free for UDP and for TCP (Windows reserves some)."""
    for _ in range(25):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.bind(("127.0.0.1", 0))
            port = udp.getsockname()[1]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
                try:
                    tcp.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
    raise RuntimeError("no port free for both UDP and TCP")


def test_run_without_gaze_clicks_on_streamed_clenches(capsys):
    cfg = Config()
    cfg.imu.host, cfg.imu.port = "127.0.0.1", free_port()
    sender = threading.Thread(target=stream, args=("127.0.0.1", cfg.imu.port, 6.5, 2.0),
                              daemon=True)
    sender.start()
    code = run(cfg, dry_run=True, use_gaze=False, duration=7.5, hotkeys=False)
    sender.join(timeout=5)
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("[dry run] left click") == 3       # clenches at 2, 4 and 6 s
    assert "3 click(s), 3 clench(es)" in out
