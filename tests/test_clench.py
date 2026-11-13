import math

import pytest

from jawcursor.clench import (ClenchDetector, ClenchRecording, GestureResolver,
                              detect_all, fit_clench_params)
from jawcursor.config import ClenchParams, GestureParams
from jawcursor.simulate import SyntheticImu


def onsets(events):
    return [round(ev.t_onset, 2) for ev in events]


def near(events, t, tol=0.2):
    return [ev for ev in events if abs(ev.t_onset - t) <= tol]


@pytest.mark.parametrize("rate", [25.0, 50.0, 100.0])
def test_quick_clenches_are_each_detected_once(rate):
    imu = SyntheticImu(seed=3)
    times = [3.0, 5.0, 7.3, 9.1]
    for t in times:
        imu.clench(t)
    events = detect_all(imu.run(12.0, rate=rate), ClenchParams())
    assert len(events) == len(times), onsets(events)
    for t in times:
        assert len(near(events, t)) == 1


def test_onset_is_reported_close_to_the_true_start():
    imu = SyntheticImu(seed=4)
    imu.clench(4.0)
    (event,) = detect_all(imu.run(8.0), ClenchParams())
    assert 3.95 <= event.t_onset <= 4.12
    assert event.t_end > event.t_onset


def test_nothing_fires_on_noise_alone():
    assert detect_all(SyntheticImu(seed=5).run(120.0), ClenchParams()) == []


def test_head_turns_do_not_click():
    imu = SyntheticImu(seed=6)
    for t, duration, peak in [(3.0, 0.8, 1.2), (6.0, 0.4, 2.0), (9.0, 1.5, 0.6), (12.0, 0.25, 1.0)]:
        imu.head_turn(t, duration, peak)
    assert detect_all(imu.run(16.0), ClenchParams()) == []


def test_clench_after_a_head_turn_still_works():
    imu = SyntheticImu(seed=7)
    imu.head_turn(3.0)
    imu.clench(6.0)
    events = detect_all(imu.run(9.0), ClenchParams())
    assert len(events) == 1 and len(near(events, 6.0)) == 1


def test_clench_during_a_head_turn_is_ignored():
    imu = SyntheticImu(seed=8)
    imu.head_turn(3.0, duration=1.2, peak=1.5)
    imu.clench(3.6)
    assert detect_all(imu.run(8.0), ClenchParams()) == []


def test_release_of_a_held_clench_is_not_a_second_click():
    imu = SyntheticImu(seed=9)
    imu.clench(3.0, hold=0.9)
    events = detect_all(imu.run(8.0), ClenchParams())
    assert len(events) == 1 and len(near(events, 3.0)) == 1


def test_without_release_suppression_the_release_shows_up():
    imu = SyntheticImu(seed=9)
    imu.clench(3.0, hold=0.9)
    events = detect_all(imu.run(8.0), ClenchParams(suppress_release=False))
    assert len(events) == 2


def test_two_quick_clenches_both_count():
    imu = SyntheticImu(seed=10)
    imu.clench(3.0)
    imu.clench(3.55)
    events = detect_all(imu.run(7.0), ClenchParams())
    assert len(events) == 2
    assert events[1].t_onset - events[0].t_onset < 0.75


def test_detector_recovers_after_a_gap_in_the_stream():
    imu = SyntheticImu(seed=11)
    imu.clench(3.0)
    imu.clench(23.0)
    samples = imu.run(5.0) + imu.run(5.0, start=20.0)
    events = detect_all(samples, ClenchParams())
    assert len(events) == 2


def test_duplicate_and_out_of_order_packets_are_ignored():
    imu = SyntheticImu(seed=15)
    imu.clench(3.0)
    samples = imu.run(6.0)
    shuffled = []
    for i, sample in enumerate(samples):
        shuffled.append(sample)
        if i % 7 == 3:
            shuffled.append(samples[i - 1])     # a late packet arrives again
            shuffled.append(sample)             # and a duplicate
    events = detect_all(shuffled, ClenchParams())
    assert len(events) == 1 and len(near(events, 3.0)) == 1


def test_noise_level_change_does_not_cause_a_storm():
    quiet = SyntheticImu(seed=12).run(10.0)
    loud = SyntheticImu(seed=13, gyro_noise=0.03, accel_noise=0.02).run(30.0, start=10.0)
    events = detect_all(quiet + loud, ClenchParams())
    assert len(events) <= 1          # at most the moment of the step itself


def test_weak_signal_stays_below_threshold():
    imu = SyntheticImu(seed=14, clench_gyro=0.012, clench_accel=0.002)
    imu.clench(3.0)
    assert detect_all(imu.run(6.0), ClenchParams()) == []


def test_gesture_default_clicks_immediately():
    resolver = GestureResolver(GestureParams())
    assert [a.button for a in resolver.on_clench(1.0)] == ["left"]
    assert [a.button for a in resolver.on_clench(1.3)] == ["left"]
    assert resolver.poll(5.0) == []


def test_gesture_right_click_mode():
    resolver = GestureResolver(GestureParams(double_action="right_click", double_window=0.6))
    assert resolver.on_clench(1.0) == []
    assert resolver.poll(1.3) == []
    (action,) = resolver.poll(1.7)                  # lone clench becomes a left click
    assert (action.button, action.t_event) == ("left", 1.0)

    assert resolver.on_clench(5.0) == []
    (action,) = resolver.on_clench(5.4)             # second one in time: right click
    assert (action.button, action.t_event) == ("right", 5.0)
    assert resolver.poll(9.0) == []

    assert resolver.on_clench(10.0) == []
    (late,) = resolver.on_clench(11.5)              # too late: first resolves as left
    assert (late.button, late.t_event) == ("left", 10.0)
    (last,) = resolver.poll(12.5)
    assert (last.button, last.t_event) == ("left", 11.5)


def make_recording(seed=20, strength=1.0, **imu_kwargs):
    imu = SyntheticImu(seed=seed, **imu_kwargs)
    for t in (10.0, 13.0, 16.0, 19.0):
        imu.head_turn(t, duration=0.7, peak=1.0)
    cues = [24.0 + 2.5 * i for i in range(10)]
    for i, cue in enumerate(cues):
        imu.clench(cue + 0.35 + 0.05 * (i % 4), strength=strength)
    samples = imu.run(50.0)
    return ClenchRecording(samples=samples, cues=cues, negative_spans=[(1.0, 22.0)])


def test_fit_finds_thresholds_that_separate_clenches_from_movement():
    rec = make_recording()
    params, report = fit_clench_params(rec)
    assert report["ok"], report
    assert report["recall"] >= 0.9
    assert report["false_alarms"] == 0
    assert 3.0 <= params.k_on <= 42.0
    events = detect_all(rec.samples, params)
    assert len(events) == 10


def test_fit_reports_failure_when_there_is_no_clench_signal():
    rec = make_recording(strength=0.0)
    params, report = fit_clench_params(rec)
    assert not report["ok"]
    assert params == ClenchParams()


def test_fit_adapts_to_a_different_acceleration_unit():
    # Same motion expressed in m/s^2 instead of g: thresholds are relative.
    rec = make_recording(seed=21, accel_noise=0.002 * 9.81, clench_accel=0.03 * 9.81)
    _, report = fit_clench_params(rec)
    assert report["ok"] and report["false_alarms"] == 0


def test_event_fields_are_sane():
    imu = SyntheticImu(seed=22)
    imu.clench(3.0)
    (event,) = detect_all(imu.run(6.0), ClenchParams())
    assert event.peak >= ClenchParams().k_on
    assert 0.0 < event.duration <= ClenchParams().max_duration
    assert 0.0 <= event.angle <= ClenchParams().max_angle
    assert math.isfinite(event.peak)


def test_reset_clears_state():
    detector = ClenchDetector()
    imu = SyntheticImu(seed=23)
    imu.clench(3.0)
    assert sum(detector.update(s) is not None for s in imu.run(6.0)) == 1
    detector.reset()
    assert detector.state == ClenchDetector.IDLE
    assert sum(detector.update(s) is not None for s in imu.run(6.0)) == 1
