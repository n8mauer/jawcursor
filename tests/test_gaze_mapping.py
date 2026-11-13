from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jawcursor.calibrate import dot_sequence, fit_gaze
from jawcursor.gaze import (FEATURE_FLOORS, FEATURE_NAMES, L_CORNERS, L_IRIS, L_LIDS,
                            R_CORNERS, R_IRIS, R_LIDS, extract_features)
from jawcursor.mapping import GazeMapper, clean_dot_samples

W, H = 1280, 720


def fake_landmarks(r_iris=(0.0, 0.0), l_iris=(0.0, 0.0), opening=10.0, roll_deg=0.0):
    """478 landmarks with two 40 px wide eyes; iris offsets are in pixels."""
    pts = np.zeros((478, 2))
    eyes = [((560.0, 300.0), R_CORNERS, R_LIDS, R_IRIS, r_iris),
            ((720.0, 300.0), L_CORNERS, L_LIDS, L_IRIS, l_iris)]
    for centre, corners, lids, iris, offset in eyes:
        c = np.array(centre)
        pts[corners[0]] = c + (-20, 0)
        pts[corners[1]] = c + (20, 0)
        pts[lids[0]] = c + (0, -opening / 2)
        pts[lids[1]] = c + (0, opening / 2)
        pts[iris] = c + offset
    angle = np.radians(roll_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = (pts - (640, 300)) @ rot.T + (640, 300)
    return [SimpleNamespace(x=x / W, y=y / H, z=0.0) for x, y in pts]


def feature(vector, name):
    return vector[FEATURE_NAMES.index(name)]


def test_feature_lists_line_up():
    assert len(FEATURE_NAMES) == len(FEATURE_FLOORS) == 14


def test_iris_offsets_are_measured_in_eye_widths():
    feats = extract_features(fake_landmarks(r_iris=(4, 0), l_iris=(-8, 2)), {}, None, W, H)
    assert feature(feats.vector, "r_iris_h") == pytest.approx(0.1)
    assert feature(feats.vector, "r_iris_v") == pytest.approx(0.0, abs=1e-9)
    assert feature(feats.vector, "l_iris_h") == pytest.approx(-0.2)
    assert feature(feats.vector, "l_iris_v") == pytest.approx(0.05)
    assert feature(feats.vector, "r_open") == pytest.approx(0.25)
    assert feats.iris_px[0] == pytest.approx((564.0, 300.0))


def test_iris_offsets_do_not_change_when_the_head_tilts():
    level = extract_features(fake_landmarks(r_iris=(4, 2), l_iris=(4, 2)), {}, None, W, H)
    tilted = extract_features(fake_landmarks(r_iris=(4, 2), l_iris=(4, 2), roll_deg=15),
                              {}, None, W, H)
    assert tilted.vector[:4] == pytest.approx(level.vector[:4], abs=1e-9)


def test_blendshapes_and_pose_are_read():
    blend = {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.6,
             "eyeLookOutLeft": 0.5, "eyeLookInRight": 0.3,
             "eyeLookUpLeft": 0.2, "eyeLookUpRight": 0.4, "eyeLookDownLeft": 0.1}
    yaw = np.radians(10)
    matrix = np.eye(4)
    matrix[0, 0], matrix[0, 2], matrix[2, 0], matrix[2, 2] = np.cos(yaw), np.sin(yaw), -np.sin(yaw), np.cos(yaw)
    matrix[:3, 3] = (5.0, -20.0, -600.0)
    feats = extract_features(fake_landmarks(), blend, matrix, W, H)
    assert feats.blink == pytest.approx(0.7)
    assert feature(feats.vector, "look_h") == pytest.approx(0.4)
    assert feature(feats.vector, "look_v") == pytest.approx(0.25)
    assert feature(feats.vector, "yaw") == pytest.approx(yaw)
    assert feature(feats.vector, "pitch") == pytest.approx(0.0, abs=1e-9)
    assert feature(feats.vector, "tz") == pytest.approx(-60.0)


def synthetic_calibration(noise=0.004, seed=0, curved=False, frames=30):
    """Features that depend on where the user looks, plus measurement noise."""
    rng = np.random.default_rng(seed)
    dots = dot_sequence()
    per_dot = []
    for x, y in dots:
        gx, gy = x - 0.5, y - 0.5
        if curved:
            gx, gy = gx + 0.4 * gx * gx, gy - 0.3 * gy * gy
        clean = np.array([0.30 * gx, 0.12 * gy, 0.28 * gx, 0.11 * gy,       # iris
                          0.8 * gx, -0.7 * gy,                              # look scores
                          0.25 - 0.2 * gy, 0.25 - 0.2 * gy,                 # lid opening
                          0.10 * gx, 0.06 * gy, 0.0,                        # head pose
                          0.0, 0.0, -6.0])
        rows = clean + rng.normal(0, noise, size=(frames, len(clean)))
        per_dot.append(rows)
    return per_dot, dots


def test_fit_recovers_the_mapping_and_reports_held_out_error():
    per_dot, dots = synthetic_calibration()
    mapper, report = fit_gaze(per_dot, dots, aspect=16 / 9)
    assert report.points == 13
    assert report.median_error < 0.03
    for rows, dot in zip(per_dot, dots):
        x, y = mapper.predict(rows.mean(axis=0))
        assert abs(x - dot[0]) < 0.03 and abs(y - dot[1]) < 0.05


def test_quadratic_model_is_chosen_when_the_relation_is_curved():
    per_dot, dots = synthetic_calibration(noise=0.001, curved=True)
    _, report = fit_gaze(per_dot, dots, aspect=16 / 9)
    assert report.kind == "quadratic"
    assert report.median_error < 0.03


def test_more_noise_gives_a_worse_reported_error():
    quiet = fit_gaze(*synthetic_calibration(noise=0.002), aspect=16 / 9)[1]
    noisy = fit_gaze(*synthetic_calibration(noise=0.03), aspect=16 / 9)[1]
    assert noisy.median_error > 2 * quiet.median_error


def test_predictions_are_clamped_to_the_screen():
    per_dot, dots = synthetic_calibration()
    mapper, _ = fit_gaze(per_dot, dots, aspect=16 / 9)
    wild = per_dot[0][0].copy()
    wild[:6] *= 50
    x, y = mapper.predict(wild)
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_unmoving_features_do_not_amplify_noise():
    # Head never moved during calibration; a later 2 degree head turn must not
    # throw the cursor across the screen.
    per_dot, dots = synthetic_calibration()
    for rows in per_dot:
        rows[:, 8:11] = np.random.default_rng(1).normal(0, 1e-4, size=(len(rows), 3))
    mapper, _ = fit_gaze(per_dot, dots, aspect=16 / 9)
    centre = per_dot[0].mean(axis=0)
    nudged = centre.copy()
    nudged[8] += np.radians(2)
    (x0, y0), (x1, y1) = mapper.predict(centre), mapper.predict(nudged)
    assert abs(x1 - x0) < 0.15 and abs(y1 - y0) < 0.15


def test_fit_fails_cleanly_with_too_few_usable_dots():
    per_dot, dots = synthetic_calibration()
    sparse = [rows if i < 3 else rows[:2] for i, rows in enumerate(per_dot)]
    with pytest.raises(ValueError):
        fit_gaze(sparse, dots, aspect=16 / 9)


def test_blink_frames_are_dropped_from_a_dot():
    rows = synthetic_calibration()[0][0].copy()
    rows[3, :4] += 0.5
    cleaned = clean_dot_samples(rows)
    assert len(cleaned) == len(rows) - 1


def test_save_and_load_round_trip(tmp_path):
    per_dot, dots = synthetic_calibration()
    mapper, report = fit_gaze(per_dot, dots, aspect=16 / 9)
    path = tmp_path / "gaze.json"
    mapper.save(path, report, (1920, 1080))
    loaded = GazeMapper.load(path)
    sample = per_dot[4][0]
    assert loaded.predict(sample) == pytest.approx(mapper.predict(sample))


def test_dot_sequence_covers_the_screen_and_starts_in_the_centre():
    dots = dot_sequence()
    assert dots[0] == (0.5, 0.5) and len(dots) == len(set(dots)) == 13
    xs, ys = [d[0] for d in dots], [d[1] for d in dots]
    assert min(xs) < 0.1 and max(xs) > 0.9 and min(ys) < 0.1 and max(ys) > 0.9
    assert dot_sequence() == dots                       # same order every time


PORTRAIT = Path(__file__).resolve().parent.parent / ".scratch" / "portrait.jpg"


@pytest.mark.skipif(not PORTRAIT.exists(), reason="sample photo not present")
def test_real_face_photo_goes_through_the_whole_tracker():
    import cv2

    from jawcursor.gaze import FaceTracker

    frame = cv2.imread(str(PORTRAIT))
    tracker = FaceTracker()
    try:
        tracker.process(frame, 0.0)
        feats = tracker.process(frame, 0.033)
    finally:
        tracker.close()
    assert feats is not None and feats.vector.shape == (14,)
    assert np.all(np.isfinite(feats.vector))
    assert abs(feature(feats.vector, "r_iris_h")) < 0.3     # iris sits inside the eye
    assert abs(feature(feats.vector, "l_iris_h")) < 0.3
    assert 0.0 <= feats.blink <= 1.0
    assert abs(feature(feats.vector, "yaw")) < 0.3          # roughly frontal portrait
    assert feature(feats.vector, "tz") < 0                  # in front of the camera
    height, width = frame.shape[:2]
    for x, y in feats.iris_px:
        assert 0 < x < width and 0 < y < height
