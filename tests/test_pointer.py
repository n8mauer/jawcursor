import math

from jawcursor.config import PointerParams
from jawcursor.filters import OneEuro
from jawcursor.pointer import DryRunMouse, PointerController

SCREEN = (2000, 1000)


def make(params=None, **kwargs):
    mouse = DryRunMouse(start=(1000, 500))
    return mouse, PointerController(mouse, SCREEN, params or PointerParams(), **kwargs)


def settle(pointer, nx, ny, t0, seconds=3.0, fps=30):
    pos = None
    for i in range(int(seconds * fps)):
        pos = pointer.update_gaze(nx, ny, t0 + i / fps) or pos
    return pos


def test_cursor_converges_near_the_gaze_point():
    mouse, pointer = make()
    settle(pointer, 0.5, 0.5, 0.0, seconds=1.0)
    pos = settle(pointer, 0.8, 0.3, 1.0)
    leash = PointerParams().deadzone * SCREEN[0]
    assert math.dist(pos, (1600, 300)) <= leash + 2
    assert mouse.position() == pos


def test_small_gaze_wander_does_not_move_the_cursor():
    _, pointer = make()
    start = settle(pointer, 0.5, 0.5, 0.0)
    positions = set()
    for i in range(90):
        wobble = 0.004 * math.sin(i * 1.7)              # +-8 px, inside the leash
        positions.add(pointer.update_gaze(0.5 + wobble, 0.5 - wobble, 3.0 + i / 30))
    assert positions == {start}


def test_cursor_stays_on_screen():
    _, pointer = make()
    pos = settle(pointer, 1.0, 1.0, 0.0)
    assert pos[0] <= SCREEN[0] - 1 and pos[1] <= SCREEN[1] - 1
    pos = settle(pointer, 0.0, 0.0, 10.0)
    assert pos[0] >= 0 and pos[1] >= 0


def test_click_lands_where_the_cursor_was_before_the_clench():
    mouse, pointer = make()
    target = settle(pointer, 0.25, 0.25, 0.0)           # resting on the target until t=3
    for i in range(12):                                 # clench at t=3 drags the estimate away
        pointer.update_gaze(0.6, 0.6, 3.0 + i / 30)
    assert mouse.position() != target
    clicked = pointer.click(t_event=3.0, t=3.4)
    assert clicked == target
    assert mouse.clicks == [("left", target)]


def test_cursor_holds_still_right_after_a_click():
    mouse, pointer = make()
    target = settle(pointer, 0.25, 0.25, 0.0)
    pointer.click(t_event=3.0, t=3.0)
    assert pointer.update_gaze(0.9, 0.9, 3.1) is None
    assert mouse.position() == target
    assert pointer.update_gaze(0.9, 0.9, 3.0 + PointerParams().freeze_after_click + 0.05)


def test_second_click_of_a_double_click_hits_the_same_spot():
    mouse, pointer = make()
    target = settle(pointer, 0.25, 0.25, 0.0)
    pointer.click(t_event=3.0, t=3.1)
    for i in range(15):
        pointer.update_gaze(0.3, 0.3, 3.1 + i / 30)
    pointer.click(t_event=3.5, t=3.6)
    assert mouse.clicks == [("left", target), ("left", target)]


def test_right_button_is_passed_through():
    mouse, pointer = make()
    settle(pointer, 0.5, 0.5, 0.0)
    pointer.click(t_event=3.0, button="right", t=3.0)
    assert mouse.clicks[0][0] == "right"


def test_moving_the_real_mouse_takes_control_back():
    mouse, pointer = make()
    settle(pointer, 0.5, 0.5, 0.0)
    mouse._pos = (100, 100)                             # the user grabbed the mouse
    before = len(mouse.moves)
    assert pointer.update_gaze(0.9, 0.9, 3.1) is None
    assert pointer.click(t_event=3.2, t=3.2) is None
    assert pointer.yielding(3.2)
    assert len(mouse.moves) == before and mouse.clicks == []
    resumed = pointer.update_gaze(0.9, 0.9, 3.1 + PointerParams().yield_seconds + 0.1)
    assert resumed is not None


def test_pause_stops_everything():
    mouse, pointer = make()
    settle(pointer, 0.5, 0.5, 0.0)
    assert pointer.toggle_pause() is True
    before = len(mouse.moves)
    assert pointer.update_gaze(0.1, 0.1, 4.0) is None
    assert pointer.click(t_event=4.0, t=4.0) is None
    assert len(mouse.moves) == before and mouse.clicks == []
    assert pointer.toggle_pause() is False
    assert pointer.update_gaze(0.1, 0.1, 5.0) is not None


def test_hold_freezes_the_cursor():
    mouse, pointer = make()
    start = settle(pointer, 0.5, 0.5, 0.0)
    pointer.hold(0.5, t=3.0)
    assert pointer.update_gaze(0.9, 0.9, 3.2) is None
    assert mouse.position() == start


def test_without_gaze_clicks_happen_at_the_current_mouse_position():
    mouse, pointer = make(yield_to_mouse=False)
    mouse._pos = (321, 654)
    assert pointer.click(t_event=1.0, t=1.0) == (321, 654)
    mouse._pos = (10, 20)                               # user moved; no yielding in this mode
    assert pointer.click(t_event=1.3, t=1.3) == (10, 20)
    assert mouse.clicks == [("left", (321, 654)), ("left", (10, 20))]


def test_one_euro_smooths_noise_but_follows_steps():
    import random

    rng = random.Random(0)
    f = OneEuro(min_cutoff=0.5, beta=1.0)
    out = [f(0.5 + rng.gauss(0, 0.01), i / 30) for i in range(300)]
    tail = out[100:]
    spread = max(tail) - min(tail)
    assert spread < 0.03                                # raw spread is about 0.06
    stepped = [f(0.9, 10 + i / 30) for i in range(30)]
    assert stepped[-1] > 0.85                           # there within a second
