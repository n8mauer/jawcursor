import json
import socket
import time
import urllib.request

from jawcursor.imu import ClockMapper, ImuServer, parse_bridge_packet, parse_sensor_logger
from jawcursor.simulate import SyntheticImu, bridge_packet


def test_bridge_packet_round_trip():
    sample = SyntheticImu(seed=1).sample(12.5)
    parsed = parse_bridge_packet(bridge_packet(sample))
    assert parsed.t == 12.5
    assert parsed.location == "left"
    assert all(abs(a - b) < 1e-4 for a, b in zip(parsed.gyro, sample.gyro))
    assert all(abs(a - b) < 1e-4 for a, b in zip(parsed.accel, sample.accel))
    assert parsed.quat == (1.0, 0.0, 0.0, 0.0)


def test_packet_built_the_way_the_swift_app_builds_it_is_understood():
    # Mirrors the String(format:) call in apple/JawBridge/MotionStreamer.swift.
    numbers = ('"t":%.4f,"rr":[%.5f,%.5f,%.5f],"ua":[%.5f,%.5f,%.5f],'
               '"g":[%.4f,%.4f,%.4f],"q":[%.5f,%.5f,%.5f,%.5f]') % (
        81234.5678, 0.01, -0.02, 0.03, 0.001, 0.002, -0.003,
        0.0, -1.0, 0.0, 0.99, 0.01, 0.02, 0.03)
    packet = ('{"v":1,' + numbers + ',"loc":"' + "right" + '"}').encode()
    parsed = parse_bridge_packet(packet)
    assert parsed.t == 81234.5678
    assert parsed.gyro == (0.01, -0.02, 0.03)
    assert parsed.accel == (0.001, 0.002, -0.003)
    assert parsed.quat == (0.99, 0.01, 0.02, 0.03)
    assert parsed.location == "right"


def test_bridge_packet_rejects_junk():
    assert parse_bridge_packet(b"not json") is None
    assert parse_bridge_packet(b'{"t": 1}') is None
    assert parse_bridge_packet(b'{"t":1,"rr":[1,2],"ua":[0,0,0]}') is None
    assert parse_bridge_packet(b"[1,2,3]") is None


def sensor_logger_item(t_ns, rate=0.1, nested=True):
    values = {
        "devicelocation": "right", "roll": 0.0, "yaw": 0.0, "pitch": 0.0,
        "rotationRateX": rate, "rotationRateY": 0.2, "rotationRateZ": 0.3,
        "quaternionW": 1.0, "quaternionX": 0.0, "quaternionY": 0.0, "quaternionZ": 0.0,
        "gravityX": 0.0, "gravityY": -1.0, "gravityZ": 0.0,
        "accelerationX": 0.01, "accelerationY": 0.02, "accelerationZ": 0.03,
    }
    item = {"name": "headphone", "time": t_ns}
    if nested:
        item["values"] = values
    else:
        item.update(values)
    return item


def test_sensor_logger_body_is_parsed_and_sorted():
    body = {"messageId": 3, "sessionId": "s", "deviceId": "d", "payload": [
        sensor_logger_item(1_698_501_144_440_000_000, rate=0.5),
        {"name": "accelerometer", "time": 1_698_501_144_401_773_000,
         "values": {"x": 0, "y": 0, "z": 0}},
        sensor_logger_item(1_698_501_144_400_000_000, rate=0.4),
    ]}
    first, second = parse_sensor_logger(body)
    assert first.gyro == (0.4, 0.2, 0.3) and second.gyro == (0.5, 0.2, 0.3)
    assert abs(second.t - first.t - 0.04) < 1e-6
    assert first.accel == (0.01, 0.02, 0.03)
    assert first.location == "right"
    assert first.quat == (1.0, 0.0, 0.0, 0.0)


def test_sensor_logger_accepts_flat_items_and_ignores_garbage():
    assert len(parse_sensor_logger({"payload": [sensor_logger_item(10**18, nested=False)]})) == 1
    assert parse_sensor_logger({}) == []
    assert parse_sensor_logger({"payload": [{"name": "headphone", "time": 1}]}) == []
    assert parse_sensor_logger({"payload": ["x", None]}) == []


def test_clock_mapper_ignores_jitter_and_batching():
    mapper = ClockMapper()
    # Sender clock = local clock - 1000, samples every 40 ms, arriving late by
    # varying amounts and then in a batch.
    assert abs(mapper.to_local(5.00, 1005.00) - 1005.00) < 1e-9
    assert abs(mapper.to_local(5.04, 1005.09) - 1005.04) < 1e-9     # 50 ms late
    for t in (5.08, 5.12, 5.16):                                    # one batch
        assert abs(mapper.to_local(t, 1005.17) - (1000 + t)) < 1e-9


def test_clock_mapper_keeps_spacing_when_the_first_thing_seen_is_a_batch():
    mapper = ClockMapper()
    local = mapper.batch_to_local([5.00, 5.04, 5.08], 1005.10)
    assert [round(t, 6) for t in local] == [1005.02, 1005.06, 1005.10]
    assert mapper.batch_to_local([], 1006.0) == []


def test_clock_mapper_resets_when_the_sender_restarts():
    mapper = ClockMapper()
    mapper.to_local(5.0, 1005.0)
    assert abs(mapper.to_local(0.5, 1010.0) - 1010.0) < 1e-9


def test_server_reports_a_taken_port_clearly():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        server = ImuServer("127.0.0.1", blocker.getsockname()[1], lambda s, t: None)
        try:
            server.start()
            raise AssertionError("expected the port to be refused")
        except OSError as error:
            assert "--port" in str(error)


def collect_server():
    got = []
    server = ImuServer("127.0.0.1", 0, lambda sample, local_t: got.append((sample, local_t)))
    server.start()
    return server, got


def wait_for(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_server_receives_udp_packets():
    server, got = collect_server()
    try:
        imu = SyntheticImu(seed=2)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        begin = time.monotonic()
        stamps = []
        for _ in range(20):                             # sent in real time, like the app
            stamps.append(100.0 + time.monotonic() - begin)     # sender clock starts at 100
            sock.sendto(bridge_packet(imu.sample(stamps[-1])), ("127.0.0.1", server.port))
            time.sleep(0.04)
        sock.sendto(b"garbage", ("127.0.0.1", server.port))
        sock.close()
        assert wait_for(lambda: len(got) == 20)
        assert server.source == "jawbridge" and server.count == 20
        assert server.is_live()
        assert server.rate() > 5
        times = [local_t for _, local_t in got]
        assert all(b > a for a, b in zip(times, times[1:]))
        # Spacing between samples survives, and they sit on the local clock.
        assert abs((times[-1] - times[0]) - (stamps[-1] - stamps[0])) < 1e-3
        assert abs(times[0] - begin) < 0.25
    finally:
        server.stop()


def test_server_receives_sensor_logger_posts():
    server, got = collect_server()
    try:
        base = 1_698_501_144_000_000_000
        body = {"messageId": 0, "sessionId": "s", "deviceId": "d",
                "payload": [sensor_logger_item(base + i * 40_000_000) for i in range(5)]}
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/data", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.status == 200
        assert wait_for(lambda: len(got) == 5)
        assert server.source == "sensor-logger"
        times = [local_t for _, local_t in got]
        assert all(abs((b - a) - 0.04) < 1e-6 for a, b in zip(times, times[1:]))
        assert abs(times[-1] - time.monotonic()) < 1.0     # newest sample = "now"

        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/", timeout=3) as response:
            assert json.loads(response.read())["samples"] == 5

        bad = urllib.request.Request(f"http://127.0.0.1:{server.port}/data", data=b"{nope")
        try:
            urllib.request.urlopen(bad, timeout=3)
            raise AssertionError("expected HTTP 400")
        except urllib.error.HTTPError as error:
            assert error.code == 400
    finally:
        server.stop()
