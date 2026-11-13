import Combine
import CoreMotion
import Foundation
import Network
#if os(iOS)
import UIKit
#endif

/// Reads AirPods motion through CMHeadphoneMotionManager and sends every
/// sample to jawcursor as one small JSON datagram over UDP.
///
/// Wire format (one object per datagram):
///   {"v":1,"t":<seconds>,"rr":[x,y,z],"ua":[x,y,z],"g":[x,y,z],"q":[w,x,y,z],"loc":"left"}
///   rr = rotation rate (rad/s), ua = user acceleration (g), g = gravity (g),
///   q = attitude quaternion, loc = which earbud supplied the sample.
///
/// Everything here runs on the main thread: motion updates are delivered to
/// the main queue and the UI calls start/stop from the main thread.
final class MotionStreamer: ObservableObject {
    @Published var isStreaming = false
    @Published var status = "Not streaming."
    @Published var packetsSent = 0
    @Published var samplesPerSecond = 0
    @Published var rotationRate = 0.0
    @Published var sensorLocation = "-"

    private let manager = CMHeadphoneMotionManager()
    private var connection: NWConnection?
    private var windowStart = Date()
    private var windowCount = 0

    func start(host: String, port: String) {
        let address = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !address.isEmpty,
              let number = UInt16(port.trimmingCharacters(in: .whitespacesAndNewlines)),
              let endpointPort = NWEndpoint.Port(rawValue: number) else {
            status = "Enter the laptop's IP address and port first."
            return
        }
        guard manager.isDeviceMotionAvailable else {
            status = "Headphone motion is not available on this device."
            return
        }
        let permission = CMHeadphoneMotionManager.authorizationStatus()
        if permission == .denied || permission == .restricted {
            status = "Motion access is off. Turn it on under Privacy & Security > Motion & Fitness."
            return
        }

        let connection = NWConnection(host: NWEndpoint.Host(address), port: endpointPort, using: .udp)
        connection.start(queue: .main)
        self.connection = connection

        packetsSent = 0
        samplesPerSecond = 0
        windowStart = Date()
        windowCount = 0
        status = "Waiting for AirPods motion. Put them in your ears."
        isStreaming = true
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = true     // keep the screen on
        #endif

        manager.startDeviceMotionUpdates(to: OperationQueue.main) { [weak self] motion, error in
            guard let self = self else { return }
            if let motion = motion {
                self.send(motion)
            } else if let error = error {
                self.status = "Motion error: \(error.localizedDescription)"
            }
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        connection?.cancel()
        connection = nil
        isStreaming = false
        status = "Not streaming."
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = false
        #endif
    }

    private func send(_ motion: CMDeviceMotion) {
        let rate = motion.rotationRate
        let acceleration = motion.userAcceleration
        let gravity = motion.gravity
        let attitude = motion.attitude.quaternion

        let location: String
        switch motion.sensorLocation {
        case .headphoneLeft: location = "left"
        case .headphoneRight: location = "right"
        default: location = "default"
        }

        let numbers = String(
            format: "\"t\":%.4f,\"rr\":[%.5f,%.5f,%.5f],\"ua\":[%.5f,%.5f,%.5f],"
                + "\"g\":[%.4f,%.4f,%.4f],\"q\":[%.5f,%.5f,%.5f,%.5f]",
            motion.timestamp,
            rate.x, rate.y, rate.z,
            acceleration.x, acceleration.y, acceleration.z,
            gravity.x, gravity.y, gravity.z,
            attitude.w, attitude.x, attitude.y, attitude.z)
        let json = "{\"v\":1," + numbers + ",\"loc\":\"" + location + "\"}"
        connection?.send(content: Data(json.utf8), completion: .idempotent)

        packetsSent += 1
        windowCount += 1
        let elapsed = Date().timeIntervalSince(windowStart)
        if elapsed >= 1.0 {
            samplesPerSecond = Int((Double(windowCount) / elapsed).rounded())
            windowStart = Date()
            windowCount = 0
            rotationRate = (rate.x * rate.x + rate.y * rate.y + rate.z * rate.z).squareRoot()
            sensorLocation = location
            status = "Streaming."
        }
    }
}
