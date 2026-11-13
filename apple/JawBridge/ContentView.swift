import SwiftUI

struct ContentView: View {
    @StateObject private var streamer = MotionStreamer()
    @AppStorage("host") private var host = ""
    @AppStorage("port") private var port = "7355"

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("JawBridge")
                .font(.largeTitle.bold())
            Text("Sends AirPods motion to jawcursor on your laptop.")
                .foregroundStyle(.secondary)

            TextField("Laptop IP address (127.0.0.1 on the same Mac)", text: $host)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
            TextField("Port", text: $port)
                .textFieldStyle(.roundedBorder)

            Button(streamer.isStreaming ? "Stop" : "Start streaming") {
                if streamer.isStreaming {
                    streamer.stop()
                } else {
                    streamer.start(host: host, port: port)
                }
            }
            .buttonStyle(.borderedProminent)

            Text(streamer.status)
            if streamer.isStreaming {
                Text("\(streamer.samplesPerSecond) samples/s from the \(streamer.sensorLocation) AirPod")
                Text("Packets sent: \(streamer.packetsSent)")
                Text(String(format: "Rotation rate: %.3f rad/s", streamer.rotationRate))
                    .monospacedDigit()
            }
            Spacer()
        }
        .padding(24)
        .frame(minWidth: 320, minHeight: 360)
    }
}
