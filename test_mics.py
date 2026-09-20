import pyaudio
import audioop
import time

p = pyaudio.PyAudio()

devices = [0, 1, 2, 6, 7, 8, 14, 15, 16, 19, 22, 26, 30, 36, 37, 38, 39]

RATE = 44100
CHANNELS = 2
CHUNK = 1024
SECONDS = 3

print("=" * 70)
print("ACCESSAI MICROPHONE TEST")
print("=" * 70)
print()
print("For EACH device:")
print("  1. Wait for RECORDING")
print("  2. Speak normally for 3 seconds")
print()
print("We are looking for a device with a LARGE RMS value.")
print("=" * 70)

results = []

for index in devices:
    try:
        info = p.get_device_info_by_index(index)

        if info["maxInputChannels"] <= 0:
            continue

        channels = min(2, int(info["maxInputChannels"]))

        # Use the device's native/default sample rate.
        rate = int(info["defaultSampleRate"])

        print()
        print("-" * 70)
        print(f"DEVICE {index}")
        print(f"Name: {info['name']}")
        print(f"Channels: {channels}")
        print(f"Sample rate: {rate}")
        print()
        print(">>> SPEAK NOW <<<")

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=index,
            frames_per_buffer=CHUNK,
        )

        rms_values = []
        peak_values = []

        start = time.time()

        while time.time() - start < SECONDS:
            data = stream.read(
                CHUNK,
                exception_on_overflow=False
            )

            rms = audioop.rms(data, 2)
            peak = audioop.max(data, 2)

            rms_values.append(rms)
            peak_values.append(peak)

        stream.stop_stream()
        stream.close()

        avg_rms = sum(rms_values) / len(rms_values)
        max_rms = max(rms_values)
        max_peak = max(peak_values)

        print(f"Average RMS : {avg_rms:.0f}")
        print(f"Maximum RMS : {max_rms}")
        print(f"Maximum Peak: {max_peak}")

        results.append(
            (index, info["name"], avg_rms, max_rms, max_peak)
        )

    except Exception as e:
        print(f"✗ Device {index} failed: {type(e).__name__}: {e}")


p.terminate()

print()
print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

for index, name, avg, max_rms, peak in results:
    print(
        f"{index:>2} | "
        f"RMS avg={avg:>7.0f} | "
        f"RMS max={max_rms:>7} | "
        f"Peak={peak:>7} | "
        f"{name}"
    )

print()
print("Look for the device with the highest RMS while you were speaking.")
print("=" * 70)