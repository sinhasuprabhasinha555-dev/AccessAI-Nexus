import pyaudio

audio = pyaudio.PyAudio()

print("=" * 70)
print("AVAILABLE INPUT DEVICES")
print("=" * 70)

for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)

    if info["maxInputChannels"] > 0:
        print(
            f"\nINDEX: {i}"
            f"\nNAME: {info['name']}"
            f"\nINPUT CHANNELS: {info['maxInputChannels']}"
            f"\nSAMPLE RATE: {info['defaultSampleRate']}"
        )

audio.terminate()