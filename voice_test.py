import pyaudio
import wave
import speech_recognition as sr
import os

# =========================================================
# ACCESSAI - AUTOMATIC MICROPHONE TEST
# =========================================================

RECORD_SECONDS = 5
CHUNK = 1024
FORMAT = pyaudio.paInt16
TEMP_AUDIO = "voice_input.wav"


# =========================================================
# HEADER
# =========================================================

print("=" * 65)
print("          ACCESSAI AUTOMATIC VOICE TEST")
print("=" * 65)


audio = pyaudio.PyAudio()


# =========================================================
# FIND WORKING MICROPHONE
# =========================================================

print("\nSearching for usable microphones...\n")

working_devices = []

for i in range(audio.get_device_count()):

    try:
        info = audio.get_device_info_by_index(i)

        # Ignore devices with no input
        if info["maxInputChannels"] <= 0:
            continue

        print(
            f"[{i}] {info['name']} "
            f"(channels={info['maxInputChannels']}, "
            f"rate={int(info['defaultSampleRate'])})"
        )

        # Try opening the device
        test_rate = int(info["defaultSampleRate"])

        # Try mono first
        try:
            test_stream = audio.open(
                format=FORMAT,
                channels=1,
                rate=test_rate,
                input=True,
                input_device_index=i,
                frames_per_buffer=CHUNK
            )

            test_stream.stop_stream()
            test_stream.close()

            working_devices.append(
                {
                    "index": i,
                    "name": info["name"],
                    "rate": test_rate,
                    "channels": 1
                }
            )

            print("    ✓ Can be opened\n")

        except Exception as e:

            print(
                f"    ✗ Cannot open in mono: "
                f"{type(e).__name__}\n"
            )

    except Exception:
        continue


# =========================================================
# NO MICROPHONE FOUND
# =========================================================

if not working_devices:

    print("=" * 65)
    print("NO WORKING MICROPHONE FOUND")
    print("=" * 65)

    print(
        "\nPyAudio can see your microphones, "
        "but Windows is not allowing an input stream."
    )

    print("\nPlease check:")
    print("1. Windows microphone permission")
    print("2. Your microphone is not being used exclusively")
    print("3. Your microphone is enabled in Windows Sound settings")

    audio.terminate()
    raise SystemExit


# =========================================================
# SELECT FIRST WORKING MICROPHONE
# =========================================================

mic = working_devices[0]

MICROPHONE_INDEX = mic["index"]
SAMPLE_RATE = mic["rate"]
CHANNELS = mic["channels"]

print("=" * 65)
print("SELECTED MICROPHONE")
print("=" * 65)

print(f"\nIndex       : {MICROPHONE_INDEX}")
print(f"Name        : {mic['name']}")
print(f"Sample rate : {SAMPLE_RATE}")
print(f"Channels    : {CHANNELS}")


# =========================================================
# RECORD
# =========================================================

print("\n" + "=" * 65)
print("RECORDING")
print("=" * 65)

print(f"\nRecording for {RECORD_SECONDS} seconds...")
print("Opening microphone...")

try:

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=MICROPHONE_INDEX,
        frames_per_buffer=CHUNK
    )

    print("Microphone opened successfully.")

    print("\n" + "=" * 65)
    print(">>> SPEAK NOW <<<")
    print("=" * 65)

    print("\nTry saying:")
    print("    click")
    print("    scroll up")
    print("    open chrome")
    print("    open calculator")

    frames = []

    total_chunks = int(
        SAMPLE_RATE / CHUNK * RECORD_SECONDS
    )

    for _ in range(total_chunks):

        data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        frames.append(data)

    print("\nRecording finished.")

    stream.stop_stream()
    stream.close()

except Exception as e:

    print("\n" + "=" * 65)
    print("RECORDING ERROR")
    print("=" * 65)

    print(f"\n{type(e).__name__}: {e}")

    audio.terminate()
    raise SystemExit


# =========================================================
# SAVE WAV
# =========================================================

try:

    with wave.open(TEMP_AUDIO, "wb") as wav_file:

        wav_file.setnchannels(CHANNELS)

        wav_file.setsampwidth(
            audio.get_sample_size(FORMAT)
        )

        wav_file.setframerate(SAMPLE_RATE)

        wav_file.writeframes(
            b"".join(frames)
        )

    audio.terminate()

    print(f"\nAudio saved successfully:")
    print(f"    {os.path.abspath(TEMP_AUDIO)}")

except Exception as e:

    print("\nError saving audio:")
    print(type(e).__name__, e)

    try:
        audio.terminate()
    except:
        pass

    raise SystemExit


# =========================================================
# SPEECH RECOGNITION
# =========================================================

print("\n" + "=" * 65)
print("SPEECH RECOGNITION")
print("=" * 65)

recognizer = sr.Recognizer()

try:

    with sr.AudioFile(TEMP_AUDIO) as source:

        recorded_audio = recognizer.record(source)

    print("\nSending audio for recognition...")

    text = recognizer.recognize_google(
        recorded_audio,
        language="en-IN"
    )

    text = text.lower().strip()

    print("\n" + "=" * 65)
    print("                 SUCCESS")
    print("=" * 65)

    print(f"\nRecognized speech:")
    print(f">>> {text}")

    # =====================================================
    # ACCESSAI COMMAND MATCHING
    # =====================================================

    print("\nCommand detection:")

    if "click" in text:
        print("✓ CLICK command detected")

    elif "scroll up" in text:
        print("✓ SCROLL UP command detected")

    elif "scroll down" in text:
        print("✓ SCROLL DOWN command detected")

    elif "open chrome" in text:
        print("✓ OPEN CHROME command detected")

    elif "open calculator" in text:
        print("✓ OPEN CALCULATOR command detected")

    elif "go back" in text:
        print("✓ GO BACK command detected")

    elif "stop accessai" in text:
        print("✓ STOP ACCESSAI command detected")

    else:
        print("⚠ Speech recognized.")
        print("⚠ No AccessAI command matched.")


except sr.UnknownValueError:

    print("\n" + "=" * 65)
    print("SPEECH NOT UNDERSTOOD")
    print("=" * 65)

    print(
        "\nThe microphone recorded audio, "
        "but Google could not understand the speech."
    )


except sr.RequestError as e:

    print("\n" + "=" * 65)
    print("SPEECH RECOGNITION NETWORK ERROR")
    print("=" * 65)

    print(f"\n{e}")


except Exception as e:

    print("\n" + "=" * 65)
    print("SPEECH RECOGNITION ERROR")
    print("=" * 65)

    print(f"\n{type(e).__name__}: {e}")


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 65)
print("VOICE TEST FINISHED")
print("=" * 65)

print(f"\nRecording file: {TEMP_AUDIO}")