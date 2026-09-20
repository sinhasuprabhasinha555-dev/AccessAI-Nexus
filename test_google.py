import pyaudio
import wave
import speech_recognition as sr
import audioop
import time

DEVICE = 8
INPUT_RATE = 44100
OUTPUT_RATE = 16000
CHANNELS = 1
CHUNK = 1024
SECONDS = 7

print("=" * 60)
print("ACCESSAI MICROPHONE + GOOGLE DIAGNOSTIC")
print("=" * 60)

p = pyaudio.PyAudio()

print(f"\nOpening device {DEVICE}...")

stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=INPUT_RATE,
    input=True,
    input_device_index=DEVICE,
    frames_per_buffer=CHUNK
)

print("\n🎙️ GET READY...")
time.sleep(2)

print("\n" + "=" * 60)
print(">>> SPEAK NOW FOR 7 SECONDS <<<")
print("Say slowly: OPEN CHROME")
print("=" * 60)

frames = []

for _ in range(int(INPUT_RATE / CHUNK * SECONDS)):
    data = stream.read(
        CHUNK,
        exception_on_overflow=False
    )
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

raw_audio = b"".join(frames)

print("\n✓ Recording complete")

# ------------------------------------------------------------
# AUDIO ANALYSIS
# ------------------------------------------------------------

rms = audioop.rms(raw_audio, 2)
peak = audioop.max(raw_audio, 2)

print(f"RMS  : {rms}")
print(f"Peak : {peak}")

# ------------------------------------------------------------
# SAVE ORIGINAL 44.1 kHz WAV
# ------------------------------------------------------------

with wave.open("voice_original.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(INPUT_RATE)
    wf.writeframes(raw_audio)

print("✓ Saved voice_original.wav")

# ------------------------------------------------------------
# RESAMPLE 44.1 kHz -> 16 kHz
# ------------------------------------------------------------

converted, state = audioop.ratecv(
    raw_audio,
    2,
    1,
    INPUT_RATE,
    OUTPUT_RATE,
    None
)

with wave.open("voice_16k.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(OUTPUT_RATE)
    wf.writeframes(converted)

print("✓ Saved voice_16k.wav")

# ------------------------------------------------------------
# GOOGLE TEST
# ------------------------------------------------------------

recognizer = sr.Recognizer()

print("\n☁ Sending 16-kHz audio to Google...")

with sr.AudioFile("voice_16k.wav") as source:
    audio = recognizer.record(source)

try:
    result = recognizer.recognize_google(
        audio,
        language="en-IN"
    )

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("Recognized:", result)
    print("=" * 60)

except sr.UnknownValueError:

    print("\n❌ Google could not understand the speech.")

    # Try US English as a second test.
    print("Trying en-US...")

    try:
        result = recognizer.recognize_google(
            audio,
            language="en-US"
        )

        print("SUCCESS!")
        print("Recognized:", result)

    except sr.UnknownValueError:
        print("❌ en-US also could not understand the speech.")

    except Exception as e:
        print("Google error:", type(e).__name__, e)

except sr.RequestError as e:
    print("Google request error:", e)

except Exception as e:
    print("Error:", type(e).__name__, e)

print("\nDiagnostic files created:")
print("  voice_original.wav")
print("  voice_16k.wav")
print()
print("IMPORTANT: Open voice_16k.wav and listen to it.")
print("You should clearly hear your own voice.")