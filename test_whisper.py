import soundcard as sc
import numpy as np
import wave
import time
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
RECORD_SECONDS = 3

print("=" * 55)
print("ACCESSAI - WHISPER VOICE TEST")
print("=" * 55)

# Load Whisper
print("\nLoading Whisper model...")
model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded!")

# Get Windows default microphone
mic = sc.default_microphone()

print("\nMicrophone:")
print(mic)

print("\nSpeak a command after the countdown.")
print("Try:")
print("  open chrome")
print("  open calculator")
print("  go back")
print("  scroll down")
print("  refresh")
print("  click")
print()

for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

print("\n🎤 LISTENING...")
print("Speak now!")

# Record
with mic.recorder(
    samplerate=SAMPLE_RATE,
    channels=1
) as recorder:

    audio = recorder.record(
        numframes=SAMPLE_RATE * RECORD_SECONDS
    )

audio = np.asarray(audio)

# Convert float audio to int16
audio = np.clip(audio, -1, 1)
audio_int16 = (audio * 32767).astype(np.int16)

# Save WAV
wav_path = "whisper_test.wav"

with wave.open(wav_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_int16.tobytes())

print("\n✓ Audio recorded")
print("✓ Running Whisper...\n")

# Transcribe
segments, info = model.transcribe(
    wav_path,
    language="en",
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=False
)

text = " ".join(segment.text.strip() for segment in segments)

print("=" * 55)
print("RECOGNIZED:")
print(text)
print("=" * 55)

if text:
    print("\n✓ Whisper successfully understood your speech.")
else:
    print("\n✗ No speech recognized.")