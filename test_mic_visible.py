import soundcard as sc
import numpy as np
import wave
import time
import os

# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 10
OUTPUT_FILE = "voice_original.wav"


# ============================================================
# FIND WINDOWS DEFAULT MICROPHONE
# ============================================================

print("\n" + "=" * 65)
print("              WINDOWS MICROPHONE TEST")
print("=" * 65)

try:
    microphone = sc.default_microphone()

    if microphone is None:
        print("\nERROR: Windows default microphone was not found.")
        raise SystemExit

    print(f"\nMicrophone found:")
    print(f"  {microphone.name}")

except Exception as e:
    print("\nERROR: Could not access Windows microphone.")
    print(e)
    raise SystemExit


# ============================================================
# COUNTDOWN
# ============================================================

print("\nGet ready...")

for i in range(3, 0, -1):
    print(f"Starting in {i}...")
    time.sleep(1)


# ============================================================
# RECORD
# ============================================================

print("\n" + "=" * 65)
print("                 >>> SPEAK NOW <<<")
print("=" * 65)
print('              Say: "OPEN CHROME"')
print("=" * 65)
print()

try:

    # Open Windows microphone
    with microphone.recorder(
        samplerate=SAMPLE_RATE,
        channels=1
    ) as recorder:

        audio_data = []

        start_time = time.monotonic()

        last_second = -1

        while True:

            elapsed = time.monotonic() - start_time

            if elapsed >= RECORD_SECONDS:
                break

            # Record a small chunk
            chunk = recorder.record(
                numframes=int(SAMPLE_RATE * 0.1)
            )

            audio_data.append(chunk)

            current_second = int(elapsed)

            if current_second != last_second:

                last_second = current_second

                remaining = max(
                    0,
                    RECORD_SECONDS - current_second
                )

                print(
                    f"  Recording... "
                    f"{current_second + 1}/{RECORD_SECONDS} sec "
                    f"({remaining}s remaining)"
                )

except Exception as e:

    print("\nERROR while recording:")
    print(e)
    raise SystemExit


# ============================================================
# COMBINE AUDIO
# ============================================================

try:

    audio_data = np.concatenate(audio_data, axis=0)

    # Make sure mono
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Prevent clipping
    audio_data = np.clip(
        audio_data,
        -1.0,
        1.0
    )

    # Convert float audio to 16-bit PCM
    pcm_data = (
        audio_data * 32767
    ).astype(np.int16)

except Exception as e:

    print("\nERROR processing recorded audio:")
    print(e)
    raise SystemExit


# ============================================================
# SAVE WAV
# ============================================================

try:

    with wave.open(
        OUTPUT_FILE,
        "wb"
    ) as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(
            pcm_data.tobytes()
        )

except Exception as e:

    print("\nERROR saving WAV file:")
    print(e)
    raise SystemExit


# ============================================================
# AUDIO ANALYSIS
# ============================================================

rms = np.sqrt(
    np.mean(
        audio_data ** 2
    )
)

peak = np.max(
    np.abs(audio_data)
)

print("\n" + "=" * 65)
print("                 RECORDING FINISHED")
print("=" * 65)

print(f"Samples recorded : {len(audio_data):,}")
print(f"Average RMS      : {rms:.5f}")
print(f"Peak             : {peak:.5f}")
print(f"Sample rate      : {SAMPLE_RATE} Hz")

print("=" * 65)


# ============================================================
# FILE CHECK
# ============================================================

if os.path.exists(OUTPUT_FILE):

    size = os.path.getsize(
        OUTPUT_FILE
    )

    print("\n✓ SUCCESS")
    print(f"✓ WAV file: {os.path.abspath(OUTPUT_FILE)}")
    print(f"✓ File size: {size:,} bytes")

    print("\nNow PLAY voice_original.wav.")

    print("\nTell me what you hear:")

    print("A = Clear voice")
    print("B = Voice but very quiet")
    print("C = Glitch/static/distortion")
    print("D = Silence")
    print("E = Robotic/chopped")

else:

    print("\n✗ WAV file was not created.")