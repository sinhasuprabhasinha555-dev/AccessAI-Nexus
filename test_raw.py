import speech_recognition as sr

r = sr.Recognizer()

print("=" * 60)
print("RAW WAV TEST")
print("=" * 60)

with sr.AudioFile("voice_input_raw.wav") as source:
    audio = r.record(source)

print("Sending raw WAV to Google...")

try:
    text = r.recognize_google(
        audio,
        language="en-IN"
    )

    print()
    print("SUCCESS!")
    print("Recognized:", text)

except sr.UnknownValueError:
    print()
    print("Google could not understand RAW WAV.")

except sr.RequestError as e:
    print()
    print("Google error:", e)