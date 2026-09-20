import speech_recognition as sr

recognizer = sr.Recognizer()

print("=" * 60)
print("WAV SPEECH TEST")
print("=" * 60)

with sr.AudioFile("voice_input.wav") as source:

    print("Loading audio...")

    audio = recognizer.record(source)

print("Sending to Google...")

try:

    result = recognizer.recognize_google(
        audio,
        language="en-US",
        show_all=True
    )

    print()
    print("RESULT:")
    print(result)

except sr.UnknownValueError:

    print()
    print("Google could not understand the recording.")

except sr.RequestError as e:

    print()
    print("Google request error:")
    print(e)