import speech_recognition as sr

MICROPHONE_INDEX = 1

recognizer = sr.Recognizer()

print("Starting microphone test...")
print("Microphone index:", MICROPHONE_INDEX)

try:
    with sr.Microphone(device_index=MICROPHONE_INDEX) as source:

        print("\nAdjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        print("Energy threshold:", recognizer.energy_threshold)

        print("\n>>> SPEAK NOW <<<")

        audio = recognizer.listen(
            source,
            timeout=10,
            phrase_time_limit=5
        )

        print("\nAudio captured successfully!")

        print("Trying to recognize speech...")

        text = recognizer.recognize_google(audio)

        print("\nYou said:")
        print(text)

except sr.WaitTimeoutError:
    print("\nERROR: No voice detected within 10 seconds.")

except sr.UnknownValueError:
    print("\nMicrophone captured audio, but speech could not be understood.")

except sr.RequestError as e:
    print("\nGoogle speech recognition error:")
    print(e)

except Exception as e:
    print("\nERROR:")
    print(type(e).__name__)
    print(e)