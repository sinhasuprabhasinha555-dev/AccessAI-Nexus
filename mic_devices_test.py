import speech_recognition as sr
import time

print("=" * 60)
print("ACCESSAI — MICROPHONE DEVICE TEST")
print("=" * 60)

microphones = sr.Microphone.list_microphone_names()

for i, name in enumerate(microphones):
    print(f"{i}: {name}")

print("\n" + "=" * 60)
print("We will test the likely microphone devices.")
print("=" * 60)

# Test these first
test_indexes = [1, 5, 9, 16, 20, 21, 22, 23]

for index in test_indexes:

    if index >= len(microphones):
        continue

    print("\n" + "-" * 60)
    print(f"TESTING MICROPHONE {index}")
    print(microphones[index])
    print("-" * 60)

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone(device_index=index) as source:

            print("Calibrating for 2 seconds...")
            recognizer.adjust_for_ambient_noise(
                source,
                duration=2
            )

            print(
                f"Energy threshold: "
                f"{recognizer.energy_threshold:.2f}"
            )

            print("\n>>> SPEAK NOW <<<")

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=4
            )

            print("\n✓ AUDIO DETECTED!")

            try:

                text = recognizer.recognize_google(audio)

                print("✓ RECOGNIZED:")
                print(text)

                print("\nTHIS MICROPHONE WORKS!")
                print(f"Use MICROPHONE_INDEX = {index}")

                break

            except sr.UnknownValueError:

                print(
                    "✓ Audio was captured, "
                    "but speech wasn't understood."
                )

                print(
                    f"This is still a GOOD microphone candidate: "
                    f"{index}"
                )

            except sr.RequestError as e:

                print("Audio captured, but Google recognition failed:")
                print(e)

    except sr.WaitTimeoutError:

        print("✗ No audio detected.")

    except Exception as e:

        print("✗ Error:")
        print(type(e).__name__, e)

    time.sleep(1)

print("\n" + "=" * 60)
print("MICROPHONE TEST FINISHED")
print("=" * 60)