STABILITY_FRAMES = 5
GESTURE_CONFIDENCE_THRESHOLD = 0.75

gesture_history = {
    "blink": []
}


def update_gesture_stability(gesture_name, detected):

    history = gesture_history[gesture_name]

    history.append(1 if detected else 0)

    if len(history) > STABILITY_FRAMES:
        history.pop(0)

    if len(history) < STABILITY_FRAMES:
        return 0.0, False

    confidence = sum(history) / len(history)

    confirmed = confidence >= GESTURE_CONFIDENCE_THRESHOLD

    return confidence, confirmed


tests = [
    True,
    True,
    True,
    True,
    True
]

for detected in tests:

    confidence, confirmed = update_gesture_stability(
        "blink",
        detected
    )

    print(
        f"Detected={detected} | "
        f"Confidence={confidence:.0%} | "
        f"Confirmed={confirmed}"
    )