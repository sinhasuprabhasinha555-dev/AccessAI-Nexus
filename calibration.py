import cv2
import mediapipe as mp
import json
import os
import time


# ============================================================
# ACCESSAI PERSONALIZED CALIBRATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)

PROFILE_PATH = os.path.join(
    BASE_DIR,
    "calibration_profile.json"
)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.IMAGE,

    num_faces=1
)


landmarker = (
    FaceLandmarker.create_from_options(
        options
    )
)


# ============================================================
# CALIBRATION SETTINGS
# ============================================================

SAMPLES_PER_POSITION = 40

POSITION_NAMES = [
    "CENTER",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("             ACCESSAI CALIBRATION")
    print("=" * 60)
    print()
    print(
        "This will personalize head movement"
    )
    print(
        "for your camera and natural movement range."
    )
    print()
    print(
        "Keep your face clearly visible."
    )
    print(
        "Move naturally — do not force your neck."
    )
    print()
    print(
        "Press Q at any time to cancel."
    )
    print("=" * 60)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Camera could not be opened."
        )

        return

    calibration_data = {}

    # ========================================================
    # CALIBRATION POSITIONS
    # ========================================================

    instructions = {

        "CENTER":
            "Look straight at the camera",

        "LEFT":
            "Move your head naturally to the LEFT",

        "RIGHT":
            "Move your head naturally to the RIGHT",

        "UP":
            "Move your head naturally UP",

        "DOWN":
            "Move your head naturally DOWN"
    }

    for position in POSITION_NAMES:

        print()
        print("-" * 60)
        print(
            f"CALIBRATION: {position}"
        )
        print(
            instructions[position]
        )
        print(
            "Get ready..."
        )

        # ----------------------------------------------------
        # Countdown
        # ----------------------------------------------------

        for countdown in [3, 2, 1]:

            print(
                f"{countdown}..."
            )

            start = time.time()

            while time.time() - start < 1:

                ret, frame = cap.read()

                if not ret:
                    continue

                frame = cv2.flip(
                    frame,
                    1
                )

                cv2.putText(

                    frame,

                    f"{position}: "
                    f"{countdown}",

                    (30, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.0,

                    (255, 255, 255),

                    2
                )

                cv2.imshow(
                    "AccessAI Calibration",
                    frame
                )

                key = (
                    cv2.waitKey(1)
                    &
                    0xFF
                )

                if key == ord("q"):

                    cap.release()

                    cv2.destroyAllWindows()

                    landmarker.close()

                    print(
                        "\nCalibration cancelled."
                    )

                    return

        # ----------------------------------------------------
        # Collect samples
        # ----------------------------------------------------

        samples_x = []

        samples_y = []

        collecting = True

        while collecting:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(

                image_format=(
                    mp.ImageFormat.SRGB
                ),

                data=rgb
            )

            result = (
                landmarker.detect(
                    image
                )
            )

            if result.face_landmarks:

                landmarks = (
                    result.face_landmarks[0]
                )

                nose = landmarks[1]

                samples_x.append(
                    float(nose.x)
                )

                samples_y.append(
                    float(nose.y)
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            progress = min(
                len(samples_x),
                SAMPLES_PER_POSITION
            )

            cv2.putText(

                frame,

                f"{position}",

                (30, 45),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (255, 255, 255),

                2
            )

            cv2.putText(

                frame,

                f"Samples: "
                f"{progress}/"
                f"{SAMPLES_PER_POSITION}",

                (30, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2
            )

            cv2.putText(

                frame,

                instructions[position],

                (30, 120),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )

            cv2.imshow(
                "AccessAI Calibration",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                cap.release()

                cv2.destroyAllWindows()

                landmarker.close()

                print(
                    "\nCalibration cancelled."
                )

                return

            if (
                len(samples_x)
                >=
                SAMPLES_PER_POSITION
            ):

                collecting = False

        # ----------------------------------------------------
        # Average samples
        # ----------------------------------------------------

        average_x = (
            sum(samples_x)
            /
            len(samples_x)
        )

        average_y = (
            sum(samples_y)
            /
            len(samples_y)
        )

        calibration_data[position] = {

            "x": round(
                average_x,
                5
            ),

            "y": round(
                average_y,
                5
            )
        }

        print(
            f"✓ {position} captured"
        )

        print(
            f"  X = {average_x:.4f}"
        )

        print(
            f"  Y = {average_y:.4f}"
        )

    # ========================================================
    # CALCULATE CONTROL RANGE
    # ========================================================

    center_x = calibration_data[
        "CENTER"
    ]["x"]

    center_y = calibration_data[
        "CENTER"
    ]["y"]

    left_x = calibration_data[
        "LEFT"
    ]["x"]

    right_x = calibration_data[
        "RIGHT"
    ]["x"]

    up_y = calibration_data[
        "UP"
    ]["y"]

    down_y = calibration_data[
        "DOWN"
    ]["y"]

    # --------------------------------------------------------
    # Safety margins
    #
    # We don't use the absolute extreme positions.
    # A small margin makes everyday control more comfortable.
    # --------------------------------------------------------

    horizontal_range = (
        right_x - left_x
    )

    vertical_range = (
        down_y - up_y
    )

    control_left = (
        left_x +
        horizontal_range * 0.10
    )

    control_right = (
        right_x -
        horizontal_range * 0.10
    )

    control_top = (
        up_y +
        vertical_range * 0.10
    )

    control_bottom = (
        down_y -
        vertical_range * 0.10
    )

    # ========================================================
    # CREATE PROFILE
    # ========================================================

    profile = {

        "version": 1,

        "calibration": {

            "center": {
                "x": center_x,
                "y": center_y
            },

            "left": {
                "x": left_x,
                "y": calibration_data[
                    "LEFT"
                ]["y"]
            },

            "right": {
                "x": right_x,
                "y": calibration_data[
                    "RIGHT"
                ]["y"]
            },

            "up": {
                "x": calibration_data[
                    "UP"
                ]["x"],

                "y": up_y
            },

            "down": {
                "x": calibration_data[
                    "DOWN"
                ]["x"],

                "y": down_y
            }
        },

        "control_zone": {

            "left": round(
                control_left,
                5
            ),

            "right": round(
                control_right,
                5
            ),

            "top": round(
                control_top,
                5
            ),

            "bottom": round(
                control_bottom,
                5
            )
        }
    }

    # ========================================================
    # SAVE
    # ========================================================

    with open(
        PROFILE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            profile,
            file,
            indent=4
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("           CALIBRATION COMPLETE ✓")
    print("=" * 60)

    print()

    print(
        "Personalized control zone:"
    )

    print(
        f"  LEFT   : {control_left:.4f}"
    )

    print(
        f"  RIGHT  : {control_right:.4f}"
    )

    print(
        f"  TOP    : {control_top:.4f}"
    )

    print(
        f"  BOTTOM : {control_bottom:.4f}"
    )

    print()

    print(
        "Profile saved to:"
    )

    print(
        PROFILE_PATH
    )

    print()

    print(
        "Your current AccessAI has NOT been changed."
    )

    print(
        "The calibration profile is ready for integration."
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()