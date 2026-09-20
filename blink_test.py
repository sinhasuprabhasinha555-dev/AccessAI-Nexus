import cv2
import mediapipe as mp
import math
import time
import pyautogui


# ============================================================
# AccessAI v0.2
# Blink Detection + Mouse Click
# ============================================================


MODEL_PATH = "models/face_landmarker.task"


# ------------------------------------------------------------
# Calculate distance between two face landmarks
# ------------------------------------------------------------

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ------------------------------------------------------------
# Calculate Eye Aspect Ratio (EAR)
# ------------------------------------------------------------

def eye_aspect_ratio(landmarks, eye):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2 * horizontal)


# ------------------------------------------------------------
# MediaPipe Face Landmarker
# ------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)


landmarker = FaceLandmarker.create_from_options(options)


# ------------------------------------------------------------
# Eye landmark indexes
# ------------------------------------------------------------

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not access camera.")

    landmarker.close()

    raise SystemExit


print()
print("==============================================")
print("              AccessAI v0.2")
print("        BLINK → LEFT CLICK SYSTEM")
print("==============================================")
print()
print("Blink normally to perform a left mouse click.")
print("Press Q to quit.")
print()


# ------------------------------------------------------------
# Blink settings
# ------------------------------------------------------------

BLINK_THRESHOLD = 0.20

MIN_BLINK_DURATION = 0.08
MAX_BLINK_DURATION = 0.80

BLINK_COOLDOWN = 1.0


blink_start = None
last_blink = 0


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read camera frame.")

        break


    # Mirror camera
    frame = cv2.flip(frame, 1)

    frame_height, frame_width, _ = frame.shape


    # --------------------------------------------------------
    # Convert OpenCV frame → MediaPipe image
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # --------------------------------------------------------
    # Detect face landmarks
    # --------------------------------------------------------

    result = landmarker.detect(mp_image)


    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # Calculate both eye ratios
        # ----------------------------------------------------

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE
        )


        # Average both eyes
        ear = (left_ear + right_ear) / 2


        # ----------------------------------------------------
        # Detect closed eyes
        # ----------------------------------------------------

        if ear < BLINK_THRESHOLD:

            if blink_start is None:

                blink_start = time.time()


        else:

            # Eyes opened again
            if blink_start is not None:

                blink_duration = (
                    time.time() - blink_start
                )

                current_time = time.time()


                # ------------------------------------------------
                # Validate blink
                # ------------------------------------------------

                valid_blink = (
                    MIN_BLINK_DURATION
                    <= blink_duration
                    <= MAX_BLINK_DURATION
                )


                cooldown_finished = (
                    current_time - last_blink
                    >= BLINK_COOLDOWN
                )


                if valid_blink and cooldown_finished:

                    print(
                        "BLINK DETECTED → LEFT CLICK"
                    )

                    # Perform left mouse click
                    pyautogui.click()

                    last_blink = current_time


                blink_start = None


        # ----------------------------------------------------
        # Display eye ratio
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Eye Ratio: {ear:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Display eye status
        # ----------------------------------------------------

        if ear < BLINK_THRESHOLD:

            status = "EYES CLOSED"

        else:

            status = "EYES OPEN"


        cv2.putText(
            frame,
            status,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Display system status
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "BLINK CONTROL: ACTIVE",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # Show camera window
    # --------------------------------------------------------

    cv2.imshow(
        "AccessAI - Blink Control",
        frame
    )


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print()
print("AccessAI blink control stopped.")