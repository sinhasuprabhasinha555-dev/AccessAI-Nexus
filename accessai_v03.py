import cv2
import mediapipe as mp
import pyautogui
import math
import time


# ============================================================
# ACCESSAI v0.3
# Hands-Free Computer Control
#
# Features:
#   1. Head movement -> Cursor
#   2. Blink -> Left Click
#   3. Head tilt -> Scroll
#
# Improvement:
#   Comfortable cursor-control zone
#   Faster scrolling
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_landmarker.task"


# ============================================================
# SCREEN CONFIGURATION
# ============================================================

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

current_x = SCREEN_WIDTH / 2
current_y = SCREEN_HEIGHT / 2

# Cursor smoothing
CURSOR_SMOOTHING = 0.20


# ============================================================
# COMFORTABLE CURSOR CONTROL ZONE
# ============================================================
#
# Only moderate head movement is required.
# This area of the camera frame maps to the entire screen.
#
# Smaller zone = less head movement required
# Larger zone = more precise but more head movement
# ============================================================

CONTROL_LEFT = 0.30
CONTROL_RIGHT = 0.70

CONTROL_TOP = 0.30
CONTROL_BOTTOM = 0.70


# ============================================================
# BLINK CONFIGURATION
# ============================================================

BLINK_THRESHOLD = 0.20

MIN_BLINK_DURATION = 0.08
MAX_BLINK_DURATION = 0.80

BLINK_COOLDOWN = 1.0

blink_start = None
last_blink = 0


# ============================================================
# SCROLL CONFIGURATION
# ============================================================

TILT_THRESHOLD = 0.12

# Increased from 3 to 7 for faster scrolling
SCROLL_AMOUNT = 7

SCROLL_COOLDOWN = 0.15

last_scroll_time = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(p1, p2):
    """
    Calculate Euclidean distance between
    two face landmarks.
    """

    return math.sqrt(
        (p1.x - p2.x) ** 2
        +
        (p1.y - p2.y) ** 2
    )


def eye_aspect_ratio(landmarks, eye):
    """
    Calculate Eye Aspect Ratio (EAR).
    Used to detect blinking.
    """

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

    return (
        vertical_1
        +
        vertical_2
    ) / (2 * horizontal)


# ============================================================
# EYE LANDMARKS
# ============================================================

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


# ============================================================
# MEDIAPIPE FACE LANDMARKER
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
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not access camera."
    )

    landmarker.close()

    raise SystemExit


# ============================================================
# START MESSAGE
# ============================================================

print()
print("================================================")
print("                 ACCESSAI v0.3")
print("================================================")
print()
print("HEAD MOVEMENT  -> CURSOR")
print("BLINK          -> LEFT CLICK")
print("HEAD TILT      -> SCROLL")
print()
print("Comfortable cursor-control zone: ACTIVE")
print("Fast scrolling: ACTIVE")
print()
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    frame_height, frame_width, _ = frame.shape


    # --------------------------------------------------------
    # Convert OpenCV frame -> MediaPipe image
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

    result = landmarker.detect(
        mp_image
    )


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ====================================================
        # 1. HEAD MOVEMENT -> CURSOR
        # ====================================================

        nose = landmarks[1]

        face_x = nose.x
        face_y = nose.y


        # ----------------------------------------------------
        # Convert comfortable control zone
        # to normalized screen position
        # ----------------------------------------------------

        normalized_x = (
            face_x - CONTROL_LEFT
        ) / (
            CONTROL_RIGHT - CONTROL_LEFT
        )


        normalized_y = (
            face_y - CONTROL_TOP
        ) / (
            CONTROL_BOTTOM - CONTROL_TOP
        )


        # ----------------------------------------------------
        # Keep cursor inside screen
        # ----------------------------------------------------

        normalized_x = max(
            0.0,
            min(
                1.0,
                normalized_x
            )
        )


        normalized_y = max(
            0.0,
            min(
                1.0,
                normalized_y
            )
        )


        # ----------------------------------------------------
        # Convert to screen coordinates
        # ----------------------------------------------------

        target_x = (
            normalized_x
            * SCREEN_WIDTH
        )


        target_y = (
            normalized_y
            * SCREEN_HEIGHT
        )


        # ----------------------------------------------------
        # Smooth cursor movement
        # ----------------------------------------------------

        current_x += (
            target_x
            -
            current_x
        ) * CURSOR_SMOOTHING


        current_y += (
            target_y
            -
            current_y
        ) * CURSOR_SMOOTHING


        # ----------------------------------------------------
        # Move mouse
        # ----------------------------------------------------

        pyautogui.moveTo(
            int(current_x),
            int(current_y),
            duration=0
        )


        # ====================================================
        # 2. BLINK -> LEFT CLICK
        # ====================================================

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE
        )


        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE
        )


        ear = (
            left_ear
            +
            right_ear
        ) / 2


        # ----------------------------------------------------
        # Detect closed eyes
        # ----------------------------------------------------

        if ear < BLINK_THRESHOLD:

            if blink_start is None:

                blink_start = time.time()


        else:

            if blink_start is not None:

                blink_duration = (
                    time.time()
                    -
                    blink_start
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
                    current_time
                    -
                    last_blink
                    >= BLINK_COOLDOWN
                )


                # ------------------------------------------------
                # Perform left click
                # ------------------------------------------------

                if (
                    valid_blink
                    and
                    cooldown_finished
                ):

                    print(
                        "BLINK -> LEFT CLICK"
                    )

                    pyautogui.click()

                    last_blink = (
                        current_time
                    )


                blink_start = None


        # ====================================================
        # 3. HEAD TILT -> SCROLL
        # ====================================================

        left_eye = landmarks[33]

        right_eye = landmarks[263]


        eye_slope = (
            right_eye.y
            -
            left_eye.y
        )


        current_time = time.time()


        # ----------------------------------------------------
        # Scroll
        # ----------------------------------------------------

        if (
            abs(eye_slope)
            >
            TILT_THRESHOLD
            and
            current_time
            -
            last_scroll_time
            >
            SCROLL_COOLDOWN
        ):

            if eye_slope > 0:

                pyautogui.scroll(
                    -SCROLL_AMOUNT
                )

                scroll_status = (
                    "SCROLL DOWN"
                )

            else:

                pyautogui.scroll(
                    SCROLL_AMOUNT
                )

                scroll_status = (
                    "SCROLL UP"
                )


            last_scroll_time = (
                current_time
            )


        else:

            scroll_status = (
                "SCROLL READY"
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            "ACCESSAI v0.3",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Eye Ratio: {ear:.3f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            scroll_status,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "HEAD = CURSOR",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "BLINK = CLICK",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DRAW COMFORTABLE CONTROL ZONE
        # ====================================================

        zone_x1 = int(
            CONTROL_LEFT
            * frame_width
        )

        zone_y1 = int(
            CONTROL_TOP
            * frame_height
        )

        zone_x2 = int(
            CONTROL_RIGHT
            * frame_width
        )

        zone_y2 = int(
            CONTROL_BOTTOM
            * frame_height
        )


        cv2.rectangle(
            frame,
            (
                zone_x1,
                zone_y1
            ),
            (
                zone_x2,
                zone_y2
            ),
            (255, 255, 255),
            1
        )


    # ========================================================
    # NO FACE DETECTED
    # ========================================================

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "AccessAI - Hands Free Control",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()


print()
print("AccessAI v0.3 stopped.")