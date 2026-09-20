import cv2
import mediapipe as mp
import pyautogui


# ============================================================
# AccessAI v0.1
# Hands-Free Head-Controlled Cursor
# ============================================================


# ------------------------------------------------------------
# 1. Screen configuration
# ------------------------------------------------------------

screen_width, screen_height = pyautogui.size()

# Start cursor at the center of the screen
current_x = screen_width // 2
current_y = screen_height // 2

# Cursor smoothing
SMOOTHING = 0.25


# ------------------------------------------------------------
# 2. MediaPipe Face Detector
# ------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = "models/face_detector.tflite"


options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.6
)


detector = FaceDetector.create_from_options(options)


# ------------------------------------------------------------
# 3. Start camera
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not access the camera.")
    detector.close()
    raise SystemExit


print()
print("===================================")
print("        AccessAI v0.1")
print("===================================")
print("Head-controlled cursor is active.")
print("Move your head to control the cursor.")
print("Press Q to quit.")
print()


# ------------------------------------------------------------
# 4. Main loop
# ------------------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    frame_height, frame_width, _ = frame.shape

    # --------------------------------------------------------
    # Convert OpenCV frame → MediaPipe image
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Detect face
    # --------------------------------------------------------

    result = detector.detect(mp_image)


    if result.detections:

        # Use the first detected face
        detection = result.detections[0]

        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y

        width = bbox.width
        height = bbox.height


        # ----------------------------------------------------
        # Calculate center of face
        # ----------------------------------------------------

        face_center_x = x + width / 2
        face_center_y = y + height / 2


        # ----------------------------------------------------
        # Convert face position to normalized coordinates
        # ----------------------------------------------------

        normalized_x = face_center_x / frame_width
        normalized_y = face_center_y / frame_height


        # Keep values between 0 and 1
        normalized_x = max(
            0,
            min(1, normalized_x)
        )

        normalized_y = max(
            0,
            min(1, normalized_y)
        )


        # ----------------------------------------------------
        # Convert face position → screen position
        # ----------------------------------------------------

        target_x = normalized_x * screen_width
        target_y = normalized_y * screen_height


        # ----------------------------------------------------
        # Smooth cursor movement
        # ----------------------------------------------------

        current_x = (
            current_x
            + (target_x - current_x) * SMOOTHING
        )

        current_y = (
            current_y
            + (target_y - current_y) * SMOOTHING
        )


        # ----------------------------------------------------
        # Move cursor
        # ----------------------------------------------------

        pyautogui.moveTo(
            int(current_x),
            int(current_y),
            duration=0
        )


        # ----------------------------------------------------
        # Draw face bounding box
        # ----------------------------------------------------

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(
            frame_width,
            x + width
        )

        y2 = min(
            frame_height,
            y + height
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Draw face center
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (
                int(face_center_x),
                int(face_center_y)
            ),
            5,
            (255, 255, 255),
            -1
        )


        # ----------------------------------------------------
        # Status text
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "HEAD CONTROL: ACTIVE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        # No face detected
        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # Display camera
    # --------------------------------------------------------

    cv2.imshow(
        "AccessAI - Head Control",
        frame
    )


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ------------------------------------------------------------
# 5. Cleanup
# ------------------------------------------------------------

cap.release()

detector.close()

cv2.destroyAllWindows()

print("AccessAI stopped.")