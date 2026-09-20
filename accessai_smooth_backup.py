import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading
import queue
import os
import wave
import difflib
import pyaudio
import speech_recognition as sr


# ============================================================
# ACCESSAI NEXUS
# Hands-Free Computer Accessibility System
#
# FACE:
#   Head movement -> Cursor
#   Blink         -> Click
#   Head tilt     -> Continuous Adaptive Scroll
#
# VOICE:
#   Currently backup controls.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)


# ============================================================
# PYAutoGUI
# ============================================================

pyautogui.PAUSE = 0.0

pyautogui.MINIMUM_DURATION = 0.0

pyautogui.MINIMUM_SLEEP = 0.0


SCREEN_WIDTH, SCREEN_HEIGHT = (
    pyautogui.size()
)


# ============================================================
# CURSOR
# ============================================================

# DO NOT CHANGE THESE.
# Cursor was already working smoothly.

CURSOR_SMOOTHING = 0.12

CURSOR_DEADZONE = 0.012

MAX_CURSOR_STEP = 45


CONTROL_LEFT = 0.30
CONTROL_RIGHT = 0.70

CONTROL_TOP = 0.30
CONTROL_BOTTOM = 0.70


# ============================================================
# BLINK
# ============================================================

# DO NOT CHANGE THESE.
# Blink was already working smoothly.

BLINK_THRESHOLD = 0.20

MIN_BLINK_DURATION = 0.06

MAX_BLINK_DURATION = 0.65

BLINK_COOLDOWN = 0.65

MIN_CLOSED_FRAMES = 2


blink_start = None

blink_closed_frames = 0

last_blink = 0


# ============================================================
# SCROLL
# ============================================================

# Lower threshold means less head movement is required.

TILT_THRESHOLD = 0.10

# Faster than previous version.
SCROLL_STEP = 5

# Faster repeated scroll events.
SCROLL_INTERVAL = 0.08

# Prevent accidental scrolling.
TILT_CONFIRM_FRAMES = 4

# How quickly scrolling stops after returning neutral.
TILT_RELEASE_FRAMES = 3


tilt_direction = 0

tilt_stable_frames = 0

tilt_release_frames = 0

last_scroll_time = 0


# ============================================================
# VOICE
# ============================================================

VOICE_CHANNELS = 1

VOICE_COOLDOWN = 1.2

VOICE_LANGUAGE = "en-US"

VOICE_RECORD_SECONDS = 5


voice_queue = queue.Queue()

voice_running = True

last_voice_command = ""

last_voice_time = 0


# ============================================================
# AUDIO
# ============================================================

audio = pyaudio.PyAudio()


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
# DISTANCE
# ============================================================

def distance(
    p1,
    p2
):

    return math.sqrt(
        (p1.x - p2.x) ** 2
        +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def eye_aspect_ratio(
    landmarks,
    eye
):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical_1 = distance(
        p2,
        p6
    )

    vertical_2 = distance(
        p3,
        p5
    )

    horizontal = distance(
        p1,
        p4
    )

    if horizontal == 0:

        return 0

    return (
        vertical_1 +
        vertical_2
    ) / (
        2 * horizontal
    )


# ============================================================
# SMOOTH CURSOR
# ============================================================

def smooth_cursor(
    current_x,
    current_y,
    target_x,
    target_y
):

    dx = target_x - current_x

    dy = target_y - current_y

    movement_distance = math.sqrt(
        dx * dx +
        dy * dy
    )

    deadzone_pixels = (
        CURSOR_DEADZONE *
        max(
            SCREEN_WIDTH,
            SCREEN_HEIGHT
        )
    )

    # Ignore tiny movements.

    if movement_distance < deadzone_pixels:

        return (
            current_x,
            current_y
        )

    # Smooth movement.

    new_x = (
        current_x +
        dx * CURSOR_SMOOTHING
    )

    new_y = (
        current_y +
        dy * CURSOR_SMOOTHING
    )

    # Limit sudden jumps.

    step_x = new_x - current_x

    step_y = new_y - current_y

    step_distance = math.sqrt(
        step_x * step_x +
        step_y * step_y
    )

    if step_distance > MAX_CURSOR_STEP:

        scale = (
            MAX_CURSOR_STEP /
            step_distance
        )

        new_x = (
            current_x +
            step_x * scale
        )

        new_y = (
            current_y +
            step_y * scale
        )

    # Keep cursor on screen.

    new_x = max(
        0,
        min(
            SCREEN_WIDTH - 1,
            new_x
        )
    )

    new_y = max(
        0,
        min(
            SCREEN_HEIGHT - 1,
            new_y
        )
    )

    return (
        new_x,
        new_y
    )


# ============================================================
# VOICE TEXT CLEANING
# ============================================================

def clean_voice_text(
    text
):

    if not text:

        return ""

    text = (
        text
        .lower()
        .strip()
    )

    replacements = {

        "please": "",

        "kindly": "",

        "can you": "",

        "could you": "",

        "would you": "",

        "i want to": "",

        "i need to": "",

        "will you": ""
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    return " ".join(
        text.split()
    )


# ============================================================
# FUZZY MATCH
# ============================================================

def fuzzy_score(
    text,
    target
):

    return difflib.SequenceMatcher(
        None,
        text,
        target
    ).ratio()


# ============================================================
# VOICE COMMAND DETECTION
# ============================================================

def detect_command(
    text
):

    text = clean_voice_text(
        text
    )

    if not text:

        return None

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if (
        "stop accessai"
        in text
    ):

        return "stop accessai"

    # --------------------------------------------------------
    # APPLICATIONS
    # --------------------------------------------------------

    if (
        "open chrome"
        in text
    ):

        return "open chrome"

    if (
        "open calculator"
        in text
    ):

        return "open calculator"

    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    if (
        "go back"
        in text
    ):

        return "go back"

    if (
        "go forward"
        in text
    ):

        return "go forward"

    if (
        "refresh"
        in text
    ):

        return "refresh"

    # --------------------------------------------------------
    # TEMPORARY VOICE SCROLL
    # --------------------------------------------------------

    if (
        "scroll up"
        in text
    ):

        return "scroll up"

    if (
        "scroll down"
        in text
    ):

        return "scroll down"

    # --------------------------------------------------------
    # TEMPORARY VOICE CLICK
    # --------------------------------------------------------

    click_words = [
        "click",
        "clique",
        "select"
    ]

    for word in text.split():

        for target in click_words:

            score = fuzzy_score(
                word,
                target
            )

            if score >= 0.88:

                return "click"

    # --------------------------------------------------------
    # ENTER
    # --------------------------------------------------------

    if (
        "press enter"
        in text
        or
        text == "enter"
    ):

        return "enter"

    # --------------------------------------------------------
    # SPACE
    # --------------------------------------------------------

    if (
        "press space"
        in text
        or
        text == "space"
    ):

        return "space"

    return None


# ============================================================
# EXECUTE VOICE COMMAND
# ============================================================

def execute_voice_command(
    command
):

    global voice_running

    if command is None:

        return

    print()

    print(
        f"✓ Executing: "
        f"{command.upper()}"
    )

    try:

        if command == "click":

            pyautogui.click()

        elif command == "scroll up":

            pyautogui.scroll(
                SCROLL_STEP * 3
            )

        elif command == "scroll down":

            pyautogui.scroll(
                -SCROLL_STEP * 3
            )

        elif command == "go back":

            pyautogui.hotkey(
                "alt",
                "left"
            )

        elif command == "go forward":

            pyautogui.hotkey(
                "alt",
                "right"
            )

        elif command == "refresh":

            pyautogui.hotkey(
                "ctrl",
                "r"
            )

        elif command == "enter":

            pyautogui.press(
                "enter"
            )

        elif command == "space":

            pyautogui.press(
                "space"
            )

        elif command == "open chrome":

            os.system(
                "start chrome"
            )

        elif command == "open calculator":

            os.system(
                "start calc"
            )

        elif command == "stop accessai":

            voice_running = False

    except Exception as e:

        print(
            f"Command error: {e}"
        )


# ============================================================
# FIND MICROPHONE
# ============================================================

def find_working_microphone():

    print()

    print(
        "Searching for microphone..."
    )

    for index in range(
        audio.get_device_count()
    ):

        try:

            info = (
                audio
                .get_device_info_by_index(
                    index
                )
            )

            max_input = int(
                info.get(
                    "maxInputChannels",
                    0
                )
            )

            if max_input <= 0:

                continue

            sample_rate = int(
                info.get(
                    "defaultSampleRate",
                    16000
                )
            )

            try:

                stream = audio.open(

                    format=pyaudio.paInt16,

                    channels=1,

                    rate=sample_rate,

                    input=True,

                    input_device_index=index,

                    frames_per_buffer=1024
                )

                stream.stop_stream()

                stream.close()

                print(
                    f"✓ Microphone selected: "
                    f"{index} - "
                    f"{info['name']}"
                )

                return (
                    index,
                    sample_rate
                )

            except Exception:

                continue

        except Exception:

            continue

    print(
        "✗ No working microphone found."
    )

    return (
        None,
        16000
    )


# ============================================================
# RECORD VOICE
# ============================================================

def record_voice_command(
    mic_index,
    sample_rate
):

    frames = []

    stream = None

    try:

        stream = audio.open(

            format=pyaudio.paInt16,

            channels=VOICE_CHANNELS,

            rate=sample_rate,

            input=True,

            input_device_index=mic_index,

            frames_per_buffer=1024
        )

        print()

        print(
            "=" * 50
        )

        print(
            ">>> SPEAK NOW <<<"
        )

        print(
            "=" * 50
        )

        total_chunks = int(
            sample_rate
            /
            1024
            *
            VOICE_RECORD_SECONDS
        )

        for _ in range(
            total_chunks
        ):

            data = stream.read(

                1024,

                exception_on_overflow=False
            )

            frames.append(
                data
            )

        print(
            "Recording finished."
        )

    except Exception as e:

        print(
            f"Microphone error: {e}"
        )

        return None

    finally:

        if stream is not None:

            try:

                stream.stop_stream()

                stream.close()

            except Exception:

                pass

    filename = (
        "voice_command.wav"
    )

    try:

        with wave.open(
            filename,
            "wb"
        ) as wf:

            wf.setnchannels(
                VOICE_CHANNELS
            )

            wf.setsampwidth(
                audio.get_sample_size(
                    pyaudio.paInt16
                )
            )

            wf.setframerate(
                sample_rate
            )

            wf.writeframes(
                b"".join(frames)
            )

        return filename

    except Exception as e:

        print(
            f"WAV save error: {e}"
        )

        return None


# ============================================================
# VOICE WORKER
# ============================================================

def voice_worker():

    global last_voice_command

    global last_voice_time

    recognizer = (
        sr.Recognizer()
    )

    mic_index, sample_rate = (
        find_working_microphone()
    )

    if mic_index is None:

        print(
            "Voice control disabled."
        )

        return

    while voice_running:

        filename = (
            record_voice_command(
                mic_index,
                sample_rate
            )
        )

        if filename is None:

            time.sleep(1)

            continue

        try:

            with sr.AudioFile(
                filename
            ) as source:

                audio_data = (
                    recognizer.record(
                        source
                    )
                )

            print(
                "Sending audio to "
                "speech recognition..."
            )

            try:

                results = (
                    recognizer
                    .recognize_google(
                        audio_data,
                        language=VOICE_LANGUAGE,
                        show_all=True
                    )
                )

            except sr.UnknownValueError:

                print(
                    "✗ Speech not understood."
                )

                continue

            except sr.RequestError as e:

                print(
                    f"✗ Recognition error: "
                    f"{e}"
                )

                continue

            alternatives = []

            if isinstance(
                results,
                dict
            ):

                for result in (
                    results.get(
                        "alternative",
                        []
                    )
                ):

                    text = result.get(
                        "transcript",
                        ""
                    )

                    if text:

                        alternatives.append(
                            text
                        )

            if not alternatives:

                print(
                    "✗ Speech not understood."
                )

                continue

            print()

            print(
                "Recognized:"
            )

            for text in alternatives[:5]:

                print(
                    f"  → {text}"
                )

            command = None

            for text in alternatives:

                detected = (
                    detect_command(
                        text
                    )
                )

                if detected:

                    command = detected

                    break

            if command is None:

                print(
                    "✗ No valid "
                    "AccessAI command."
                )

                continue

            now = time.time()

            if (
                command ==
                last_voice_command
                and
                now -
                last_voice_time
                <
                VOICE_COOLDOWN
            ):

                print(
                    "✗ Voice cooldown."
                )

                continue

            last_voice_command = command

            last_voice_time = now

            print(
                f"✓ Valid command: "
                f"{command}"
            )

            voice_queue.put(
                command
            )

        except Exception as e:

            print(
                f"Voice processing error: "
                f"{e}"
            )

        finally:

            try:

                if os.path.exists(
                    filename
                ):

                    os.remove(
                        filename
                    )

            except Exception:

                pass

        time.sleep(
            0.1
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global blink_start

    global blink_closed_frames

    global last_blink

    global tilt_direction

    global tilt_stable_frames

    global tilt_release_frames

    global last_scroll_time

    print()

    print(
        "=" * 65
    )

    print(
        "                    ACCESSAI NEXUS"
    )

    print(
        "              Hands-Free Computer Control"
    )

    print(
        "=" * 65
    )

    print()

    print(
        "FACIAL CONTROLS"
    )

    print(
        "  Head movement  -> Cursor"
    )

    print(
        "  Blink          -> Click"
    )

    print(
        "  Head tilt      -> Continuous Adaptive Scroll"
    )

    print()

    print(
        "SCROLL SETTINGS"
    )

    print(
        f"  Tilt threshold : "
        f"{TILT_THRESHOLD}"
    )

    print(
        f"  Scroll step    : "
        f"{SCROLL_STEP}"
    )

    print(
        f"  Scroll interval: "
        f"{SCROLL_INTERVAL}s"
    )

    print()

    print(
        "VOICE BACKUP"
    )

    print(
        "  Open Chrome"
    )

    print(
        "  Open Calculator"
    )

    print(
        "  Go Back"
    )

    print(
        "  Refresh"
    )

    print()

    print(
        "Press Q to quit."
    )

    print(
        "=" * 65
    )

    # ========================================================
    # VOICE THREAD
    # ========================================================

    voice_thread = threading.Thread(

        target=voice_worker,

        daemon=True
    )

    voice_thread.start()

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "✗ Camera could not be opened."
        )

        return

    # ========================================================
    # CURSOR
    # ========================================================

    cursor_x = (
        SCREEN_WIDTH / 2
    )

    cursor_y = (
        SCREEN_HEIGHT / 2
    )

    # ========================================================
    # UI
    # ========================================================

    current_gesture = "NONE"

    gesture_confidence = 0.0

    action_status = "READY"

    status_until = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = (
            cap.read()
        )

        if not ret:

            print(
                "Camera frame error."
            )

            break

        # Mirror camera.

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

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

        # ====================================================
        # FACE FOUND
        # ====================================================

        if result.face_landmarks:

            landmarks = (
                result.face_landmarks[0]
            )

            now = time.time()

            # =================================================
            # HEAD → CURSOR
            # =================================================

            nose = landmarks[1]

            normalized_x = (
                nose.x -
                CONTROL_LEFT
            ) / (
                CONTROL_RIGHT -
                CONTROL_LEFT
            )

            normalized_y = (
                nose.y -
                CONTROL_TOP
            ) / (
                CONTROL_BOTTOM -
                CONTROL_TOP
            )

            normalized_x = max(
                0,
                min(
                    1,
                    normalized_x
                )
            )

            normalized_y = max(
                0,
                min(
                    1,
                    normalized_y
                )
            )

            target_x = (
                normalized_x *
                SCREEN_WIDTH
            )

            target_y = (
                normalized_y *
                SCREEN_HEIGHT
            )

            cursor_x, cursor_y = (
                smooth_cursor(

                    cursor_x,

                    cursor_y,

                    target_x,

                    target_y
                )
            )

            pyautogui.moveTo(

                int(cursor_x),

                int(cursor_y),

                duration=0
            )

            # =================================================
            # BLINK → CLICK
            # =================================================

            left_ear = (
                eye_aspect_ratio(
                    landmarks,
                    LEFT_EYE
                )
            )

            right_ear = (
                eye_aspect_ratio(
                    landmarks,
                    RIGHT_EYE
                )
            )

            ear = (
                left_ear +
                right_ear
            ) / 2

            eyes_closed = (
                ear <
                BLINK_THRESHOLD
            )

            # -------------------------------------------------
            # Eyes closing
            # -------------------------------------------------

            if eyes_closed:

                blink_closed_frames += 1

                if (
                    blink_start is None
                    and
                    blink_closed_frames
                    >=
                    MIN_CLOSED_FRAMES
                ):

                    blink_start = now

            # -------------------------------------------------
            # Eyes opening
            # -------------------------------------------------

            else:

                if blink_start is not None:

                    blink_duration = (
                        now -
                        blink_start
                    )

                    valid_blink = (

                        MIN_BLINK_DURATION
                        <=
                        blink_duration
                        <=
                        MAX_BLINK_DURATION
                    )

                    if (
                        valid_blink
                        and
                        now -
                        last_blink
                        >=
                        BLINK_COOLDOWN
                    ):

                        pyautogui.click()

                        last_blink = now

                        current_gesture = (
                            "BLINK"
                        )

                        gesture_confidence = (
                            1.0
                        )

                        action_status = (
                            "CLICK"
                        )

                        status_until = (
                            now +
                            0.7
                        )

                    blink_start = None

                blink_closed_frames = 0

            # =================================================
            # HEAD TILT
            # =================================================

            left_eye = landmarks[33]

            right_eye = landmarks[263]

            eye_slope = (
                right_eye.y -
                left_eye.y
            )

            # -------------------------------------------------
            # Determine direction
            # -------------------------------------------------

            if (
                eye_slope <
                -TILT_THRESHOLD
            ):

                detected_direction = -1

            elif (
                eye_slope >
                TILT_THRESHOLD
            ):

                detected_direction = 1

            else:

                detected_direction = 0

            # =================================================
            # TILT STABILITY
            # =================================================

            if detected_direction != 0:

                tilt_release_frames = 0

                if (
                    detected_direction
                    ==
                    tilt_direction
                ):

                    tilt_stable_frames += 1

                else:

                    tilt_direction = (
                        detected_direction
                    )

                    tilt_stable_frames = 1

            else:

                tilt_release_frames += 1

                if (
                    tilt_release_frames
                    >=
                    TILT_RELEASE_FRAMES
                ):

                    tilt_direction = 0

                    tilt_stable_frames = 0

            # =================================================
            # TILT CONFIDENCE
            # =================================================

            if (
                tilt_direction != 0
            ):

                gesture_confidence = min(

                    1.0,

                    tilt_stable_frames
                    /
                    TILT_CONFIRM_FRAMES
                )

            else:

                gesture_confidence = 0.0

            # =================================================
            # ADAPTIVE CONTINUOUS SCROLL
            # =================================================

            if (
                tilt_direction != 0
                and
                tilt_stable_frames
                >=
                TILT_CONFIRM_FRAMES
            ):

                if (
                    now -
                    last_scroll_time
                    >=
                    SCROLL_INTERVAL
                ):

                    # -----------------------------------------
                    # Calculate tilt strength
                    # -----------------------------------------

                    tilt_strength = abs(
                        eye_slope
                    )

                    # -----------------------------------------
                    # Additional speed for stronger tilt
                    #
                    # Normal:
                    #     5
                    #
                    # Strong:
                    #     6
                    #
                    # Very strong:
                    #     7
                    #
                    # Maximum:
                    #     8
                    # -----------------------------------------

                    extra_speed = int(
                        max(
                            0,
                            min(
                                3,
                                (
                                    tilt_strength
                                    -
                                    TILT_THRESHOLD
                                )
                                /
                                0.04
                            )
                        )
                    )

                    scroll_amount = (
                        SCROLL_STEP +
                        extra_speed
                    )

                    # -----------------------------------------
                    # TILT UP
                    # -----------------------------------------

                    if tilt_direction == -1:

                        pyautogui.scroll(
                            scroll_amount
                        )

                        current_gesture = (
                            "TILT UP"
                        )

                        action_status = (
                            f"SCROLL UP "
                            f"x{scroll_amount}"
                        )

                    # -----------------------------------------
                    # TILT DOWN
                    # -----------------------------------------

                    else:

                        pyautogui.scroll(
                            -scroll_amount
                        )

                        current_gesture = (
                            "TILT DOWN"
                        )

                        action_status = (
                            f"SCROLL DOWN "
                            f"x{scroll_amount}"
                        )

                    last_scroll_time = now

                    status_until = (
                        now +
                        0.3
                    )

            # =================================================
            # NEUTRAL
            # =================================================

            elif (
                tilt_direction == 0
                and
                blink_start is None
            ):

                current_gesture = (
                    "NONE"
                )

                gesture_confidence = 0.0

        # ====================================================
        # NO FACE
        # ====================================================

        else:

            current_gesture = (
                "NO FACE"
            )

            gesture_confidence = 0.0

            blink_start = None

            blink_closed_frames = 0

            tilt_direction = 0

            tilt_stable_frames = 0

            tilt_release_frames = 0

        # ====================================================
        # VOICE QUEUE
        # ====================================================

        while not voice_queue.empty():

            try:

                command = (
                    voice_queue
                    .get_nowait()
                )

                execute_voice_command(
                    command
                )

                action_status = (
                    command.upper()
                )

                status_until = (
                    time.time()
                    +
                    1.0
                )

            except queue.Empty:

                break

        # ====================================================
        # RESET STATUS
        # ====================================================

        if (
            time.time()
            >
            status_until
        ):

            action_status = (
                "READY"
            )

        # ====================================================
        # UI
        # ====================================================

        h, w = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # Control zone
        # ----------------------------------------------------

        left = int(
            CONTROL_LEFT * w
        )

        right = int(
            CONTROL_RIGHT * w
        )

        top = int(
            CONTROL_TOP * h
        )

        bottom = int(
            CONTROL_BOTTOM * h
        )

        cv2.rectangle(

            frame,

            (left, top),

            (right, bottom),

            (255, 255, 255),

            2
        )

        # ====================================================
        # TITLE
        # ====================================================

        cv2.putText(

            frame,

            "ACCESSAI NEXUS",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2
        )

        # ====================================================
        # GESTURE
        # ====================================================

        cv2.putText(

            frame,

            f"Gesture: "
            f"{current_gesture}",

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (255, 255, 255),

            2
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        cv2.putText(

            frame,

            f"Stability: "
            f"{gesture_confidence:.0%}",

            (20, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (255, 255, 255),

            2
        )

        # ====================================================
        # ACTION
        # ====================================================

        cv2.putText(

            frame,

            f"Action: "
            f"{action_status}",

            (20, 130),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (255, 255, 255),

            2
        )

        # ====================================================
        # VOICE
        # ====================================================

        voice_status = (

            "VOICE: ON"

            if voice_running

            else

            "VOICE: OFF"
        )

        cv2.putText(

            frame,

            voice_status,

            (20, 160),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2
        )

        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(

            frame,

            "HEAD = CURSOR",

            (20, h - 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            2
        )

        cv2.putText(

            frame,

            "BLINK = CLICK",

            (200, h - 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            2
        )

        cv2.putText(

            frame,

            "TILT = SCROLL",

            (370, h - 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            2
        )

        cv2.putText(

            frame,

            "Q = QUIT",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            2
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "AccessAI Nexus",
            frame
        )

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if key == ord("q"):

            break

        if not voice_running:

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    try:

        landmarker.close()

    except Exception:

        pass

    try:

        audio.terminate()

    except Exception:

        pass

    print()

    print(
        "AccessAI Nexus stopped."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()