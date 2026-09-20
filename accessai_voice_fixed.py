import os
import sys
import json
import math
import time
import threading
import subprocess
import wave
import tempfile
import pyaudio

import cv2
import mediapipe as mp
import pyautogui

# SpeechRecognition is optional at import time so the facial system can
# still start if the voice package is unavailable.
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_AVAILABLE = False


# ============================================================
# ACCESSAI NEXUS
# Hands-Free Computer Accessibility
#
# FACE
#   Head movement -> cursor
#   Blink         -> left click
#   Head tilt     -> scroll
#
# VOICE
#   Open Chrome
#   Open Calculator
#   Open Documents
#   Go back / forward
#   Refresh
#   Scroll up / down
#   Click
#   Enter / Space
#   Stop AccessAI
#
# IMPORTANT FIX
# Voice recognition runs in ONE background thread and has a
# strict SpeechRecognition operation timeout. It never runs
# inside the camera loop, so a Google/network delay cannot
# freeze the camera window.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
CALIBRATION_PATH = os.path.join(BASE_DIR, "calibration_profile.json")


# ============================================================
# CAMERA / PERFORMANCE
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 1

# Keep OpenCV camera buffer small to reduce latency.
CAMERA_BUFFER_SIZE = 1


# ============================================================
# SCREEN
# ============================================================

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

current_x = SCREEN_WIDTH / 2
current_y = SCREEN_HEIGHT / 2

CURSOR_SMOOTHING = 0.28
MAX_CURSOR_STEP = 100
CURSOR_DEADZONE = 0.005


# ============================================================
# VISUAL CONTROL BOX
# ============================================================

BOX_LEFT = 0.20
BOX_RIGHT = 0.80
BOX_TOP = 0.12
BOX_BOTTOM = 0.88


# ============================================================
# CENTER LOCK
# ============================================================

CENTER_LOCK_ENTER_X = 0.060
CENTER_LOCK_ENTER_Y = 0.055

CENTER_LOCK_EXIT_X = 0.080
CENTER_LOCK_EXIT_Y = 0.075

center_locked = False


# ============================================================
# FALLBACK CALIBRATION
# These are the user's latest calibration values.
# calibration_profile.json is preferred when available.
# ============================================================

CALIBRATION = {
    "center_x": 0.4088,
    "center_y": 0.5114,
    "left_x": 0.2157,
    "right_x": 0.5380,
    "up_y": 0.2805,
    "down_y": 0.7689,
}


def load_calibration():
    data = dict(CALIBRATION)

    if not os.path.exists(CALIBRATION_PATH):
        print("⚠ calibration_profile.json not found.")
        print("  Using the latest saved calibration values.")
        return data

    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # Accept either the current naming convention or the
        # common uppercase convention used by earlier versions.
        aliases = {
            "center_x": ["center_x", "CENTER_X", "CENTER", "center"],
            "center_y": ["center_y", "CENTER_Y"],
            "left_x": ["left_x", "LEFT_X", "left"],
            "right_x": ["right_x", "RIGHT_X", "right"],
            "up_y": ["up_y", "UP_Y", "up"],
            "down_y": ["down_y", "DOWN_Y", "down"],
        }

        for key, possible in aliases.items():
            for name in possible:
                if name in loaded:
                    value = loaded[name]

                    # Some calibration files may store CENTER as
                    # [x, y].
                    if key == "center_x" and isinstance(value, (list, tuple)):
                        value = value[0]
                    elif key == "center_y" and isinstance(value, (list, tuple)):
                        value = value[1]

                    data[key] = float(value)
                    break

        print("✓ Calibration loaded")
        print(
            f"  CENTER: ({data['center_x']:.4f}, "
            f"{data['center_y']:.4f})"
        )
        print(
            f"  LEFT:   {data['left_x']:.4f}   "
            f"RIGHT: {data['right_x']:.4f}"
        )
        print(
            f"  UP:     {data['up_y']:.4f}   "
            f"DOWN: {data['down_y']:.4f}"
        )

    except Exception as e:
        print(f"⚠ Calibration file could not be read: {e}")
        print("  Using the latest saved calibration values.")

    return data


CAL = load_calibration()

CENTER_X = CAL["center_x"]
CENTER_Y = CAL["center_y"]
LEFT_X = CAL["left_x"]
RIGHT_X = CAL["right_x"]
UP_Y = CAL["up_y"]
DOWN_Y = CAL["down_y"]

# Use the larger side of the calibration span so the mapping is
# symmetric around the user's real center.
HORIZONTAL_SPAN = max(
    abs(CENTER_X - LEFT_X),
    abs(RIGHT_X - CENTER_X),
)

VERTICAL_SPAN = max(
    abs(CENTER_Y - UP_Y),
    abs(DOWN_Y - CENTER_Y),
)

if HORIZONTAL_SPAN < 0.03:
    HORIZONTAL_SPAN = 0.20

if VERTICAL_SPAN < 0.03:
    VERTICAL_SPAN = 0.20


# ============================================================
# BLINK
# ============================================================

BLINK_THRESHOLD = 0.20
MIN_BLINK_DURATION = 0.08
MAX_BLINK_DURATION = 0.80
BLINK_COOLDOWN = 1.0

blink_start = None
last_blink = 0.0


# ============================================================
# SCROLL
# ============================================================

TILT_THRESHOLD = 0.10
SCROLL_STEP = 5

SCROLL_INTERVAL = 0.08

TILT_CONFIRM_FRAMES = 4
TILT_RELEASE_FRAMES = 3

tilt_state = 0
tilt_candidate = 0
tilt_candidate_frames = 0
tilt_release_frames = 0

last_scroll_time = 0.0


# ============================================================
# VOICE
# ============================================================

VOICE_ENABLED = True

# Device 0 was the microphone path that previously produced
# successful recognition in testing.
VOICE_DEVICE_INDEX = 0

VOICE_SAMPLE_RATE = 44100
VOICE_RECORD_SECONDS = 2.5

VOICE_COOLDOWN = 1.5
VOICE_IDLE_PAUSE = 0.5

# IMPORTANT:
# SpeechRecognition applies this timeout to operations such as
# the Google request. This prevents an indefinitely blocked
# recognition call.
VOICE_OPERATION_TIMEOUT = 5

voice_stop_event = threading.Event()
voice_thread = None

last_voice_text = ""
last_voice_time = 0.0

# Prevent multiple voice workers from ever being started.
voice_started = False


# ============================================================
# APPLICATION STATE
# ============================================================

running = True
frame_counter = 0

status_text = "READY"
status_until = 0.0

face_detected = False
voice_status = "VOICE READY"


def set_status(message, seconds=1.5):
    global status_text, status_until
    status_text = message
    status_until = time.time() + seconds
    print(message)


# ============================================================
# FACE HELPERS
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


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
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (2 * horizontal)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# CURSOR MAPPING
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def map_cursor(face_x, face_y):
    """
    Map the calibrated face center to the exact screen center.

    This deliberately does NOT use LEFT_X -> 0 and RIGHT_X -> 1
    directly because the user's calibration center is not exactly
    halfway between those measured points.
    """

    global center_locked

    dx = face_x - CENTER_X
    dy = face_y - CENTER_Y

    # Center deadzone.
    if (
        abs(dx) <= CENTER_LOCK_ENTER_X
        and abs(dy) <= CENTER_LOCK_ENTER_Y
    ):
        center_locked = True

    elif (
        abs(dx) > CENTER_LOCK_EXIT_X
        or abs(dy) > CENTER_LOCK_EXIT_Y
    ):
        center_locked = False

    if center_locked:
        return SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2

    normalized_x = 0.5 + (
        dx / (2.0 * HORIZONTAL_SPAN)
    )

    normalized_y = 0.5 + (
        dy / (2.0 * VERTICAL_SPAN)
    )

    normalized_x = clamp(normalized_x, 0.0, 1.0)
    normalized_y = clamp(normalized_y, 0.0, 1.0)

    return (
        normalized_x * (SCREEN_WIDTH - 1),
        normalized_y * (SCREEN_HEIGHT - 1),
    )


def move_cursor(target_x, target_y):
    global current_x, current_y

    if center_locked:
        current_x = SCREEN_WIDTH / 2
        current_y = SCREEN_HEIGHT / 2

        pyautogui.moveTo(
            int(current_x),
            int(current_y),
            duration=0
        )
        return

    dx = target_x - current_x
    dy = target_y - current_y

    if abs(dx) < SCREEN_WIDTH * CURSOR_DEADZONE:
        dx = 0

    if abs(dy) < SCREEN_HEIGHT * CURSOR_DEADZONE:
        dy = 0

    # Limit a single frame's cursor jump.
    distance_to_target = math.hypot(dx, dy)

    if distance_to_target > MAX_CURSOR_STEP:
        scale = MAX_CURSOR_STEP / distance_to_target
        dx *= scale
        dy *= scale

    current_x += dx * CURSOR_SMOOTHING
    current_y += dy * CURSOR_SMOOTHING

    current_x = clamp(current_x, 0, SCREEN_WIDTH - 1)
    current_y = clamp(current_y, 0, SCREEN_HEIGHT - 1)

    pyautogui.moveTo(
        int(current_x),
        int(current_y),
        duration=0
    )


# ============================================================
# VOICE ACTIONS
# ============================================================

def open_chrome():
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", "chrome"],
            shell=False
        )
        set_status("VOICE → OPEN CHROME")
    except Exception:
        try:
            os.startfile("https://www.google.com")
            set_status("VOICE → OPEN CHROME")
        except Exception as e:
            print(f"✗ Could not open Chrome: {e}")


def open_calculator():
    try:
        subprocess.Popen(["calc.exe"])
        set_status("VOICE → OPEN CALCULATOR")
    except Exception as e:
        print(f"✗ Could not open Calculator: {e}")


def open_documents():
    try:
        os.startfile(os.path.expanduser("~/Documents"))
        set_status("VOICE → OPEN DOCUMENTS")
    except Exception as e:
        print(f"✗ Could not open Documents: {e}")


def execute_voice_command(text):
    global running

    command = " ".join(text.lower().strip().split())

    print(f"✓ Recognized: {command}")

    # More specific commands first.
    if "stop accessai" in command or "stop access ai" in command:
        set_status("VOICE → STOP ACCESSAI", 2)
        running = False
        voice_stop_event.set()
        return

    if (
        "open chrome" in command
        or "launch chrome" in command
        or "start chrome" in command
    ):
        open_chrome()
        return

    if (
        "open calculator" in command
        or "open calc" in command
        or "launch calculator" in command
    ):
        open_calculator()
        return

    if (
        "open documents" in command
        or "open document" in command
        or "open my documents" in command
    ):
        open_documents()
        return

    if "go back" in command or command == "back":
        pyautogui.hotkey("alt", "left")
        set_status("VOICE → GO BACK")
        return

    if "go forward" in command or command == "forward":
        pyautogui.hotkey("alt", "right")
        set_status("VOICE → GO FORWARD")
        return

    if "refresh" in command or "reload" in command:
        pyautogui.press("f5")
        set_status("VOICE → REFRESH")
        return

    if "scroll up" in command:
        pyautogui.scroll(5)
        set_status("VOICE → SCROLL UP")
        return

    if "scroll down" in command:
        pyautogui.scroll(-5)
        set_status("VOICE → SCROLL DOWN")
        return

    if (
        command == "click"
        or command == "left click"
        or "click here" in command
    ):
        pyautogui.click()
        set_status("VOICE → CLICK")
        return

    if command == "enter" or "press enter" in command:
        pyautogui.press("enter")
        set_status("VOICE → ENTER")
        return

    if command == "space" or "press space" in command:
        pyautogui.press("space")
        set_status("VOICE → SPACE")
        return

    if command == "home" or "go home" in command:
        pyautogui.press("home")
        set_status("VOICE → HOME")
        return

    print("  Command not in allow-list.")


# ============================================================
# VOICE THREAD
# ============================================================

def _record_wav_with_pyaudio():
    """
    Proven voice path:
        PyAudio -> PCM frames -> WAV file

    We deliberately avoid sr.Microphone() and avoid constructing
    sr.AudioData from manually converted bytes.  The WAV is then
    decoded
    by SpeechRecognition's AudioFile reader.
    """
    pa = pyaudio.PyAudio()
    stream = None
    wav_path = None

    # Device 0 was the microphone path that previously produced
    # successful recognition. These are only fallbacks if it cannot
    # be opened on the current run.
    candidate_devices = [VOICE_DEVICE_INDEX, 1, 2, 8, 14, 15]
    candidate_devices = list(dict.fromkeys(candidate_devices))

    rate = VOICE_SAMPLE_RATE
    channels = 2
    sample_width = pa.get_sample_size(pyaudio.paInt16)
    chunk = 1024

    try:
        selected_device = None

        for device_index in candidate_devices:
            try:
                info = pa.get_device_info_by_index(device_index)

                if int(info.get("maxInputChannels", 0)) < 1:
                    continue

                # Prefer the known 44.1 kHz path. If it rejects that
                # rate, the fallback below tries the device's default.
                try:
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk,
                    )
                except Exception:
                    default_rate = int(info.get("defaultSampleRate", rate))
                    default_channels = min(
                        2, int(info.get("maxInputChannels", channels))
                    )
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=default_channels,
                        rate=default_rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk,
                    )
                    rate = default_rate
                    channels = default_channels

                selected_device = device_index
                break

            except Exception as e:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                    stream = None

                print(f"   Mic device {device_index} unavailable: {e}")

        if stream is None:
            raise RuntimeError("No usable microphone input device could be opened.")

        try:
            info = pa.get_device_info_by_index(selected_device)
            device_name = info.get("name", f"Device {selected_device}")
        except Exception:
            device_name = f"Device {selected_device}"

        print(f"   ✓ Microphone opened: {selected_device} - {device_name}")
        print(f"   Format: {rate} Hz, {channels} channel(s), 16-bit")

        frames = []
        total_chunks = int(rate / chunk * VOICE_RECORD_SECONDS)

        print("🎙️ RECORDING...")
        for _ in range(total_chunks):
            if voice_stop_event.is_set() or not running:
                break

            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        if not frames:
            raise RuntimeError("No microphone audio frames were captured.")

        # Save the exact PCM frames to a standard WAV file.
        fd, wav_path = tempfile.mkstemp(
            prefix="accessai_voice_",
            suffix=".wav"
        )
        os.close(fd)

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

        print(f"   ✓ Audio captured ({len(frames)} chunks)")
        print("   ✓ WAV created")

        return wav_path

    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        pa.terminate()


def voice_loop():
    """
    Dedicated voice worker.

    Critical design rule:
    Nothing in this function is executed by the camera/UI loop.
    A slow microphone or Google request therefore cannot freeze
    the OpenCV window.

    Voice path:
        PyAudio -> WAV -> SpeechRecognition AudioFile -> Google
    """
    global voice_status
    global last_voice_text
    global last_voice_time

    if not SPEECH_AVAILABLE:
        voice_status = "VOICE PACKAGE MISSING"
        print("⚠ SpeechRecognition is not installed.")
        return

    recognizer = sr.Recognizer()
    recognizer.operation_timeout = VOICE_OPERATION_TIMEOUT

    print()
    print("🎙 Voice thread started.")
    print(f"   Preferred microphone device index: {VOICE_DEVICE_INDEX}")
    print(f"   Recording: {VOICE_RECORD_SECONDS:.1f}s")
    print(f"   Google operation timeout: {VOICE_OPERATION_TIMEOUT}s")
    print("   Pipeline: PyAudio → WAV → Google SpeechRecognition")
    print()

    while not voice_stop_event.is_set() and running:
        wav_path = None

        try:
            voice_status = "LISTENING"

            wav_path = _record_wav_with_pyaudio()

            if voice_stop_event.is_set() or not running:
                break

            voice_status = "RECOGNIZING"
            print("   ☁ Sending WAV to Google...")

            # Read the WAV through SpeechRecognition's tested AudioFile
            # path rather than manually constructing AudioData.
            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)

            text = recognizer.recognize_google(
                audio,
                language="en-US"
            )

            now = time.time()
            normalized = text.strip().lower()

            if (
                normalized == last_voice_text
                and now - last_voice_time < 3.0
            ):
                print("   Duplicate command ignored.")
            else:
                last_voice_text = normalized
                last_voice_time = now
                execute_voice_command(text)

            voice_status = "VOICE READY"

            voice_stop_event.wait(VOICE_COOLDOWN)

        except sr.WaitTimeoutError:
            print("   Microphone timeout.")
            voice_status = "VOICE READY"

        except sr.UnknownValueError:
            print("   ✗ Google could not understand the speech.")
            voice_status = "VOICE READY"

        except sr.RequestError as e:
            print(f"   ✗ Google recognition unavailable: {e}")
            voice_status = "GOOGLE UNAVAILABLE"
            voice_stop_event.wait(3.0)

        except Exception as e:
            # Voice errors must never terminate the facial system.
            print(f"   ✗ Voice error: {type(e).__name__}: {e}")
            voice_status = "VOICE ERROR"
            voice_stop_event.wait(2.0)

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    voice_status = "VOICE STOPPED"
    print("🎙 Voice thread stopped.")


def start_voice_thread():
    global voice_thread
    global voice_started

    if not VOICE_ENABLED:
        print("Voice control disabled.")
        return

    if not SPEECH_AVAILABLE:
        print("⚠ SpeechRecognition unavailable; voice disabled.")
        return

    if voice_started:
        return

    voice_started = True

    voice_thread = threading.Thread(
        target=voice_loop,
        name="AccessAI-Voice",
        daemon=True
    )

    voice_thread.start()


# ============================================================
# MEDIAPIPE
# ============================================================

def create_landmarker():
    if not os.path.exists(MODEL_PATH):
        print()
        print("ERROR: Face model not found:")
        print(MODEL_PATH)
        print()
        raise SystemExit(1)

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

    return FaceLandmarker.create_from_options(options)


# ============================================================
# DRAWING
# ============================================================

def draw_ui(frame, ear=None, scroll_status="SCROLL READY"):
    h, w = frame.shape[:2]

    # Control box.
    x1 = int(w * BOX_LEFT)
    x2 = int(w * BOX_RIGHT)
    y1 = int(h * BOX_TOP)
    y2 = int(h * BOX_BOTTOM)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    # Center marker.
    cx = int(w * 0.5)
    cy = int(h * 0.5)

    cv2.line(
        frame,
        (cx - 12, cy),
        (cx + 12, cy),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (cx, cy - 12),
        (cx, cy + 12),
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "ACCESSAI NEXUS",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    face_text = "FACE: ACTIVE" if face_detected else "FACE: SEARCHING"

    cv2.putText(
        frame,
        face_text,
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"VOICE: {voice_status}",
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "HEAD = CURSOR",
        (18, h - 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "BLINK = CLICK",
        (18, h - 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "TILT = SCROLL",
        (18, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1
    )

    if ear is not None:
        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (w - 125, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

    if time.time() < status_until:
        cv2.putText(
            frame,
            status_text,
            (w - 250, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

    if center_locked:
        cv2.putText(
            frame,
            "CENTER LOCK",
            (w - 135, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():
    global running
    global frame_counter
    global face_detected
    global blink_start
    global last_blink
    global last_scroll_time
    global tilt_state
    global tilt_candidate
    global tilt_candidate_frames
    global tilt_release_frames

    print()
    print("=" * 60)
    print("                    ACCESSAI NEXUS")
    print("=" * 60)
    print()
    print("FACE")
    print("  Head movement  -> Cursor")
    print("  Blink          -> Left click")
    print("  Head tilt      -> Scroll")
    print()
    print("VOICE")
    print("  Open Chrome / Calculator / Documents")
    print("  Click / Scroll / Back / Forward / Refresh")
    print("  Enter / Space / Stop AccessAI")
    print()
    print("Performance:")
    print(f"  Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print("  Voice: isolated background thread")
    print("  Voice: PyAudio → WAV → Google SpeechRecognition")
    print("  Voice network timeout:", VOICE_OPERATION_TIMEOUT, "seconds")
    print()
    print("Press Q in the camera window to quit.")
    print("=" * 60)
    print()

    # Disable PyAutoGUI's default pause so cursor movement is responsive.
    pyautogui.PAUSE = 0

    landmarker = None
    cap = None

    try:
        print("Loading face model...")
        landmarker = create_landmarker()
        print("✓ Face model loaded")

        print("Starting camera...")

        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not cap.isOpened():
            # Fallback if DirectShow is unavailable.
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            raise RuntimeError("Could not access camera.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("✓ Camera started")
        print()

        # Start voice ONLY after camera initialization.
        # The voice thread is completely independent of the
        # camera processing loop.
        start_voice_thread()

        print("AccessAI is running.")
        print()

        while running:

            success, frame = cap.read()

            if not success:
                print("⚠ Camera frame read failed.")
                time.sleep(0.05)
                continue

            frame_counter += 1

            # Mirror camera.
            frame = cv2.flip(frame, 1)

            ear_value = None
            scroll_status = "SCROLL READY"

            # Process every frame by default. Set to 2 if CPU usage
            # becomes high on a slower computer.
            should_process = (
                frame_counter % PROCESS_EVERY_N_FRAMES == 0
            )

            if should_process:

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                try:
                    result = landmarker.detect(mp_image)
                except Exception as e:
                    print(f"⚠ Face detection error: {e}")
                    result = None

                if result is not None and result.face_landmarks:

                    face_detected = True
                    landmarks = result.face_landmarks[0]

                    # --------------------------------------------
                    # 1. HEAD -> CURSOR
                    # --------------------------------------------

                    nose = landmarks[1]

                    face_x = float(nose.x)
                    face_y = float(nose.y)

                    target_x, target_y = map_cursor(
                        face_x,
                        face_y
                    )

                    move_cursor(
                        target_x,
                        target_y
                    )

                    # --------------------------------------------
                    # 2. BLINK -> CLICK
                    # --------------------------------------------

                    left_ear = eye_aspect_ratio(
                        landmarks,
                        LEFT_EYE
                    )

                    right_ear = eye_aspect_ratio(
                        landmarks,
                        RIGHT_EYE
                    )

                    ear_value = (
                        left_ear + right_ear
                    ) / 2.0

                    now = time.time()

                    if ear_value < BLINK_THRESHOLD:

                        if blink_start is None:
                            blink_start = now

                    else:

                        if blink_start is not None:

                            blink_duration = (
                                now - blink_start
                            )

                            valid_blink = (
                                MIN_BLINK_DURATION
                                <= blink_duration
                                <= MAX_BLINK_DURATION
                            )

                            cooldown_finished = (
                                now - last_blink
                                >= BLINK_COOLDOWN
                            )

                            if (
                                valid_blink
                                and cooldown_finished
                            ):
                                print("BLINK -> LEFT CLICK")
                                pyautogui.click()
                                set_status(
                                    "FACE → CLICK",
                                    0.8
                                )
                                last_blink = now

                            blink_start = None

                    # --------------------------------------------
                    # 3. HEAD TILT -> SCROLL
                    # --------------------------------------------

                    left_eye = landmarks[33]
                    right_eye = landmarks[263]

                    eye_slope = (
                        right_eye.y - left_eye.y
                    )

                    if eye_slope > TILT_THRESHOLD:
                        candidate = 1
                    elif eye_slope < -TILT_THRESHOLD:
                        candidate = -1
                    else:
                        candidate = 0

                    # Hysteresis / stability:
                    # require several frames before changing state.
                    if candidate != 0:

                        tilt_release_frames = 0

                        if candidate == tilt_candidate:
                            tilt_candidate_frames += 1
                        else:
                            tilt_candidate = candidate
                            tilt_candidate_frames = 1

                        if (
                            tilt_candidate_frames
                            >= TILT_CONFIRM_FRAMES
                        ):
                            tilt_state = candidate

                    else:

                        tilt_candidate_frames = 0
                        tilt_candidate = 0
                        tilt_release_frames += 1

                        if (
                            tilt_release_frames
                            >= TILT_RELEASE_FRAMES
                        ):
                            tilt_state = 0

                    now = time.time()

                    if (
                        tilt_state != 0
                        and now - last_scroll_time
                        >= SCROLL_INTERVAL
                    ):

                        if tilt_state > 0:
                            pyautogui.scroll(-SCROLL_STEP)
                            scroll_status = "SCROLL DOWN"
                        else:
                            pyautogui.scroll(SCROLL_STEP)
                            scroll_status = "SCROLL UP"

                        last_scroll_time = now

                    elif tilt_state > 0:
                        scroll_status = "SCROLL DOWN"

                    elif tilt_state < 0:
                        scroll_status = "SCROLL UP"

                else:
                    face_detected = False
                    blink_start = None
                    tilt_state = 0
                    tilt_candidate = 0
                    tilt_candidate_frames = 0
                    tilt_release_frames = 0

            # --------------------------------------------
            # DISPLAY
            # --------------------------------------------

            frame = draw_ui(
                frame,
                ear=ear_value,
                scroll_status=scroll_status
            )

            cv2.imshow(
                "AccessAI Nexus",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                running = False
                voice_stop_event.set()
                break

            # ESC also exits.
            if key == 27:
                running = False
                voice_stop_event.set()
                break

    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt received.")
        running = False
        voice_stop_event.set()

    except Exception as e:
        print()
        print("ERROR:", type(e).__name__)
        print(e)
        running = False
        voice_stop_event.set()

    finally:
        print()
        print("Stopping AccessAI...")

        running = False
        voice_stop_event.set()

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        cv2.destroyAllWindows()

        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass

        # Do not wait indefinitely for a network operation.
        if voice_thread is not None and voice_thread.is_alive():
            voice_thread.join(timeout=1.0)

        print("✓ Camera released")
        print("✓ Voice worker stopped")
        print("✓ AccessAI closed")


if __name__ == "__main__":
    main()
