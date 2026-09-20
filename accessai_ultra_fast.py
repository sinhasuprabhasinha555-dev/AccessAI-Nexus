import os
import sys
import json
import math
import time
import threading
import subprocess
import wave
import tempfile
import difflib
import ctypes

try:
    import soundcard as sc
    import numpy as np
    SOUNDCARD_AVAILABLE = True
except ImportError:
    sc = None
    np = None
    SOUNDCARD_AVAILABLE = False

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
# ACCESSAI
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

CENTER_LOCK_ENTER_X = 0.035
CENTER_LOCK_ENTER_Y = 0.032

CENTER_LOCK_EXIT_X = 0.060
CENTER_LOCK_EXIT_Y = 0.055

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

# Device 8 is the pTron BT headset input that produced the strongest
# real microphone signal in the latest hardware test.
# AccessAI now uses the Windows default microphone through SoundCard/WASAPI.
# This avoids the corrupted PyAudio/PortAudio input endpoints encountered
# during testing.
VOICE_SAMPLE_RATE = 16000
VOICE_CHANNELS = 1
VOICE_RECORD_SECONDS = 2.2

# Shorter pause/cooldown makes commands feel more immediate.
VOICE_COOLDOWN = 0.00
VOICE_IDLE_PAUSE = 0.12

# IMPORTANT:
# SpeechRecognition applies this timeout to operations such as
# the Google request. This prevents an indefinitely blocked
# recognition call.
VOICE_OPERATION_TIMEOUT = 2.5

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

    # When the face is close to the calibrated center, smoothly anchor the
    # cursor instead of repeatedly jumping between the locked center and a
    # mapped position. The narrower enter zone + wider exit zone above gives
    # hysteresis and prevents center jitter.
    if center_locked:
        target_x = SCREEN_WIDTH / 2
        target_y = SCREEN_HEIGHT / 2

    dx = target_x - current_x
    dy = target_y - current_y

    if abs(dx) < SCREEN_WIDTH * CURSOR_DEADZONE:
        dx = 0

    if abs(dy) < SCREEN_HEIGHT * CURSOR_DEADZONE:
        dy = 0

    distance_to_target = math.hypot(dx, dy)

    if distance_to_target > MAX_CURSOR_STEP:
        scale = MAX_CURSOR_STEP / distance_to_target
        dx *= scale
        dy *= scale

    # Use slightly stronger damping near the center so tiny nose/landmark
    # fluctuations don't look like cursor shaking.
    damping = CURSOR_SMOOTHING
    if abs(dx) < 120 and abs(dy) < 120:
        damping = 0.18

    current_x += dx * damping
    current_y += dy * damping

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

def focus_window_under_cursor():
    """Bring the top-level window under the current cursor to the foreground.

    This is more reliable than searching for a Chrome window because the user
    may have multiple Chrome windows/tabs, and Windows can reject a background
    SetForegroundWindow call. The click should go to whatever the face cursor
    is actually pointing at.
    """
    if os.name != "nt":
        return False

    try:
        user32 = ctypes.windll.user32

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        point = POINT()
        user32.GetCursorPos(ctypes.byref(point))

        hwnd = user32.WindowFromPoint(point)
        if not hwnd:
            return False

        # GA_ROOT = 2
        root = user32.GetAncestor(hwnd, 2)
        if root:
            hwnd = root

        user32.ShowWindow(hwnd, 9)  # SW_RESTORE

        # Attach the current thread to the target window's thread so Windows
        # is more willing to allow the foreground change.
        foreground = user32.GetForegroundWindow()
        current_thread = user32.GetWindowThreadProcessId(
            user32.GetForegroundWindow(), None
        )
        target_thread = user32.GetWindowThreadProcessId(
            hwnd, None
        )

        attached = False
        if current_thread and target_thread and current_thread != target_thread:
            attached = bool(
                user32.AttachThreadInput(
                    current_thread, target_thread, True
                )
            )

        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)

        if attached:
            user32.AttachThreadInput(
                current_thread, target_thread, False
            )

        return True

    except Exception as e:
        print(f"   ⚠ Could not focus window under cursor: {e}")
        return False


def open_chrome():
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", "chrome"],
            shell=False
        )

        # Give Chrome a moment to create its window, then focus it.
        time.sleep(0.45)
        focus_window_under_cursor()
        set_status("VOICE → OPEN CHROME")

    except Exception:
        try:
            os.startfile("https://www.google.com")
            time.sleep(0.45)
            focus_window_under_cursor()
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


def _normalize_voice_command(text):
    """Normalize common speech-recognition spacing/punctuation variations."""
    command = text.lower().strip()
    command = command.replace("access ai", "accessai")
    command = command.replace("access-ai", "accessai")
    command = command.replace("access i", "accessai")
    command = command.replace("access eye", "accessai")
    command = " ".join(command.split())
    return command


def _looks_like_stop_command(command):
    """Accept common short recognition variants for the emergency stop command."""
    if command in {
        "stop",
        "stop access",
        "stop accessai",
        "stop access ai",
        "stop access i",
        "stop access eye",
        "stop assistant",
        "exit accessai",
        "quit accessai",
    }:
        return True

    # A short command such as just "stop" is intentionally sufficient.
    # Google often drops the second half of "stop accessai"; requiring the
    # full phrase makes the emergency stop unnecessarily unreliable.
    if command.startswith("stop") and len(command) <= 24:
        remainder = command[4:].strip()
        allowed_words = {
            "", "access", "accessai", "access ai", "access i",
            "access eye", "assistant", "the assistant"
        }
        if remainder in allowed_words:
            return True

    # Google occasionally produces a small transcription variation.
    # Only compare against stop phrases when the command begins with
    # "stop", so ordinary commands are not accidentally stopped.
    if command.startswith("stop "):
        target = command.replace(" ", "")
        candidates = [
            "stopaccessai",
            "stopaccessai",
            "stopaccess",
            "stopassistant",
        ]
        return any(
            difflib.SequenceMatcher(None, target, candidate).ratio() >= 0.78
            for candidate in candidates
        )

    return False


def execute_voice_command(text):
    global running

    command = _normalize_voice_command(text)

    print(f"✓ Recognized: {command}")

    # --------------------------------------------------------
    # STOP — intentionally forgiving because Google often
    # drops the final "AccessAI" words.
    # --------------------------------------------------------
    if _looks_like_stop_command(command):
        set_status("VOICE → STOP ACCESSAI", 2)
        running = False
        voice_stop_event.set()
        return

    # --------------------------------------------------------
    # CHROME
    # --------------------------------------------------------
    if (
        "open chrome" in command
        or "launch chrome" in command
        or "start chrome" in command
    ):
        open_chrome()
        return

    # --------------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------------
    if (
        "open calculator" in command
        or "open calc" in command
        or "launch calculator" in command
    ):
        open_calculator()
        return

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------
    if (
        "open documents" in command
        or "open document" in command
        or "open my documents" in command
    ):
        open_documents()
        return

    # --------------------------------------------------------
    # BROWSER NAVIGATION
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # CHROME TAB CONTROL
    # These make voice useful for opening/closing tabs and then
    # using the face cursor + click to select links/sites.
    # --------------------------------------------------------
    if (
        "new tab" in command
        or "open new tab" in command
        or "create new tab" in command
    ):
        pyautogui.hotkey("ctrl", "t")
        set_status("VOICE → NEW TAB")
        return

    if (
        "close tab" in command
        or "close this tab" in command
    ):
        pyautogui.hotkey("ctrl", "w")
        set_status("VOICE → CLOSE TAB")
        return

    if (
        "next tab" in command
        or "switch tab" in command
    ):
        pyautogui.hotkey("ctrl", "tab")
        set_status("VOICE → NEXT TAB")
        return

    if (
        "previous tab" in command
        or "last tab" in command
    ):
        pyautogui.hotkey("ctrl", "shift", "tab")
        set_status("VOICE → PREVIOUS TAB")
        return

    # --------------------------------------------------------
    # SCROLL
    # Increased from 5 to 12 wheel units because the previous
    # setting produced only about a line of movement in Chrome.
    # --------------------------------------------------------
    if "scroll up" in command or "scroll upward" in command:
        pyautogui.scroll(12)
        set_status("VOICE → SCROLL UP")
        return

    if "scroll down" in command or "scroll downward" in command:
        pyautogui.scroll(-12)
        set_status("VOICE → SCROLL DOWN")
        return

    # --------------------------------------------------------
    # CLICK
    # "click" acts at the current face-controlled cursor
    # position. In Chrome, place the cursor over the link/button
    # first, then say click.
    # --------------------------------------------------------
    if (
        command == "click"
        or command == "left click"
        or "click here" in command
        or "click link" in command
        or "click on link" in command
        or command == "select"
    ):
        # If Chrome is open, make sure the click goes to Chrome rather than
        # the VS Code terminal that launched AccessAI.
        focus_window_under_cursor()
        time.sleep(0.12)
        pyautogui.click(button="left")
        set_status("VOICE → CLICK")
        return

    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------
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

def _record_wav_with_soundcard():
    """
    Fast speech capture using the Windows default microphone via SoundCard/WASAPI.

    Instead of recording a fixed 5-second block, this uses a small energy-based
    speech endpoint detector. It starts with a short pre-roll, waits for speech,
    and stops after a short silence or the maximum recording time.
    This makes short commands much faster while keeping enough audio for Google.
    """
    if not SOUNDCARD_AVAILABLE:
        raise RuntimeError(
            "SoundCard is not installed. Run: python -m pip install soundcard numpy"
        )

    microphone = sc.default_microphone()
    if microphone is None:
        raise RuntimeError("Windows default microphone was not found.")

    # 100 ms chunks make endpoint detection responsive without excessive overhead.
    chunk_frames = int(VOICE_SAMPLE_RATE * 0.10)
    max_frames = int(VOICE_RECORD_SECONDS * VOICE_SAMPLE_RATE)
    pre_roll_chunks = 3
    silence_chunks_needed = max(1, int(VOICE_IDLE_PAUSE / 0.10))
    min_speech_chunks = 2  # about 200 ms minimum speech

    print(f"   ✓ Windows microphone: {microphone.name}")
    print(f"   Format: {VOICE_SAMPLE_RATE} Hz, mono")
    print("   🎙️ LISTENING — say your command")

    chunks = []
    speech_started = False
    speech_chunks = 0
    silent_chunks = 0
    total_frames = 0

    # First few chunks establish the local noise floor. We still keep them as
    # pre-roll so the first syllable is less likely to be clipped.
    noise_levels = []

    with microphone.recorder(
        samplerate=VOICE_SAMPLE_RATE,
        channels=VOICE_CHANNELS
    ) as recorder:
        start_time = time.monotonic()

        while total_frames < max_frames:
            if voice_stop_event.is_set() or not running:
                break

            data = recorder.record(numframes=chunk_frames)
            if data is None or len(data) == 0:
                continue

            data = np.asarray(data, dtype=np.float32)
            if data.ndim > 1:
                data = data[:, 0]
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = np.clip(data, -1.0, 1.0)

            level = float(np.sqrt(np.mean(data * data)))
            chunks.append(data.copy())
            total_frames += len(data)

            elapsed = time.monotonic() - start_time

            # Estimate noise from the first 200 ms. A minimum threshold prevents
            # a very quiet microphone from becoming oversensitive.
            if not speech_started and len(noise_levels) < pre_roll_chunks:
                noise_levels.append(level)

            noise_floor = float(np.median(noise_levels)) if noise_levels else 0.003
            threshold = max(0.004, noise_floor * 2.2)

            if level >= threshold:
                if not speech_started:
                    speech_started = True
                    print("   🎤 SPEECH DETECTED")
                speech_chunks += 1
                silent_chunks = 0
            elif speech_started:
                silent_chunks += 1

                # Once enough speech has been captured, end the command after
                # a short silence instead of waiting for the full 3 seconds.
                if (
                    speech_chunks >= min_speech_chunks
                    and silent_chunks >= silence_chunks_needed
                ):
                    break

            # Hard safety limit even if there is continuous background noise.
            if elapsed >= VOICE_RECORD_SECONDS:
                break

    if not chunks:
        raise RuntimeError("No microphone audio was captured.")

    samples = np.concatenate(chunks, axis=0)
    samples = samples[:max_frames]

    # If speech was detected, remove excessive trailing silence while keeping
    # a small tail so the final word is not cut off.
    if speech_started:
        keep_tail = int(VOICE_SAMPLE_RATE * 0.15)
        samples = samples[:min(len(samples), total_frames - max(0, silent_chunks * chunk_frames) + keep_tail)]

    samples = np.asarray(samples, dtype=np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)

    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    duration = len(samples) / VOICE_SAMPLE_RATE
    print(f"   Audio RMS : {rms:.5f}")
    print(f"   Audio Peak: {peak:.5f}")
    print(f"   ✓ Captured speech window: {duration:.2f}s")

    # Convert float samples to standard 16-bit PCM WAV.
    pcm = (samples * 32767.0).astype(np.int16)

    fd, wav_path = tempfile.mkstemp(
        prefix="accessai_voice_",
        suffix=".wav"
    )
    os.close(fd)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(VOICE_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    print("   ✓ Windows/WASAPI WAV created")
    return wav_path


def voice_loop():
    """
    Low-latency voice pipeline.

    Capture and speech recognition are separated:
      microphone capture -> command queue -> recognition worker -> action

    The microphone immediately starts listening for the next command while
    the previous command is being sent to Google. This removes the old
    record -> recognize -> wait -> record cycle and makes back-to-back
    commands much more responsive.

    Important: Google/network recognition still has unavoidable latency.
    This version optimizes the pipeline; it cannot guarantee 0.1 s
    end-to-end recognition over the internet.
    """
    global voice_status, last_voice_text, last_voice_time

    if not SPEECH_AVAILABLE:
        voice_status = "VOICE PACKAGE MISSING"
        print("⚠ SpeechRecognition is not installed.")
        return

    if not SOUNDCARD_AVAILABLE:
        voice_status = "SOUNDCARD MISSING"
        print("⚠ SoundCard is not installed.")
        return

    import queue

    command_queue = queue.Queue(maxsize=3)
    recognizer = sr.Recognizer()
    recognizer.operation_timeout = VOICE_OPERATION_TIMEOUT

    print()
    print("🎙 Voice thread started — LOW LATENCY MODE")
    print("   Microphone: Windows default input (SoundCard/WASAPI)")
    print(f"   Endpoint silence: {VOICE_IDLE_PAUSE:.2f}s")
    print("   Pipeline: MIC → QUEUE → GOOGLE → ACTION")
    print("   Next command capture starts immediately.")
    print()

    def capture_loop():
        """Continuously capture short speech segments."""
        nonlocal command_queue

        microphone = sc.default_microphone()
        if microphone is None:
            raise RuntimeError("Windows default microphone was not found.")

        chunk_frames = int(VOICE_SAMPLE_RATE * 0.05)  # 50 ms
        max_frames = int(VOICE_RECORD_SECONDS * VOICE_SAMPLE_RATE)
        silence_needed = max(1, int(VOICE_IDLE_PAUSE / 0.05))
        min_speech_chunks = 2

        with microphone.recorder(
            samplerate=VOICE_SAMPLE_RATE,
            channels=VOICE_CHANNELS
        ) as recorder:

            while not voice_stop_event.is_set() and running:
                chunks = []
                speech_started = False
                speech_chunks = 0
                silent_chunks = 0
                total_frames = 0

                # Use a very short rolling noise calibration.
                noise_levels = []

                while total_frames < max_frames:
                    if voice_stop_event.is_set() or not running:
                        return

                    data = recorder.record(numframes=chunk_frames)
                    if data is None or len(data) == 0:
                        continue

                    data = np.asarray(data, dtype=np.float32)
                    if data.ndim > 1:
                        data = data[:, 0]
                    data = np.nan_to_num(
                        data, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    data = np.clip(data, -1.0, 1.0)

                    level = float(np.sqrt(np.mean(data * data)))
                    chunks.append(data.copy())
                    total_frames += len(data)

                    if not speech_started and len(noise_levels) < 4:
                        noise_levels.append(level)

                    noise_floor = (
                        float(np.median(noise_levels))
                        if noise_levels else 0.003
                    )
                    threshold = max(0.004, noise_floor * 2.0)

                    if level >= threshold:
                        if not speech_started:
                            speech_started = True
                            voice_status = "LISTENING"
                            print("   🎤 SPEECH")
                        speech_chunks += 1
                        silent_chunks = 0
                    elif speech_started:
                        silent_chunks += 1
                        if (
                            speech_chunks >= min_speech_chunks
                            and silent_chunks >= silence_needed
                        ):
                            break

                if not speech_started:
                    voice_status = "VOICE READY"
                    continue

                samples = np.concatenate(chunks, axis=0)
                samples = samples[:max_frames]

                # Keep a tiny tail so the last syllable is preserved.
                keep_tail = int(VOICE_SAMPLE_RATE * 0.08)
                trim_frames = max(0, silent_chunks * chunk_frames)
                end_frame = max(
                    1, min(len(samples), len(samples) - trim_frames + keep_tail)
                )
                samples = samples[:end_frame]

                samples = np.asarray(samples, dtype=np.float32)
                samples = np.clip(
                    np.nan_to_num(
                        samples, nan=0.0, posinf=0.0, neginf=0.0
                    ),
                    -1.0,
                    1.0
                )

                pcm = (samples * 32767.0).astype(np.int16)

                fd, wav_path = tempfile.mkstemp(
                    prefix="accessai_fast_voice_",
                    suffix=".wav"
                )
                os.close(fd)

                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(VOICE_SAMPLE_RATE)
                    wf.writeframes(pcm.tobytes())

                # Never block microphone capture on recognition.
                try:
                    command_queue.put_nowait(wav_path)
                    voice_status = "RECOGNIZING"
                except queue.Full:
                    print("   ⚠ Voice queue full — dropping oldest command.")
                    try:
                        old = command_queue.get_nowait()
                        if os.path.exists(old):
                            os.remove(old)
                    except Exception:
                        pass
                    command_queue.put_nowait(wav_path)

    def recognition_loop():
        """Recognize queued commands while capture continues."""
        nonlocal command_queue

        while not voice_stop_event.is_set() and running:
            try:
                wav_path = command_queue.get(timeout=0.1)
            except queue.Empty:
                if running:
                    voice_status_local = "VOICE READY"
                    # Avoid assigning global here repeatedly.
                continue

            try:
                print("   ☁ RECOGNIZING...")
                with sr.AudioFile(wav_path) as source:
                    audio = recognizer.record(source)

                try:
                    text_result = recognizer.recognize_google(
                        audio, language="en-IN"
                    )
                except sr.UnknownValueError:
                    text_result = recognizer.recognize_google(
                        audio, language="en-US"
                    )

                now = time.time()
                normalized = text_result.strip().lower()

                print(f"   ✓ Recognized: {normalized}")

                if (
                    normalized != last_voice_text
                    or now - last_voice_time >= 3.0
                ):
                    last_voice_text = normalized
                    last_voice_time = now
                    execute_voice_command(text_result)
                else:
                    print("   Duplicate command ignored.")

                if running:
                    globals()["voice_status"] = "VOICE READY"

            except sr.UnknownValueError:
                print("   ✗ Speech not understood.")
                globals()["voice_status"] = "VOICE READY"

            except sr.RequestError as e:
                print(f"   ✗ Google recognition unavailable: {e}")
                globals()["voice_status"] = "GOOGLE UNAVAILABLE"

            except Exception as e:
                print(f"   ✗ Voice recognition error: {type(e).__name__}: {e}")
                globals()["voice_status"] = "VOICE ERROR"

            finally:
                if wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
                command_queue.task_done()

    capture_thread = threading.Thread(
        target=capture_loop,
        name="AccessAI-VoiceCapture",
        daemon=True
    )

    recognition_thread = threading.Thread(
        target=recognition_loop,
        name="AccessAI-VoiceRecognition",
        daemon=True
    )

    try:
        capture_thread.start()
        recognition_thread.start()

        while running and not voice_stop_event.is_set():
            time.sleep(0.05)

    except Exception as e:
        print(f"   ✗ Voice pipeline error: {type(e).__name__}: {e}")

    finally:
        voice_stop_event.set()
        globals()["voice_status"] = "VOICE STOPPED"
        print("🎙 Voice pipeline stopped.")


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
    """Polished hackathon presentation overlay; control logic is unchanged."""
    h, w = frame.shape[:2]

    def put(txt, x, y, scale=0.45, thick=1):
        cv2.putText(
            frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (255, 255, 255), thick, cv2.LINE_AA
        )

    # Header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 68), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)

    put("ACCESSAI", 18, 29, 0.72, 2)
    put("HANDS-FREE COMPUTER ACCESSIBILITY", 20, 51, 0.36, 1)

    face_badge = "FACE  ACTIVE" if face_detected else "FACE  SEARCHING"
    put(face_badge, w - 205, 29, 0.40, 1)
    put("VOICE  " + voice_status, w - 205, 51, 0.36, 1)

    # Main control zone
    x1, x2 = int(w * BOX_LEFT), int(w * BOX_RIGHT)
    y1, y2 = int(h * BOX_TOP), int(h * BOX_BOTTOM)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    # Corner accents
    c = 14
    for a, b in [
        ((x1, y1), (x1 + c, y1)), ((x1, y1), (x1, y1 + c)),
        ((x2, y1), (x2 - c, y1)), ((x2, y1), (x2, y1 + c)),
        ((x1, y2), (x1 + c, y2)), ((x1, y2), (x1, y2 - c)),
        ((x2, y2), (x2 - c, y2)), ((x2, y2), (x2, y2 - c)),
    ]:
        cv2.line(frame, a, b, (255, 255, 255), 2, cv2.LINE_AA)

    # Center target
    cx, cy = int(w * 0.5), int(h * 0.5)
    cv2.circle(frame, (cx, cy), 18, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (cx - 27, cy), (cx + 27, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 27), (cx, cy + 27), (255, 255, 255), 1)

    # Status bar
    cv2.rectangle(frame, (12, 78), (w - 12, 116), (0, 0, 0), 1)
    current = status_text if time.time() < status_until else "READY FOR HANDS-FREE CONTROL"
    put("STATUS", 22, 102, 0.33, 1)
    put(current, 78, 102, 0.42, 1)
    if center_locked:
        put("CENTER LOCK", w - 122, 102, 0.33, 1)

    # Bottom feature strip
    strip = h - 82
    cv2.rectangle(frame, (0, strip), (w, h), (0, 0, 0), -1)
    put("FACE CONTROLS", 14, strip + 19, 0.33, 1)
    put("HEAD  ->  CURSOR", 14, strip + 41, 0.39, 1)
    put("BLINK  ->  CLICK", 14, strip + 62, 0.39, 1)

    divider = int(w * 0.50)
    cv2.line(frame, (divider, strip + 10), (divider, h - 10), (255, 255, 255), 1)
    put("HEAD TILT", divider + 14, strip + 19, 0.33, 1)
    put(scroll_status, divider + 14, strip + 41, 0.39, 1)
    put("VOICE  ->  COMMANDS", divider + 14, strip + 62, 0.39, 1)

    # Small diagnostics
    if ear is not None:
        put(f"EAR {ear:.3f}", w - 88, 132, 0.30, 1)
    put("Q / ESC  EXIT", w - 105, h - 91, 0.29, 1)

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
    print("                    ACCESSAI")
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
    print("  Voice: Windows/WASAPI → WAV → Google SpeechRecognition")
    print("  Voice network timeout:", VOICE_OPERATION_TIMEOUT, "seconds")
    print()
    print("Press Q in the camera window to quit.")
    print()
    print("DEMO FLOW")
    print("  Head movement  -> Cursor")
    print("  Blink          -> Click")
    print("  Head tilt      -> Scroll")
    print("  Voice          -> App and browser commands")
    print()
    print("=" * 64)
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
                "AccessAI",
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
