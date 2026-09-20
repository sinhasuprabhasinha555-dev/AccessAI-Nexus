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
import re

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

# Faster-Whisper is optional at import time so the facial system can
# still start if the voice package is unavailable.
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    WHISPER_AVAILABLE = False


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
# VOICE — FASTER-WHISPER
# ============================================================

VOICE_ENABLED = True

# Windows default microphone through SoundCard/WASAPI.
# This is the microphone path that worked reliably during testing.
VOICE_SAMPLE_RATE = 16000
VOICE_CHANNELS = 1

# Short-command capture. The endpoint detector stops after a brief silence,
# so commands do not wait for a long fixed recording window.
VOICE_RECORD_SECONDS = 2.2
VOICE_IDLE_PAUSE = 0.25
VOICE_COOLDOWN = 0.10

# Local Whisper settings. base.en is a good accuracy/speed balance for
# English voice commands on CPU. beam_size=1 keeps command recognition fast.
VOICE_MODEL_SIZE = "base.en"
VOICE_COMPUTE_TYPE = "int8"
VOICE_BEAM_SIZE = 1

# Bias Whisper toward the small vocabulary used by AccessAI. This helps keep
# command words such as Chrome, calculator and AccessAI intact.
VOICE_INITIAL_PROMPT = (
    "AccessAI computer commands: open Chrome, open calculator, open Documents, "
    "go back, go forward, refresh, reload, new tab, close tab, next tab, "
    "previous tab, scroll up, scroll down, click, left click, enter, space, "
    "home, stop AccessAI."
)

voice_stop_event = threading.Event()
voice_thread = None
voice_model = None

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
    """Normalize Whisper output before intent matching."""
    command = (text or "").lower().strip()

    # Remove punctuation Whisper may append to a short command.
    command = re.sub(r"[^a-z0-9\s-]", " ", command)
    command = command.replace("access-ai", "accessai")
    command = command.replace("access ai", "accessai")
    command = command.replace("access i", "accessai")
    command = command.replace("access eye", "accessai")
    command = " ".join(command.split())
    return command


def _fuzzy_match(command, aliases, threshold=0.78):
    """Return True when Whisper's short transcript closely matches an alias."""
    for alias in aliases:
        if command == alias:
            return True

        # Fuzzy matching is only used for multi-word commands. This avoids
        # dangerous false positives for tiny commands such as "click".
        if len(alias.split()) >= 2 and len(command) >= 5:
            ratio = difflib.SequenceMatcher(None, command, alias).ratio()
            if ratio >= threshold:
                return True
    return False


def _looks_like_stop_command(command):
    """Conservative emergency-stop matching."""
    exact = {
        "stop",
        "stop access",
        "stop accessai",
        "stop assistant",
        "exit accessai",
        "quit accessai",
    }
    if command in exact:
        return True

    if command.startswith("stop "):
        remainder = command[5:].strip()
        allowed = {"access", "accessai", "assistant", "the assistant"}
        if remainder in allowed:
            return True

        # Whisper can slightly distort the final word, but only accept a
        # close match when the transcript already begins with "stop".
        target = command.replace(" ", "")
        candidates = ["stopaccessai", "stopaccess", "stopassistant"]
        return any(
            difflib.SequenceMatcher(None, target, candidate).ratio() >= 0.82
            for candidate in candidates
        )

    return False


def execute_voice_command(text):
    """Convert Whisper transcripts into a small, safe AccessAI command set."""
    global running

    command = _normalize_voice_command(text)
    print(f"✓ Whisper: {command}")

    # --------------------------------------------------------
    # STOP — checked first and intentionally reliable.
    # --------------------------------------------------------
    if _looks_like_stop_command(command):
        set_status("VOICE → STOP ACCESSAI", 2)
        running = False
        voice_stop_event.set()
        return

    # --------------------------------------------------------
    # CHROME
    # --------------------------------------------------------
    if _fuzzy_match(command, {
        "open chrome", "launch chrome", "start chrome",
        "open google chrome", "open chrom", "open crow", "open flow",
    }, threshold=0.76) or command == "chrome":
        open_chrome()
        return

    # --------------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------------
    if _fuzzy_match(command, {
        "open calculator", "open calc", "launch calculator",
        "start calculator", "open the calculator",
    }, threshold=0.76):
        open_calculator()
        return

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------
    if _fuzzy_match(command, {
        "open documents", "open document", "open my documents",
        "open the documents", "open my document",
    }, threshold=0.76):
        open_documents()
        return

    # --------------------------------------------------------
    # BROWSER NAVIGATION
    # --------------------------------------------------------
    if _fuzzy_match(command, {
        "go back", "back", "go backward", "backward", "go bake"
    }, threshold=0.76):
        pyautogui.hotkey("alt", "left")
        set_status("VOICE → GO BACK")
        return

    if _fuzzy_match(command, {
        "go forward", "forward", "go forwards", "move forward"
    }, threshold=0.76):
        pyautogui.hotkey("alt", "right")
        set_status("VOICE → GO FORWARD")
        return

    if _fuzzy_match(command, {
        "refresh", "reload", "refresh page", "reload page"
    }, threshold=0.76):
        pyautogui.press("f5")
        set_status("VOICE → REFRESH")
        return

    # --------------------------------------------------------
    # CHROME TAB CONTROL
    # --------------------------------------------------------
    if _fuzzy_match(command, {
        "new tab", "open new tab", "create new tab", "new browser tab"
    }, threshold=0.76):
        pyautogui.hotkey("ctrl", "t")
        set_status("VOICE → NEW TAB")
        return

    if _fuzzy_match(command, {
        "close tab", "close this tab", "close browser tab"
    }, threshold=0.76):
        pyautogui.hotkey("ctrl", "w")
        set_status("VOICE → CLOSE TAB")
        return

    if _fuzzy_match(command, {
        "next tab", "switch tab", "next browser tab"
    }, threshold=0.76):
        pyautogui.hotkey("ctrl", "tab")
        set_status("VOICE → NEXT TAB")
        return

    if _fuzzy_match(command, {
        "previous tab", "last tab", "previous browser tab"
    }, threshold=0.76):
        pyautogui.hotkey("ctrl", "shift", "tab")
        set_status("VOICE → PREVIOUS TAB")
        return

    # --------------------------------------------------------
    # SCROLL
    # --------------------------------------------------------
    if _fuzzy_match(command, {
        "scroll up", "scroll upward", "scroll upwards", "scroll to top"
    }, threshold=0.76):
        pyautogui.scroll(12)
        set_status("VOICE → SCROLL UP")
        return

    if _fuzzy_match(command, {
        "scroll down", "scroll downward", "scroll downwards"
    }, threshold=0.76):
        pyautogui.scroll(-12)
        set_status("VOICE → SCROLL DOWN")
        return

    # --------------------------------------------------------
    # CLICK — intentionally exact/alias based, not fuzzy.
    # --------------------------------------------------------
    if command in {
        "click", "left click", "click here", "click link",
        "click on link", "select"
    }:
        focus_window_under_cursor()
        time.sleep(0.08)
        pyautogui.click(button="left")
        set_status("VOICE → CLICK")
        return

    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------
    if command in {"enter", "press enter", "hit enter"}:
        pyautogui.press("enter")
        set_status("VOICE → ENTER")
        return

    if command in {"space", "press space", "hit space"}:
        pyautogui.press("space")
        set_status("VOICE → SPACE")
        return

    if command in {"home", "go home", "press home"}:
        pyautogui.press("home")
        set_status("VOICE → HOME")
        return

    print("  Command not in AccessAI allow-list.")


def _record_audio_with_soundcard():
    """Capture a short command from the Windows default microphone."""
    if not SOUNDCARD_AVAILABLE:
        raise RuntimeError(
            "SoundCard is not installed. Run: python -m pip install soundcard numpy"
        )

    microphone = sc.default_microphone()
    if microphone is None:
        raise RuntimeError("Windows default microphone was not found.")

    chunk_frames = int(VOICE_SAMPLE_RATE * 0.10)
    max_frames = int(VOICE_RECORD_SECONDS * VOICE_SAMPLE_RATE)
    pre_roll_chunks = 3
    silence_chunks_needed = max(1, int(VOICE_IDLE_PAUSE / 0.10))
    min_speech_chunks = 2

    chunks = []
    speech_started = False
    speech_chunks = 0
    silent_chunks = 0
    total_frames = 0
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
                if (
                    speech_chunks >= min_speech_chunks
                    and silent_chunks >= silence_chunks_needed
                ):
                    break

            if time.monotonic() - start_time >= VOICE_RECORD_SECONDS:
                break

    if not chunks:
        raise RuntimeError("No microphone audio was captured.")

    samples = np.concatenate(chunks, axis=0)
    samples = samples[:max_frames]

    if speech_started:
        keep_tail = int(VOICE_SAMPLE_RATE * 0.15)
        cut = max(0, len(samples) - silent_chunks * chunk_frames)
        samples = samples[:min(len(samples), cut + keep_tail)]

    samples = np.asarray(samples, dtype=np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)
    return samples


def _transcribe_with_whisper(audio):
    """Fast local transcription with Faster-Whisper."""
    segments, _ = voice_model.transcribe(
        audio,
        language="en",
        beam_size=VOICE_BEAM_SIZE,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        without_timestamps=True,
        initial_prompt=VOICE_INITIAL_PROMPT,
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


def voice_loop():
    """Dedicated local Whisper worker; it never blocks the camera loop."""
    global voice_status
    global last_voice_text
    global last_voice_time
    global voice_model

    if not WHISPER_AVAILABLE:
        voice_status = "WHISPER MISSING"
        print("⚠ Faster-Whisper is not installed.")
        print("   Run: python -m pip install faster-whisper")
        return

    if not SOUNDCARD_AVAILABLE:
        voice_status = "SOUNDCARD MISSING"
        print("⚠ SoundCard is not installed.")
        return

    try:
        voice_status = "LOADING WHISPER"
        print()
        print("🎙 Loading Faster-Whisper...")
        print(f"   Model: {VOICE_MODEL_SIZE}")
        print(f"   Compute: {VOICE_COMPUTE_TYPE}")
        voice_model = WhisperModel(
            VOICE_MODEL_SIZE,
            device="cpu",
            compute_type=VOICE_COMPUTE_TYPE,
        )
        print("   ✓ Whisper model loaded")
        print("   ✓ Voice pipeline: WASAPI → Whisper → intent matcher")
        print()
    except Exception as e:
        voice_status = "WHISPER ERROR"
        print(f"✗ Could not load Faster-Whisper: {type(e).__name__}: {e}")
        return

    while not voice_stop_event.is_set() and running:
        try:
            voice_status = "LISTENING"
            audio = _record_audio_with_soundcard()

            if voice_stop_event.is_set() or not running:
                break

            voice_status = "RECOGNIZING"
            started = time.monotonic()
            text = _transcribe_with_whisper(audio)
            elapsed = time.monotonic() - started

            if not text:
                print("   — No speech recognized.")
                voice_status = "VOICE READY"
                continue

            normalized = _normalize_voice_command(text)
            now = time.time()

            # Avoid executing the same recognized command twice if a user
            # repeats it quickly or Whisper emits overlapping segments.
            if (
                normalized == last_voice_text
                and now - last_voice_time < 1.5
            ):
                print("   Duplicate command ignored.")
            else:
                last_voice_text = normalized
                last_voice_time = now
                print(f"   ✓ Heard in {elapsed:.2f}s: {normalized}")
                execute_voice_command(text)

            voice_status = "VOICE READY"
            voice_stop_event.wait(VOICE_COOLDOWN)

        except Exception as e:
            # Voice errors must never terminate the facial system.
            print(f"   ✗ Voice error: {type(e).__name__}: {e}")
            voice_status = "VOICE ERROR"
            voice_stop_event.wait(0.5)

    voice_status = "VOICE STOPPED"
    print("🎙 Voice thread stopped.")


def start_voice_thread():
    global voice_thread
    global voice_started

    if not VOICE_ENABLED:
        print("Voice control disabled.")
        return

    if not WHISPER_AVAILABLE:
        print("⚠ Faster-Whisper unavailable; voice disabled.")
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
    print("  Voice: Windows/WASAPI → Faster-Whisper → intent matcher")
    print("  Voice: local/offline recognition")
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
