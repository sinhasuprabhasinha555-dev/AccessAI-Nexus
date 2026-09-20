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
# ACCESSAI NEXUS — INTENT ENGINE
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
    "home, typing mode, exit typing mode, start typing, press enter, press backspace, select all, copy, paste, stop AccessAI."
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

# ============================================================
# HANDS-FREE TEXT INPUT / KEYBOARD MODE
# ============================================================

typing_mode = False

SPECIAL_KEYS = {
    "enter": "enter",
    "return": "enter",
    "backspace": "backspace",
    "delete": "delete",
    "space": "space",
    "tab": "tab",
    "escape": "esc",
    "esc": "esc",
}

def _clean_typed_text(text):
    """Prepare Whisper text for direct keyboard input."""
    text = (text or "").strip()
    # Whisper may return these spoken forms for basic formatting.
    text = re.sub(r"(?i)\bnew line\b", "\n", text)
    text = re.sub(r"(?i)\bnewline\b", "\n", text)
    text = re.sub(r"(?i)\btab character\b", "\t", text)
    return text

def _press_special_key(key):
    key = SPECIAL_KEYS.get((key or "").strip().lower(), (key or "").strip().lower())
    try:
        pyautogui.press(key)
        return True
    except Exception as e:
        print(f"   ✗ Keyboard error: {e}")
        return False

def _handle_text_input(text):
    """
    Handle typing-mode commands before normal intent classification.
    Returns True when the transcript was consumed.
    """
    global typing_mode, running

    raw = (text or "").strip()
    command = raw.lower().strip()
    command = re.sub(r"\s+", " ", command)

    # Emergency stop always has priority.
    if command in {
        "stop accessai", "stop access ai", "exit accessai",
        "quit accessai", "emergency stop", "emergency"
    }:
        typing_mode = False
        running = False
        voice_stop_event.set()
        set_status("VOICE → STOP ACCESSAI", 2)
        return True

    # Enter / exit typing mode.
    if command in {
        "typing mode", "start typing", "start typing mode",
        "enable typing mode", "enter typing mode"
    }:
        typing_mode = True
        set_status("TYPING MODE ON", 2)
        print("⌨ TYPING MODE ON — speak normally to type")
        return True

    if command in {
        "exit typing mode", "stop typing", "stop typing mode",
        "disable typing mode", "end typing mode"
    }:
        typing_mode = False
        set_status("COMMAND MODE", 2)
        print("⌨ COMMAND MODE")
        return True

    # One-shot typing without entering a mode.
    match = re.match(r"^(?:type|write|enter text)\s+(.+)$", raw, flags=re.I)
    if match:
        typed = _clean_typed_text(match.group(1))
        if typed:
            pyautogui.write(typed, interval=0.005)
            set_status("VOICE → TYPE", 1)
        return True

    # If typing mode is active, consume keyboard/editing commands first.
    if typing_mode:
        if command in {"press enter", "enter", "hit enter"}:
            _press_special_key("enter")
            return True
        if command in {"press backspace", "backspace"}:
            _press_special_key("backspace")
            return True
        if command in {"press space", "space", "press the space bar"}:
            _press_special_key("space")
            return True
        if command in {"press tab", "tab"}:
            _press_special_key("tab")
            return True
        if command in {"press escape", "press esc", "escape"}:
            _press_special_key("esc")
            return True
        if command in {"press delete", "delete"}:
            _press_special_key("delete")
            return True
        if command in {"select all", "select everything"}:
            pyautogui.hotkey("ctrl", "a")
            return True
        if command in {"copy", "copy that"}:
            pyautogui.hotkey("ctrl", "c")
            return True
        if command in {"paste", "paste that"}:
            pyautogui.hotkey("ctrl", "v")
            return True

        # Ordinary speech becomes keyboard text.
        typed = _clean_typed_text(raw)
        if typed:
            pyautogui.write(typed + " ", interval=0.005)
            set_status("VOICE → TYPING", 0.7)
        return True

    return False

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
    """Normalize Whisper output for robust local intent classification."""
    command = (text or "").lower().strip()
    command = re.sub(r"[^a-z0-9\s-]", " ", command)
    command = command.replace("access-ai", "accessai")
    command = command.replace("access ai", "accessai")
    command = command.replace("access i", "accessai")
    command = command.replace("access eye", "accessai")
    command = " ".join(command.split())
    return command


# Local intent vocabulary. The engine scores semantic-ish phrase patterns,
# token overlap and fuzzy similarity. It does not send speech to the internet.
INTENT_ALIASES = {
    "OPEN_CHROME": [
        "open chrome", "launch chrome", "start chrome", "open google chrome",
        "open browser", "launch browser", "start browser", "open my browser",
        "i want to browse", "take me to the browser", "open chrom", "open crow",
        "open flow", "chrome",
    ],
    "OPEN_CALCULATOR": [
        "open calculator", "launch calculator", "start calculator", "open calc",
        "open the calculator", "bring up calculator", "show calculator",
    ],
    "OPEN_DOCUMENTS": [
        "open documents", "open document", "open my documents",
        "open the documents", "open my document", "show my documents",
        "bring up documents",
    ],
    "GO_BACK": [
        "go back", "back", "go backward", "go backwards", "move back",
        "return to the previous page", "take me back", "previous page", "go bake",
    ],
    "GO_FORWARD": [
        "go forward", "forward", "go forwards", "move forward",
        "return to the next page", "next page",
    ],
    "REFRESH": [
        "refresh", "reload", "refresh page", "reload page", "refresh this page",
        "reload this page", "update the page",
    ],
    "NEW_TAB": [
        "new tab", "open new tab", "create new tab", "new browser tab",
        "open another tab", "start a new tab",
    ],
    "CLOSE_TAB": [
        "close tab", "close this tab", "close browser tab", "close the tab",
        "shut this tab",
    ],
    "NEXT_TAB": [
        "next tab", "switch tab", "next browser tab", "go to next tab",
        "move to next tab",
    ],
    "PREVIOUS_TAB": [
        "previous tab", "last tab", "previous browser tab", "go to previous tab",
        "go to the last tab",
    ],
    "SCROLL_UP": [
        "scroll up", "scroll upward", "scroll upwards", "scroll to top",
        "move up", "go up", "scroll back up",
    ],
    "SCROLL_DOWN": [
        "scroll down", "scroll downward", "scroll downwards", "move down",
        "go down", "scroll further down", "scroll to bottom",
    ],
    "CLICK": [
        "click", "left click", "click here", "click link", "click on link",
        "select", "press the button", "select this",
    ],
    "ENTER": ["enter", "press enter", "hit enter", "confirm"],
    "SPACE": ["space", "press space", "hit space", "press the space bar"],
    "HOME": ["home", "go home", "press home", "go to home"],
}

# Words that are useful evidence for an intent. These are deliberately
# conservative: an action is only executed after a minimum score is reached.
INTENT_KEYWORDS = {
    "OPEN_CHROME": {"chrome", "browser", "browse", "internet"},
    "OPEN_CALCULATOR": {"calculator", "calc"},
    "OPEN_DOCUMENTS": {"documents", "document", "files"},
    "GO_BACK": {"back", "backward", "previous", "return"},
    "GO_FORWARD": {"forward", "forwards", "next"},
    "REFRESH": {"refresh", "reload", "update"},
    "NEW_TAB": {"new", "tab", "another"},
    "CLOSE_TAB": {"close", "tab", "shut"},
    "NEXT_TAB": {"next", "tab", "switch"},
    "PREVIOUS_TAB": {"previous", "last", "tab", "switch"},
    "SCROLL_UP": {"scroll", "upward", "up", "top"},
    "SCROLL_DOWN": {"scroll", "downward", "down", "bottom"},
    "CLICK": {"click", "select", "button"},
    "ENTER": {"enter", "confirm"},
    "SPACE": {"space", "bar"},
    "HOME": {"home"},
}


def _intent_score(command, alias, keywords):
    """Score a command against one alias using several cheap local signals."""
    if command == alias:
        return 1.0

    command_tokens = set(command.split())
    alias_tokens = set(alias.split())
    overlap = len(command_tokens & alias_tokens) / max(1, len(alias_tokens))
    ratio = difflib.SequenceMatcher(None, command, alias).ratio()

    keyword_hits = len(command_tokens & keywords)
    keyword_score = min(1.0, keyword_hits / 2.0)

    # Phrase/token overlap helps with natural wording; fuzzy similarity helps
    # with Whisper's small spelling distortions (e.g. "bake" for "back").
    return max(
        ratio * 0.72 + overlap * 0.28,
        keyword_score * 0.72 + ratio * 0.28,
    )


def classify_voice_intent(text):
    """Return (intent, confidence, normalized_text) or (None, 0, text)."""
    command = _normalize_voice_command(text)
    if not command:
        return None, 0.0, command

    # Emergency stop is intentionally separate and more conservative.
    if _looks_like_stop_command(command):
        return "STOP", 1.0, command

    best_intent = None
    best_score = 0.0
    second_score = 0.0

    for intent, aliases in INTENT_ALIASES.items():
        intent_best = max(
            _intent_score(command, alias, INTENT_KEYWORDS.get(intent, set()))
            for alias in aliases
        )
        if intent_best > best_score:
            second_score = best_score
            best_score = intent_best
            best_intent = intent
        elif intent_best > second_score:
            second_score = intent_best

    # Require both reasonable confidence and a useful gap over the runner-up.
    # Short commands such as "click" are protected by their explicit aliases.
    min_confidence = 0.72 if len(command.split()) >= 2 else 0.90
    margin_required = 0.07

    if best_intent is None or best_score < min_confidence:
        return None, best_score, command
    if best_score - second_score < margin_required and len(command.split()) >= 2:
        return None, best_score, command

    return best_intent, best_score, command


def _looks_like_stop_command(command):
    """Conservative emergency-stop matching."""
    exact = {
        "stop", "stop access", "stop accessai", "stop assistant",
        "exit accessai", "quit accessai",
    }
    if command in exact:
        return True

    if command.startswith("stop "):
        remainder = command[5:].strip()
        allowed = {"access", "accessai", "assistant", "the assistant"}
        if remainder in allowed:
            return True
        target = command.replace(" ", "")
        candidates = ["stopaccessai", "stopaccess", "stopassistant"]
        return any(
            difflib.SequenceMatcher(None, target, candidate).ratio() >= 0.82
            for candidate in candidates
        )
    return False


def execute_voice_command(text):
    """Handle typing/keyboard input first, then the existing local intents."""
    global running

    # Text input must be checked BEFORE classify_voice_intent(), because
    # ordinary sentences are intentionally not normal AccessAI commands.
    if _handle_text_input(text):
        return

    intent, confidence, command = classify_voice_intent(text)
    print(f"✓ Whisper: {command}")
    print(f"   Intent: {intent or 'UNKNOWN'} | confidence={confidence:.2f}")

    if intent == "STOP":
        set_status("VOICE → STOP ACCESSAI", 2)
        running = False
        voice_stop_event.set()
        return

    if intent == "OPEN_CHROME":
        open_chrome(); return
    if intent == "OPEN_CALCULATOR":
        open_calculator(); return
    if intent == "OPEN_DOCUMENTS":
        open_documents(); return
    if intent == "GO_BACK":
        pyautogui.hotkey("alt", "left"); set_status("VOICE → GO BACK"); return
    if intent == "GO_FORWARD":
        pyautogui.hotkey("alt", "right"); set_status("VOICE → GO FORWARD"); return
    if intent == "REFRESH":
        pyautogui.press("f5"); set_status("VOICE → REFRESH"); return
    if intent == "NEW_TAB":
        pyautogui.hotkey("ctrl", "t"); set_status("VOICE → NEW TAB"); return
    if intent == "CLOSE_TAB":
        pyautogui.hotkey("ctrl", "w"); set_status("VOICE → CLOSE TAB"); return
    if intent == "NEXT_TAB":
        pyautogui.hotkey("ctrl", "tab"); set_status("VOICE → NEXT TAB"); return
    if intent == "PREVIOUS_TAB":
        pyautogui.hotkey("ctrl", "shift", "tab"); set_status("VOICE → PREVIOUS TAB"); return
    if intent == "SCROLL_UP":
        pyautogui.scroll(12); set_status("VOICE → SCROLL UP"); return
    if intent == "SCROLL_DOWN":
        pyautogui.scroll(-12); set_status("VOICE → SCROLL DOWN"); return
    if intent == "CLICK":
        focus_window_under_cursor()
        time.sleep(0.08)
        pyautogui.click(button="left")
        set_status("VOICE → CLICK")
        return
    if intent == "ENTER":
        pyautogui.press("enter"); set_status("VOICE → ENTER"); return
    if intent == "SPACE":
        pyautogui.press("space"); set_status("VOICE → SPACE"); return
    if intent == "HOME":
        pyautogui.press("home"); set_status("VOICE → HOME"); return

    print("  Command not confidently understood — no action taken.")



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
        print("   ✓ Local intent engine enabled")
        print("   ✓ Voice pipeline: WASAPI → Whisper → local intent engine")
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

    if voice_thread is not None and voice_thread.is_alive():
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
        f"VOICE: {voice_status} | " + ("TYPING MODE" if typing_mode else "COMMAND MODE"),
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
# CALIBRATION WIZARD
# ============================================================

def _save_calibration_profile(profile):
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def _refresh_calibration_globals(profile):
    global CAL, CENTER_X, CENTER_Y, LEFT_X, RIGHT_X, UP_Y, DOWN_Y
    global HORIZONTAL_SPAN, VERTICAL_SPAN, center_locked

    CAL = dict(profile)
    CENTER_X = CAL["center_x"]
    CENTER_Y = CAL["center_y"]
    LEFT_X = CAL["left_x"]
    RIGHT_X = CAL["right_x"]
    UP_Y = CAL["up_y"]
    DOWN_Y = CAL["down_y"]

    HORIZONTAL_SPAN = max(abs(CENTER_X - LEFT_X), abs(RIGHT_X - CENTER_X))
    VERTICAL_SPAN = max(abs(CENTER_Y - UP_Y), abs(DOWN_Y - CENTER_Y))

    if HORIZONTAL_SPAN < 0.03:
        HORIZONTAL_SPAN = 0.20
    if VERTICAL_SPAN < 0.03:
        VERTICAL_SPAN = 0.20

    center_locked = False


def _draw_calibration_screen(frame, title, instruction, progress, total, countdown=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)

    h, w = frame.shape[:2]
    cv2.putText(frame, "ACCESSAI  |  CALIBRATION", (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2)
    cv2.putText(frame, title, (28, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(frame, instruction, (28, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1)

    bar_x, bar_y, bar_w, bar_h = 28, h - 62, w - 56, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
    filled = int(bar_w * min(1.0, progress / max(1, total)))
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (255, 255, 255), -1)
    cv2.putText(frame, f"Samples: {progress}/{total}", (28, h - 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    if countdown is not None:
        text = str(max(0, int(math.ceil(countdown))))
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 4)[0]
        cv2.putText(frame, text, ((w - size[0]) // 2, h // 2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4)

    return frame


def run_calibration_wizard(cap, landmarker):
    """Guide the user through center/left/right/up/down calibration.

    This runs on the camera/UI thread so it never races with camera reads.
    Voice is temporarily paused by the caller.
    """
    targets = [
        ("CENTER", "Look straight at the camera and keep your head neutral.", "center_x", "center_y"),
        ("LEFT", "Slowly look LEFT and hold still.", "left_x", None),
        ("RIGHT", "Slowly look RIGHT and hold still.", "right_x", None),
        ("UP", "Slowly look UP and hold still.", "up_y", None),
        ("DOWN", "Slowly look DOWN and hold still.", "down_y", None),
    ]

    samples_needed = 30
    profile = {}

    print()
    print("=" * 60)
    print("ACCESSAI CALIBRATION WIZARD")
    print("Keep your face visible. Move only when instructed.")
    print("Press ESC to cancel calibration.")
    print("=" * 60)

    for index, (name, instruction, key1, key2) in enumerate(targets, start=1):
        samples = []
        countdown_start = time.monotonic()

        while True:
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            elapsed = time.monotonic() - countdown_start
            remaining = 2.0 - elapsed
            display = _draw_calibration_screen(
                frame,
                f"STEP {index}/{len(targets)} — {name}",
                instruction,
                0,
                samples_needed,
                remaining if remaining > 0 else None,
            )
            cv2.imshow("AccessAI Nexus", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key in (ord("q"), ord("Q")):
                print("Calibration cancelled.")
                return False
            if remaining <= 0:
                break

        while len(samples) < samples_needed:
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            try:
                result = landmarker.detect(mp_image)
            except Exception:
                result = None

            nose = None
            if result is not None and result.face_landmarks:
                nose = result.face_landmarks[0][1]

            if nose is not None:
                samples.append((float(nose.x), float(nose.y)))

            progress = len(samples)
            display = _draw_calibration_screen(
                frame,
                f"STEP {index}/{len(targets)} — {name}",
                instruction,
                progress,
                samples_needed,
            )
            cv2.imshow("AccessAI Nexus", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key in (ord("q"), ord("Q")):
                print("Calibration cancelled.")
                return False

        xs = [p[0] for p in samples]
        ys = [p[1] for p in samples]
        avg_x = float(np.median(xs))
        avg_y = float(np.median(ys))

        profile[key1] = avg_x if key1.endswith("_x") else avg_y
        if key2:
            profile[key2] = avg_y

        print(f"✓ {name}: ({avg_x:.4f}, {avg_y:.4f})")

    _save_calibration_profile(profile)
    _refresh_calibration_globals(profile)

    print("✓ Calibration saved to calibration_profile.json")
    print("✓ New calibration is active immediately.")

    end_time = time.monotonic() + 2.0
    while time.monotonic() < end_time:
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cv2.putText(frame, "CALIBRATION COMPLETE", (w // 2 - 190, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Your personalized profile is now active.", (w // 2 - 225, h // 2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.imshow("AccessAI Nexus", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    return True


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
    print("  Typing mode: speak text → keyboard input")
    print("  Keyboard: backspace / tab / select all / copy / paste")
    print()
    print("Performance:")
    print(f"  Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print("  Voice: isolated background thread")
    print("  Voice: Windows/WASAPI → Faster-Whisper → intent matcher")
    print("  Voice: local/offline recognition")
    print()
    print("Press C in the camera window to recalibrate.")
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

            # C opens the personalized calibration wizard.
            # Pause voice while calibration owns the camera/UI loop.
            if key == ord("c") or key == ord("C"):
                print("\nStarting calibration wizard...")
                voice_stop_event.set()
                if run_calibration_wizard(cap, landmarker):
                    set_status("CALIBRATION COMPLETE", 2.5)
                else:
                    set_status("CALIBRATION CANCELLED", 2.0)
                if running:
                    voice_stop_event.clear()
                    start_voice_thread()

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
