"""
AccessAI v4 — Hands-Free Keyboard & Text Input Layer

IMPORTANT:
- This is a new file. It does NOT modify accessai_v3.py or accessai_STABLE.py.
- Use this as the integration layer for the next AccessAI milestone.
- The existing face engine / calibration / Whisper engine from v3 can be
  copied into the marked integration section.

New capabilities:
  1. "typing mode" -> voice is treated as text
  2. "exit typing mode" -> return to command mode
  3. "type hello world" -> types the requested text once
  4. Keyboard commands: Enter, Backspace, Space, Tab, Escape
  5. Editing commands: Select All, Copy, Paste, Delete
  6. Emergency stop has highest priority

This file is intentionally self-contained for testing the new input layer.
"""

import re
import time
import pyautogui

APP_NAME = "AccessAI"
typing_mode = False
running = True


# -----------------------------
# Text cleanup / normalization
# -----------------------------

def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_typed_text(text: str) -> str:
    """Clean a Whisper transcript before sending it to the keyboard."""
    text = text.strip()

    # Small speech-to-text conveniences.
    replacements = {
        "new line": "\n",
        "newline": "\n",
        "tab character": "\t",
    }

    lower = text.lower()
    for spoken, symbol in replacements.items():
        lower = lower.replace(spoken, symbol)

    # Preserve normal capitalization as much as possible.
    return lower


# -----------------------------
# Keyboard action layer
# -----------------------------

SPECIAL_KEYS = {
    "enter": "enter",
    "return": "enter",
    "backspace": "backspace",
    "delete": "delete",
    "space": "space",
    "tab": "tab",
    "escape": "esc",
    "esc": "esc",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
}


def press_key(key: str) -> bool:
    key = SPECIAL_KEYS.get(normalize(key), normalize(key))
    try:
        pyautogui.press(key)
        return True
    except Exception as exc:
        print(f"[KEYBOARD ERROR] {exc}")
        return False


def type_text(text: str) -> bool:
    if not text:
        return False
    try:
        pyautogui.write(text, interval=0.005)
        return True
    except Exception as exc:
        print(f"[TYPE ERROR] {exc}")
        return False


def hotkey(*keys: str) -> bool:
    try:
        pyautogui.hotkey(*keys)
        return True
    except Exception as exc:
        print(f"[HOTKEY ERROR] {exc}")
        return False


# -----------------------------
# Intent handling
# -----------------------------

def is_emergency_stop(text: str) -> bool:
    t = normalize(text)
    stop_phrases = {
        "stop accessai",
        "stop access ai",
        "emergency stop",
        "emergency",
        "exit accessai",
        "quit accessai",
    }
    return t in stop_phrases


def handle_keyboard_intent(transcript: str):
    """
    Returns:
      (handled: bool, should_exit: bool)
    """
    global typing_mode, running

    raw = transcript.strip()
    t = normalize(raw)

    # Safety / emergency command gets first priority.
    if is_emergency_stop(raw):
        running = False
        typing_mode = False
        print("[ACCESSAI] Emergency stop activated.")
        return True, True

    # Mode switching.
    if t in {
        "typing mode",
        "start typing mode",
        "enable typing mode",
        "enter typing mode",
        "start typing",
    }:
        typing_mode = True
        print("[ACCESSAI] TYPING MODE ON")
        return True, False

    if t in {
        "exit typing mode",
        "stop typing mode",
        "disable typing mode",
        "end typing mode",
        "stop typing",
    }:
        typing_mode = False
        print("[ACCESSAI] COMMAND MODE")
        return True, False

    # In typing mode, ordinary speech is text.
    # Keyboard/editing commands remain available explicitly.
    if typing_mode:
        if t == "press enter":
            press_key("enter")
            return True, False
        if t == "press backspace":
            press_key("backspace")
            return True, False
        if t == "press space":
            press_key("space")
            return True, False
        if t == "press tab":
            press_key("tab")
            return True, False
        if t in {"press escape", "press esc"}:
            press_key("esc")
            return True, False
        if t == "select all":
            hotkey("ctrl", "a")
            return True, False
        if t == "copy":
            hotkey("ctrl", "c")
            return True, False
        if t == "paste":
            hotkey("ctrl", "v")
            return True, False
        if t in {"delete", "press delete"}:
            press_key("delete")
            return True, False

        # Everything else is spoken text.
        typed = clean_typed_text(raw)
        if typed:
            type_text(typed + " ")
            return True, False

    # One-shot typing in command mode.
    m = re.match(r"^(?:type|write|enter text)\s+(.+)$", raw, flags=re.I)
    if m:
        text_to_type = clean_typed_text(m.group(1))
        type_text(text_to_type)
        return True, False

    # Explicit keyboard commands in command mode.
    if t in {"press enter", "enter"}:
        press_key("enter")
        return True, False

    if t in {"press backspace", "backspace"}:
        press_key("backspace")
        return True, False

    if t in {"press space", "space"}:
        press_key("space")
        return True, False

    if t in {"press tab", "tab"}:
        press_key("tab")
        return True, False

    if t in {"press escape", "press esc", "escape"}:
        press_key("esc")
        return True, False

    if t in {"select all", "select everything"}:
        hotkey("ctrl", "a")
        return True, False

    if t in {"copy", "copy that"}:
        hotkey("ctrl", "c")
        return True, False

    if t in {"paste", "paste that"}:
        hotkey("ctrl", "v")
        return True, False

    if t in {"delete", "delete that"}:
        press_key("delete")
        return True, False

    return False, False


# -----------------------------
# Integration point for v3
# -----------------------------

def process_whisper_transcript(transcript: str):
    """
    Call this from the v3 Whisper loop.

    Example:
        transcript = whisper_result_text
        handled, should_exit = process_whisper_transcript(transcript)

    If handled is False, pass the transcript to v3's existing AI intent
    engine so commands such as "open chrome", "go back", etc. continue working.
    """
    if not transcript:
        return False, False

    handled, should_exit = handle_keyboard_intent(transcript)

    if handled:
        return True, should_exit

    # IMPORTANT:
    # Do not replace v3's existing intent engine here.
    # Return False so the existing v3 command pipeline can process it.
    return False, False


# -----------------------------
# Standalone test mode
# -----------------------------

def standalone_test():
    print("\n" + "=" * 58)
    print("ACCESSAI v4 — HANDS-FREE INPUT TEST")
    print("=" * 58)
    print("This test uses typed commands instead of the microphone.")
    print("It is safe to test the input layer before integrating Whisper.")
    print("\nTry:")
    print('  type hello world')
    print('  typing mode')
    print('  this sentence will be typed')
    print('  press enter')
    print('  select all')
    print('  copy')
    print('  paste')
    print('  exit typing mode')
    print('  emergency stop')
    print("\nOpen Notepad (or another text field) before testing.")
    print("=" * 58)

    while running:
        try:
            transcript = input("\nVoice transcript simulation > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not transcript:
            continue

        handled, should_exit = process_whisper_transcript(transcript)

        if not handled:
            print("[V3 INTENT ENGINE] Not a keyboard/text intent.")
            print("→ Pass this transcript to accessai_v3.py's existing intent engine.")

        if should_exit:
            break

    print("[ACCESSAI] Test ended.")


if __name__ == "__main__":
    standalone_test()
