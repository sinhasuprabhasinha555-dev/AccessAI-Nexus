import tkinter as tk
from tkinter import ttk
import json
import os
import subprocess
import sys


# ============================================================
# ACCESSAI NEXUS — DASHBOARD
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROFILE_PATH = os.path.join(
    BASE_DIR,
    "calibration_profile.json"
)


# ============================================================
# COLORS
# ============================================================

BG = "#0b1020"
CARD = "#141b2d"
CARD_2 = "#1b2438"
TEXT = "#f5f7ff"
MUTED = "#8d99b8"
ACCENT = "#6c7cff"
SUCCESS = "#39d98a"
WARNING = "#ffc857"


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "AccessAI Nexus"
)

root.geometry(
    "1050x700"
)

root.minsize(
    900,
    620
)

root.configure(
    bg=BG
)


# ============================================================
# STYLE
# ============================================================

style = ttk.Style()

try:
    style.theme_use("clam")
except Exception:
    pass


style.configure(
    "TProgressbar",
    thickness=8
)


# ============================================================
# VARIABLES
# ============================================================

gesture_var = tk.StringVar(
    value="READY"
)

confidence_var = tk.StringVar(
    value="0%"
)

action_var = tk.StringVar(
    value="SYSTEM READY"
)

calibration_var = tk.StringVar(
    value="Checking..."
)


# ============================================================
# HELPERS
# ============================================================

def create_card(parent):

    return tk.Frame(
        parent,
        bg=CARD,
        highlightthickness=1,
        highlightbackground="#27314a"
    )


def status_dot(
    parent,
    text,
    active=True
):

    frame = tk.Frame(
        parent,
        bg=CARD
    )

    dot_color = (
        SUCCESS
        if active
        else WARNING
    )

    dot = tk.Label(
        frame,
        text="●",
        fg=dot_color,
        bg=CARD,
        font=("Segoe UI", 12)
    )

    dot.pack(
        side="left",
        padx=(0, 8)
    )

    label = tk.Label(
        frame,
        text=text,
        fg=TEXT,
        bg=CARD,
        font=("Segoe UI", 11)
    )

    label.pack(
        side="left"
    )

    return frame


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=BG
)

header.pack(
    fill="x",
    padx=35,
    pady=(28, 15)
)


title = tk.Label(
    header,
    text="ACCESSAI",
    fg=TEXT,
    bg=BG,
    font=(
        "Segoe UI",
        28,
        "bold"
    )
)

title.pack(
    anchor="w"
)


subtitle = tk.Label(
    header,
    text=(
        "NEXUS  •  HANDS-FREE "
        "COMPUTER ACCESSIBILITY"
    ),
    fg=ACCENT,
    bg=BG,
    font=(
        "Segoe UI",
        10,
        "bold"
    )
)

subtitle.pack(
    anchor="w",
    pady=(2, 0)
)


# ============================================================
# MAIN CONTENT
# ============================================================

content = tk.Frame(
    root,
    bg=BG
)

content.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=10
)


# ============================================================
# LEFT COLUMN
# ============================================================

left_column = tk.Frame(
    content,
    bg=BG
)

left_column.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)


# ============================================================
# CAMERA CARD
# ============================================================

camera_card = create_card(
    left_column
)

camera_card.pack(
    fill="both",
    expand=True
)


camera_title = tk.Label(
    camera_card,
    text="LIVE CONTROL",
    fg=TEXT,
    bg=CARD,
    font=(
        "Segoe UI",
        14,
        "bold"
    )
)

camera_title.pack(
    anchor="w",
    padx=22,
    pady=(20, 5)
)


camera_subtitle = tk.Label(
    camera_card,
    text=(
        "Facial interaction engine"
    ),
    fg=MUTED,
    bg=CARD,
    font=(
        "Segoe UI",
        9
    )
)

camera_subtitle.pack(
    anchor="w",
    padx=22
)


# ------------------------------------------------------------
# Fake camera area
# ------------------------------------------------------------

camera_area = tk.Frame(
    camera_card,
    bg="#090e1b"
)

camera_area.pack(
    fill="both",
    expand=True,
    padx=22,
    pady=18
)


camera_icon = tk.Label(
    camera_area,
    text="◉",
    fg=ACCENT,
    bg="#090e1b",
    font=(
        "Segoe UI",
        42
    )
)

camera_icon.place(
    relx=0.5,
    rely=0.40,
    anchor="center"
)


camera_message = tk.Label(
    camera_area,
    text=(
        "Camera feed appears\n"
        "when AccessAI is running"
    ),
    fg=MUTED,
    bg="#090e1b",
    justify="center",
    font=(
        "Segoe UI",
        11
    )
)

camera_message.place(
    relx=0.5,
    rely=0.58,
    anchor="center"
)


# ============================================================
# RIGHT COLUMN
# ============================================================

right_column = tk.Frame(
    content,
    bg=BG,
    width=330
)

right_column.pack(
    side="right",
    fill="y"
)

right_column.pack_propagate(
    False
)


# ============================================================
# GESTURE STATUS CARD
# ============================================================

gesture_card = create_card(
    right_column
)

gesture_card.pack(
    fill="x",
    pady=(0, 10)
)


tk.Label(
    gesture_card,
    text="CURRENT GESTURE",
    fg=MUTED,
    bg=CARD,
    font=(
        "Segoe UI",
        9,
        "bold"
    )
).pack(
    anchor="w",
    padx=20,
    pady=(18, 3)
)


tk.Label(
    gesture_card,
    textvariable=gesture_var,
    fg=TEXT,
    bg=CARD,
    font=(
        "Segoe UI",
        22,
        "bold"
    )
).pack(
    anchor="w",
    padx=20
)


tk.Label(
    gesture_card,
    text="Confidence",
    fg=MUTED,
    bg=CARD,
    font=(
        "Segoe UI",
        9
    )
).pack(
    anchor="w",
    padx=20,
    pady=(15, 2)
)


confidence_label = tk.Label(
    gesture_card,
    textvariable=confidence_var,
    fg=SUCCESS,
    bg=CARD,
    font=(
        "Segoe UI",
        18,
        "bold"
    )
)

confidence_label.pack(
    anchor="w",
    padx=20,
    pady=(0, 18)
)


# ============================================================
# ACTION CARD
# ============================================================

action_card = create_card(
    right_column
)

action_card.pack(
    fill="x",
    pady=10
)


tk.Label(
    action_card,
    text="LAST ACTION",
    fg=MUTED,
    bg=CARD,
    font=(
        "Segoe UI",
        9,
        "bold"
    )
).pack(
    anchor="w",
    padx=20,
    pady=(18, 3)
)


tk.Label(
    action_card,
    textvariable=action_var,
    fg=TEXT,
    bg=CARD,
    font=(
        "Segoe UI",
        14,
        "bold"
    )
).pack(
    anchor="w",
    padx=20,
    pady=(0, 18)
)


# ============================================================
# CONTROL MODULES
# ============================================================

controls_card = create_card(
    right_column
)

controls_card.pack(
    fill="x",
    pady=10
)


tk.Label(
    controls_card,
    text="CONTROL MODULES",
    fg=MUTED,
    bg=CARD,
    font=(
        "Segoe UI",
        9,
        "bold"
    )
).pack(
    anchor="w",
    padx=20,
    pady=(18, 10)
)


for text in [
    "Face Cursor",
    "Blink Click",
    "Adaptive Scroll",
    "Voice Commands"
]:

    status_dot(
        controls_card,
        text,
        True
    ).pack(
        anchor="w",
        padx=20,
        pady=5
    )


# ============================================================
# PERSONALIZATION CARD
# ============================================================

personal_card = create_card(
    right_column
)

personal_card.pack(
    fill="x",
    pady=10
)


tk.Label(
    personal_card,
    text="PERSONALIZATION",
    fg=MUTED,
    bg=CARD,
    font=(
        "Segoe UI",
        9,
        "bold"
    )
).pack(
    anchor="w",
    padx=20,
    pady=(18, 5)
)


tk.Label(
    personal_card,
    textvariable=calibration_var,
    fg=SUCCESS,
    bg=CARD,
    font=(
        "Segoe UI",
        10,
        "bold"
    )
).pack(
    anchor="w",
    padx=20,
    pady=(0, 18)
)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Frame(
    root,
    bg=BG
)

footer.pack(
    fill="x",
    padx=35,
    pady=(5, 25)
)


def launch_accessai():

    try:

        subprocess.Popen(
            [
                sys.executable,
                os.path.join(
                    BASE_DIR,
                    "accessai.py"
                )
            ]
        )

        action_var.set(
            "ACCESSAI STARTED"
        )

    except Exception as e:

        action_var.set(
            f"ERROR: {e}"
        )


def recalibrate():

    try:

        subprocess.Popen(
            [
                sys.executable,
                os.path.join(
                    BASE_DIR,
                    "calibration.py"
                )
            ]
        )

        action_var.set(
            "CALIBRATION STARTED"
        )

    except Exception as e:

        action_var.set(
            f"ERROR: {e}"
        )


def close_dashboard():

    root.destroy()


# ============================================================
# BUTTONS
# ============================================================

button_style = {
    "font": (
        "Segoe UI",
        10,
        "bold"
    ),
    "bd": 0,
    "padx": 22,
    "pady": 10,
    "cursor": "hand2"
}


start_button = tk.Button(
    footer,
    text="START ACCESSAI",
    command=launch_accessai,
    bg=ACCENT,
    fg="white",
    activebackground="#5968e8",
    activeforeground="white",
    **button_style
)

start_button.pack(
    side="left"
)


calibrate_button = tk.Button(
    footer,
    text="RECALIBRATE",
    command=recalibrate,
    bg=CARD_2,
    fg=TEXT,
    activebackground="#27314a",
    activeforeground=TEXT,
    **button_style
)

calibrate_button.pack(
    side="left",
    padx=10
)


exit_button = tk.Button(
    footer,
    text="EXIT",
    command=close_dashboard,
    bg=CARD_2,
    fg=MUTED,
    activebackground="#27314a",
    activeforeground=TEXT,
    **button_style
)

exit_button.pack(
    side="right"
)


# ============================================================
# CALIBRATION STATUS
# ============================================================

def check_calibration():

    if os.path.exists(
        PROFILE_PATH
    ):

        try:

            with open(
                PROFILE_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                profile = json.load(file)

            if (
                "calibration"
                in profile
            ):

                calibration_var.set(
                    "✓ CALIBRATION LOADED"
                )

            else:

                calibration_var.set(
                    "⚠ PROFILE INCOMPLETE"
                )

        except Exception:

            calibration_var.set(
                "⚠ PROFILE ERROR"
            )

    else:

        calibration_var.set(
            "⚠ NOT CALIBRATED"
        )


check_calibration()


# ============================================================
# DEMO STATUS ANIMATION
# ============================================================

def demo_status():

    # This is only dashboard preview data.
    # It does NOT control AccessAI.

    states = [
        (
            "READY",
            "0%",
            "SYSTEM READY"
        ),
        (
            "HEAD TRACKING",
            "94%",
            "CURSOR CONTROL"
        ),
        (
            "BLINK",
            "98%",
            "CLICK"
        ),
        (
            "TILT DOWN",
            "96%",
            "SCROLL DOWN"
        ),
        (
            "VOICE",
            "92%",
            "VOICE COMMAND"
        )
    ]

    index = (
        demo_status.counter
        %
        len(states)
    )

    gesture, confidence, action = (
        states[index]
    )

    gesture_var.set(
        gesture
    )

    confidence_var.set(
        confidence
    )

    action_var.set(
        action
    )

    demo_status.counter += 1

    root.after(
        3500,
        demo_status
    )


demo_status.counter = 0

demo_status()


# ============================================================
# START
# ============================================================

root.mainloop()