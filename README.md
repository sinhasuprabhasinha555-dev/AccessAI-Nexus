# AccessAI Nexus

### Hands-Free Computer Accessibility

**AccessAI Nexus** is a hands-free computer interaction system that allows users to control a Windows computer using **facial movements, voice commands, and voice typing**.

The project combines computer vision and speech recognition to provide an alternative to conventional mouse and keyboard interaction.

---

## 🎯 Problem

Traditional computer interfaces rely heavily on a mouse, keyboard, and precise hand movements.

For users who experience difficulty with conventional input methods, an alternative interaction mechanism can make computer use more accessible.

**AccessAI Nexus explores a simple idea:**

> **What if a computer could respond to facial movements and voice instead?**

---

## 💡 Solution

AccessAI Nexus combines two primary input channels:

### 👤 Facial Control

Using a camera and facial landmark detection:

* **Head movement** → Cursor movement
* **Blink** → Left click
* **Head tilt** → Scrolling
* Adaptive calibration for individual users
* Smooth cursor movement and gesture stabilization

### 🎤 Voice Control

Using local speech recognition:

* Open Chrome
* Open Calculator
* Open Documents
* Browser navigation
* Refresh
* New tab / close tab
* Next / previous tab
* Voice scrolling
* Keyboard commands
* Stop AccessAI

### ⌨️ Voice Typing

AccessAI Nexus also supports a dedicated typing mode:

* Speak naturally to type text
* Enter / Backspace / Space / Tab / Escape / Delete
* Select all
* Copy
* Paste
* One-shot voice typing
* Exit typing mode using voice commands

---

## 🧠 System Architecture

```text
                    ┌─────────────────┐
                    │     CAMERA      │
                    └────────┬────────┘
                             ↓
                       MediaPipe
                             ↓
                     Facial Intent
                             │
                             │
                             ├──────────────┐
                             │              │
                             ↓              ↓
                    ACCESSAI INTENT ENGINE
                             ↑              ↑
                             │              │
                             │              │
                       Voice Intent         │
                             ↑              │
                        Faster-Whisper      │
                             ↑              │
                    ┌─────────────────┐     │
                    │   MICROPHONE    │─────┘
                    └─────────────────┘
                             │
                             ↓
                    COMPUTER CONTROL
```

Both facial and voice inputs are converted into intents and then mapped to computer actions.

---

## 🛠️ Technology Stack

| Technology              | Purpose                                            |
| ----------------------- | -------------------------------------------------- |
| **Python**              | Core application                                   |
| **OpenCV**              | Camera processing                                  |
| **MediaPipe**           | Facial landmark and gesture detection              |
| **Faster-Whisper**      | Speech recognition                                 |
| **SoundCard / WASAPI**  | Microphone audio capture                           |
| **PyAutoGUI**           | Mouse and keyboard automation                      |
| **Local Intent Engine** | Converts recognized commands into computer actions |

---

## ✨ Key Features

* 🖱️ Hands-free cursor control
* 👁️ Blink-based clicking
* ↕️ Head-tilt scrolling
* 🎤 Voice commands
* ⌨️ Voice typing
* 🧠 Local intent classification
* 🎯 Adaptive calibration
* ⚡ Gesture stabilization
* 🛑 Emergency stop command
* 🪟 Windows application control

---

## 🎬 Demo

The demonstration showcases:

1. Facial cursor control
2. Blink-based clicking
3. Head-tilt scrolling
4. Voice commands
5. Voice typing
6. Hands-free computer interaction

**Demo Video:**
*https://youtu.be/vei9GdRWxRg*
https://github.com/sinhasuprabhasinha555-dev/AccessAI-Nexus
---

## 🔐 Privacy

AccessAI Nexus is designed around local computer interaction.

The project does not require sending computer-control commands to a remote AI service.

Users should still review the software, dependencies, and configuration before running the project on their own systems.

**Never commit API keys, passwords, ****`.env`**** files, or other sensitive information to the repository.**

---

## 🔮 Future Scope

Planned areas for further development include:

* Personalized user profiles
* Additional facial gestures
* Application-specific command sets
* More accessibility-focused computer actions
* Improved gesture robustness
* Structured real-world usability testing
* Expanded configuration and calibration options

---

## 📌 Project Status

**Current status: Working prototype**

The current prototype demonstrates facial computer control, voice commands, and voice typing on Windows.

The project is intended as an accessibility-focused prototype and is not presented as a replacement for professional assistive technology or clinical evaluation.

---

## 👩‍💻 Creator

**SUPRABHA SINHA**

Built as an accessibility-focused computer interaction project.

---

## 📄 License

Add your preferred open-source license here if you intend to distribute the source code under one.
