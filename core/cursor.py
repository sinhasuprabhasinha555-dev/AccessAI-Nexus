import pyautogui
import numpy as np


class CursorController:
    def __init__(self, screen_width, screen_height, smoothing=0.25):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.smoothing = smoothing

        self.current_x = screen_width // 2
        self.current_y = screen_height // 2

    def move(self, face_x, face_y):
        # Convert face position (0–1) to screen coordinates
        target_x = np.interp(
            face_x,
            [0.2, 0.8],
            [0, self.screen_width]
        )

        target_y = np.interp(
            face_y,
            [0.2, 0.8],
            [0, self.screen_height]
        )

        # Smooth cursor movement
        self.current_x += (target_x - self.current_x) * self.smoothing
        self.current_y += (target_y - self.current_y) * self.smoothing

        pyautogui.moveTo(
            int(self.current_x),
            int(self.current_y),
            duration=0
        )