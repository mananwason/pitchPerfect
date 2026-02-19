"""
Calibration tool — take a screenshot and let you tap to find button positions.
Run this to identify the correct coordinates for your device's Hinge layout.

Usage:
  python3 calibrate.py

It will capture a screenshot, then let you interactively identify button positions.
"""

import os
import sys
import json
from device import detect_device


CALIBRATION_FILE = "hinge_buttons.json"

DEFAULT_BUTTONS = {
    "like": {"x_pct": 0.90, "y_pct": 0.67, "description": "Heart/Like button"},
    "dislike": {"x_pct": 0.15, "y_pct": 0.85, "description": "X/Skip button"},
    "comment_field": {"x_pct": 0.50, "y_pct": 0.77, "description": "Comment text field"},
    "send": {"x_pct": 0.85, "y_pct": 0.50, "description": "Send/Submit button"},
}


def load_buttons(width, height):
    """Load calibrated button positions, or use defaults."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        buttons = {}
        for name, coords in data.items():
            buttons[name] = (int(coords["x"]), int(coords["y"]))
        print(f"Loaded calibrated buttons from {CALIBRATION_FILE}")
        return buttons

    # Use proportional defaults
    buttons = {}
    for name, info in DEFAULT_BUTTONS.items():
        buttons[name] = (int(width * info["x_pct"]), int(height * info["y_pct"]))
    print("Using default button positions (run 'python3 calibrate.py' to calibrate)")
    return buttons


def save_buttons(buttons):
    """Save calibrated button positions."""
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(buttons, f, indent=2)
    print(f"Saved to {CALIBRATION_FILE}")


def calibrate():
    device = detect_device()
    width, height = device.get_screen_resolution()
    print(f"Device: {width}x{height}")

    # Take a screenshot for reference
    path = device.capture_screenshot("calibrate")
    print(f"Screenshot saved: {path}")
    print(f"Open this image to identify button positions.\n")

    buttons = {}
    for name, info in DEFAULT_BUTTONS.items():
        default_x = int(width * info["x_pct"])
        default_y = int(height * info["y_pct"])

        print(f"\n--- {info['description']} ({name}) ---")
        print(f"  Default: ({default_x}, {default_y})")

        user_input = input(f"  Enter 'x,y' or press Enter for default: ").strip()
        if user_input:
            parts = user_input.split(",")
            x, y = int(parts[0].strip()), int(parts[1].strip())
        else:
            x, y = default_x, default_y

        buttons[name] = {"x": x, "y": y}
        print(f"  Set to: ({x}, {y})")

        # Test tap
        test = input("  Tap to test? (y/N): ").strip().lower()
        if test == "y":
            device.tap(x, y)
            print("  Tapped!")

    save_buttons(buttons)
    print("\nCalibration complete! Run 'python3 main.py' to use these positions.")


if __name__ == "__main__":
    calibrate()