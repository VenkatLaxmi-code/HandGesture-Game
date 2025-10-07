# Gesture Surfer — Control Subway Surfers with Hand Gestures

This project lets you play Subway Surfers (web or desktop) using simple hand gestures captured from your webcam.
It uses **OpenCV** (video), **MediaPipe Hands** (gesture landmarks), and **PyAutoGUI** (to press keys).

## Quick Start

1) Create and activate a virtual environment (recommended):

**Windows (PowerShell):**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate
```

**macOS/Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Launch the script:
```bash
python gesture_surfer.py
```

4) Focus the Subway Surfers game window (browser tab or app). Keep the webcam in view.

## Default Gestures → Game Actions

- **Swipe Right** (fast index fingertip move to the right) → Press **Right Arrow** (change lane right)
- **Swipe Left** (fast index fingertip move to the left) → Press **Left Arrow** (change lane left)
- **Open Palm** (all five fingers up) → Press **Up Arrow** (jump)
- **Fist** (no fingers up) → Press **Down Arrow** (roll/slide)

### Tips
- Gestures trigger once and then enter a short **cooldown** to avoid repeats.
- Make gestures clearly within the camera view and roughly centered.
- You can adjust thresholds near the top of `gesture_surfer.py` for sensitivity and cooldown.

## Troubleshooting

- If nothing happens, ensure the game window is focused (click it once).
- If your cursor moves unexpectedly, PyAutoGUI might be failing due to permissions (macOS Accessibility permission).
  - On macOS, add your terminal/IDE to **System Settings → Privacy & Security → Accessibility**.
- If the webcam is wrong, change `CAM_INDEX` in `gesture_surfer.py` (0, 1, 2…).
- If MediaPipe fails to install on Apple Silicon, try:
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install mediapipe
  ```

## Safety Note
PyAutoGUI sends real key presses to the **active** application. Always keep control of the environment when running the script.