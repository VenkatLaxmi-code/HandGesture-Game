import cv2
import time
import numpy as np
import pyautogui
import mediapipe as mp
from collections import deque

# -------------------------
# Config (tweak to taste)
# -------------------------
CAM_INDEX = 0                 # your webcam index
MAX_HANDS = 1                 # track one hand
DETECTION_CONF = 0.6
TRACKING_CONF = 0.6
SWIPE_VEL_THRESH = 0.015      # normalized units per second (bigger = harder to trigger)
SWIPE_WINDOW = 6              # frames for velocity smoothing
ACTION_COOLDOWN = 0.6         # seconds between triggers
SHOW_HUD = True

# Open palm & fist thresholds (simple finger-up heuristic)
# Finger considered "up" if fingertip y is above PIP y (for vertical upright hand)
# Landmarks: https://google.github.io/mediapipe/solutions/hands#hand-landmark-model
FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
FINGER_PIPS = [3, 6, 10, 14, 18]  # thumb ip, index pip, ...

# Key mapping for actions
KEY_LEFT  = "left"
KEY_RIGHT = "right"
KEY_UP    = "up"
KEY_DOWN  = "down"

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # no added delay

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def count_fingers_up(landmarks, handedness_label):
    """
    Simple heuristic to count how many fingers are up.
    Uses y for index..pinky; uses x for thumb depending on handedness.
    """
    # landmarks are normalized [0..1] in image coords
    tips = FINGER_TIPS
    pips = FINGER_PIPS

    # y smaller = higher
    up = 0

    # Thumb logic: compare x depending on left/right hand
    # For right hand, thumb is to the left (smaller x) when extended; reverse for left hand.
    # Using tip (4) vs ip (3)
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    if handedness_label == "Right":
        if thumb_tip.x < thumb_ip.x:
            up += 1
    else:  # Left
        if thumb_tip.x > thumb_ip.x:
            up += 1

    # Other 4 fingers: tip y is above pip y when extended
    for tip_idx, pip_idx in zip(tips[1:], pips[1:]):
        if landmarks[tip_idx].y < landmarks[pip_idx].y:
            up += 1

    return up

class SwipeDetector:
    def __init__(self, window=SWIPE_WINDOW):
        self.window = window
        self.x_hist = deque(maxlen=window)
        self.t_hist = deque(maxlen=window)

    def update(self, x_norm):
        t = time.time()
        self.x_hist.append(x_norm)
        self.t_hist.append(t)

    def velocity(self):
        if len(self.x_hist) < 2:
            return 0.0
        dx = self.x_hist[-1] - self.x_hist[0]
        dt = self.t_hist[-1] - self.t_hist[0]
        return 0.0 if dt <= 0 else dx / dt  # normalized units per second

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    swipe = SwipeDetector()
    last_action_ts = 0
    last_action = ""

    with mp_hands.Hands(
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural control
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            action_text = ""
            now = time.time()

            if results.multi_hand_landmarks and results.multi_handedness:
                # take first hand
                hand_lms = results.multi_hand_landmarks[0]
                hand_label = results.multi_handedness[0].classification[0].label  # "Left"/"Right"

                # draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                # swipe detection based on index fingertip x velocity
                index_tip = hand_lms.landmark[8]
                swipe.update(index_tip.x)  # normalized

                vel = swipe.velocity()  # units per sec, positive = moving right
                triggered = False

                # Finger count for open palm / fist
                fingers_up = count_fingers_up(hand_lms.landmark, hand_label)

                def ready():
                    return (now - last_action_ts) >= ACTION_COOLDOWN

                if ready():
                    if fingers_up >= 5:
                        pyautogui.press(KEY_UP)
                        last_action_ts = now
                        last_action = "JUMP (Open palm)"
                        action_text = last_action
                        triggered = True
                    elif fingers_up <= 1:
                        pyautogui.press(KEY_DOWN)
                        last_action_ts = now
                        last_action = "ROLL (Fist)"
                        action_text = last_action
                        triggered = True

                # Swipes take priority only if not just triggered by finger count
                if not triggered and ready():
                    if vel >= SWIPE_VEL_THRESH:
                        pyautogui.press(KEY_RIGHT)
                        last_action_ts = now
                        last_action = f"RIGHT (Swipe →, vel={vel:.2f})"
                        action_text = last_action
                    elif vel <= -SWIPE_VEL_THRESH:
                        pyautogui.press(KEY_LEFT)
                        last_action_ts = now
                        last_action = f"LEFT (Swipe ←, vel={vel:.2f})"
                        action_text = last_action

                if SHOW_HUD:
                    cx = int(index_tip.x * w)
                    cy = int(index_tip.y * h)
                    cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)

            # HUD / status text
            if SHOW_HUD:
                hud_lines = [
                    "Gesture Surfer",
                    f"Cooldown: {ACTION_COOLDOWN:.1f}s  VelThresh: {SWIPE_VEL_THRESH:.3f}",
                    f"Last: {last_action}",
                    "Gestures: OpenPalm=Jump ↑ | Fist=Roll ↓ | Swipe→=Right | Swipe←=Left",
                    "Press 'q' to quit",
                ]
                y0 = 28
                for i, line in enumerate(hud_lines):
                    y = y0 + i * 24
                    cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

                if action_text:
                    cv2.putText(frame, action_text, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("Gesture Surfer — Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()