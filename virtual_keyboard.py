import cv2
import mediapipe as mp
import time
import math

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Keyboard layout
keys = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M","SPACE"]
]

final_text = ""
last_press_time = 0
press_delay = 0.8

def draw_keyboard(img):
    start_x, start_y = 50, 200
    key_w, key_h = 60, 60

    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = start_x + j * key_w
            y = start_y + i * key_h

            cv2.rectangle(img, (x, y), (x+key_w, y+key_h),
                          (255, 0, 255), 2)
            cv2.putText(img, key, (x+10, y+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

def get_key(ix, iy):
    start_x, start_y = 50, 200
    key_w, key_h = 60, 60

    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = start_x + j * key_w
            y = start_y + i * key_h
            if x < ix < x+key_w and y < iy < y+key_h:
                return key
    return None

def distance(x1, y1, x2, y2):
    return math.hypot(x2-x1, y2-y1)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_keyboard(frame)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)

            key = get_key(ix, iy)
            d = distance(ix, iy, tx, ty)
            current_time = time.time()

            if key and d < 30 and current_time - last_press_time > press_delay:
                if key == "SPACE":
                    final_text += " "
                else:
                    final_text += key
                last_press_time = current_time

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Text output box
    cv2.rectangle(frame, (50, 120), (700, 170), (0, 255, 255), 2)
    cv2.putText(frame, final_text, (60, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    cv2.imshow("Virtual Keyboard - Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
