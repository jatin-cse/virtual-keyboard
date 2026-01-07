import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

keys = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L","ENTER"],
    ["Z","X","C","V","B","N","M","BACK","SHIFT"],
    ["SPACE"]
]

final_text = ""
last_press_time = 0
press_delay = 0.7
shift_on = False

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # 🔹 AUTO SCALE
    max_keys = 10
    key_w = w // (max_keys + 2)
    key_h = key_w
    start_x = (w - key_w * max_keys) // 2
    start_y = int(h * 0.45)

    def draw_keyboard():
        for i, row in enumerate(keys):
            for j, key in enumerate(row):
                if key == "SPACE":
                    x = start_x
                    y = start_y + i * key_h
                    width = key_w * 5
                else:
                    x = start_x + j * key_w
                    y = start_y + i * key_h
                    width = key_w

                color = (255, 0, 255)
                if key == "SHIFT" and shift_on:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + width, y + key_h), color, 2)
                cv2.putText(frame, key, (x + 8, y + int(key_h * 0.7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

    def get_key(ix, iy):
        for i, row in enumerate(keys):
            for j, key in enumerate(row):
                if key == "SPACE":
                    x = start_x
                    y = start_y + i * key_h
                    width = key_w * 5
                else:
                    x = start_x + j * key_w
                    y = start_y + i * key_h
                    width = key_w

                if x < ix < x + width and y < iy < y + key_h:
                    return key
        return None

    draw_keyboard()

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)

            key = get_key(ix, iy)
            d = distance(ix, iy, tx, ty)
            current_time = time.time()

            if key and d < 25 and current_time - last_press_time > press_delay:
                if key == "SPACE":
                    final_text += " "
                elif key == "BACK":
                    final_text = final_text[:-1]
                elif key == "ENTER":
                    final_text += "\n"
                elif key == "SHIFT":
                    shift_on = not shift_on
                else:
                    final_text += key.upper() if shift_on else key.lower()
                    shift_on = False

                last_press_time = current_time

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # 🔹 TEXT BOX (SAFE)
    cv2.rectangle(frame, (20, 20), (w - 20, 110), (0, 255, 255), 2)
    y_text = 60
    for line in final_text.split("\n")[-2:]:
        cv2.putText(frame, line, (30, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        y_text += 30

    cv2.putText(frame, f"SHIFT: {'ON' if shift_on else 'OFF'}",
                (w - 160, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if shift_on else (0, 0, 255), 2)

    cv2.imshow("Virtual Keyboard - Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
