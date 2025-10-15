import cv2
import math
from common import add_text

def slider_effect(frame, hand_landmarks_list, state):
    if not hand_landmarks_list:
        return frame

    h, w, _ = frame.shape
    overlay = frame.copy()

    for hand_landmarks in hand_landmarks_list:
        index_finger_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]
        distance = math.dist((index_finger_tip.x, index_finger_tip.y), (thumb_tip.x, thumb_tip.y))
        print(f"Distance between index finger tip and thumb tip: {distance}")
        
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        cv2.circle(overlay, (cx, cy), 50, (255, 0, 0), -1)

        cx, cy = int(thumb_tip.x * w), int(thumb_tip.y * h)
        cv2.circle(overlay, (cx, cy), 50, (255, 0, 0), -1)

        line_color = (0, 255, 0) if distance < 0.2 else (0, 0, 255)

        cv2.line(
            overlay, 
            (int(index_finger_tip.x * w), int(index_finger_tip.y * h)),
            (int(thumb_tip.x * w), int(thumb_tip.y * h)), 
            line_color, 
            5
        )

        if distance < 0.2:
            center_y = int((index_finger_tip.y + thumb_tip.y) / 2 * h)
            center_x = int((index_finger_tip.x + thumb_tip.x) / 2 * w)
            cv2.line(overlay, (0, center_y), (w, center_y), (0, 255, 0), 3)

            is_left = center_x < w / 2

            slider_value = int((1 - center_y / h) * 100)

            if is_left:
                state['slider_left_value'] = slider_value
            else:
                state['slider_right_value'] = slider_value


    frame = add_text(frame, f"Distance: {distance:.2f}")

    return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0), state