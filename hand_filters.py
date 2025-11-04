import cv2
import math

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
        
        if distance < 0.2:
            center_y = int((index_finger_tip.y + thumb_tip.y) / 2 * h)
            center_x = int((index_finger_tip.x + thumb_tip.x) / 2 * w)

            is_left = center_x < w / 2

            slider_value = int((1 - center_y / h) * 100)

            if is_left:
                state['slider_left_value'] = slider_value
            else:
                state['slider_right_value'] = slider_value


    return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0), state

def draw_slider_value(frame, state, disable_right=False, disable_left=False):
    slider_right_value = state.get('slider_right_value', None)
    slider_left_value = state.get('slider_left_value', 10)

    h, w, _ = frame.shape
    bar_right_x = w - 50
    bar_left_x = 50
    bar_y1 = int(h * 0.1)
    bar_y2 = int(h * 0.9)

    if not disable_right:
        cv2.rectangle(frame, (bar_right_x - 10, bar_y1), (bar_right_x + 10, bar_y2), (200, 200, 200), 2)

    if not disable_left:
        cv2.rectangle(frame, (bar_left_x - 10, bar_y1), (bar_left_x + 10, bar_y2), (200, 200, 200), 2)

    if slider_right_value is not None and not disable_right:
        fill_y = int(bar_y2 - (slider_right_value / 100) * (bar_y2 - bar_y1))
        cv2.rectangle(frame, (bar_right_x - 10, fill_y), (bar_right_x + 10, bar_y2), (0, 255, 0), -1)

    if slider_left_value is not None and not disable_left:
        fill_y = int(bar_y2 - (slider_left_value / 100) * (bar_y2 - bar_y1))
        cv2.rectangle(frame, (bar_left_x - 10, fill_y), (bar_left_x + 10, bar_y2), (255, 0, 0), -1)

    return frame, state
