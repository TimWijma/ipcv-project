import cv2
import sys
from common import add_text
from facelandmark import FaceLandmarkerHandler, FilterManager, HandLandmarkerHandler
from filters import blush_filter, draw_hand_landmarks_filter, draw_landmarks_filter
from hand_filters import slider_effect


def draw_slider_value(frame, state):
    print(f"Received state: {state}")

    slider_right_value = state.get('slider_right_value', None)
    slider_left_value = state.get('slider_left_value', 10)

    h, w, _ = frame.shape
    bar_right_x = w - 50
    bar_left_x = 50
    bar_y1 = int(h * 0.1)
    bar_y2 = int(h * 0.9)
    cv2.rectangle(frame, (bar_right_x - 10, bar_y1), (bar_right_x + 10, bar_y2), (200, 200, 200), 2)
    cv2.rectangle(frame, (bar_left_x - 10, bar_y1), (bar_left_x + 10, bar_y2), (200, 200, 200), 2)

    if slider_right_value is not None:
        fill_y = int(bar_y2 - (slider_right_value / 100) * (bar_y2 - bar_y1))
        cv2.rectangle(frame, (bar_right_x - 10, fill_y), (bar_right_x + 10, bar_y2), (0, 255, 0), -1)

    if slider_left_value is not None:
        fill_y = int(bar_y2 - (slider_left_value / 100) * (bar_y2 - bar_y1))
        cv2.rectangle(frame, (bar_left_x - 10, fill_y), (bar_left_x + 10, bar_y2), (255, 0, 0), -1)

    return frame, state

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open camera.")

    face_handler = FaceLandmarkerHandler()
    hand_handler = HandLandmarkerHandler()
    face_filters = FilterManager()
    hand_filters = FilterManager()

    ### Add filters for faces here
    # face_filters.add_filter(draw_landmarks_filter)
    # face_filters.add_filter(blush_filter)

    ### Add filters for hands here
    hand_filters.add_filter(draw_hand_landmarks_filter)

    hand_filters.add_filter(slider_effect)

    try:
        while True:
            ret, frame_bgr = cap.read()

            ### flipping frame for mirror effect
            ### disable for original camera view
            frame_bgr = cv2.flip(frame_bgr, 1)

            if not ret:
                print("Failed to capture frame")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            face_handler.detect(frame_rgb, timestamp_ms)
            hand_handler.detect(frame_rgb, timestamp_ms)

            face_landmarks = face_handler.get_landmarks()
            hand_landmarks = hand_handler.get_landmarks()

            face_filtered, face_state = face_filters.apply(frame_bgr, face_landmarks)
            hand_filtered, hand_state = hand_filters.apply(face_filtered, hand_landmarks)

            final_frame, _ = draw_slider_value(hand_filtered, hand_state)

            print(f"Hand state: {hand_state}")

            cv2.imshow('Live Camera', final_frame)

    finally:
        face_handler.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()