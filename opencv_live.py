import cv2
import sys
from common import add_text
from face_distorsion import face_swirl_filter
from facelandmark import FaceLandmarkerHandler, FilterManager, HandLandmarkerHandler
from filters import blush_filter, draw_hand_landmarks_filter, draw_landmarks_filter
from hand_filters import slider_effect, draw_slider_value
from snapchat_filters import draw_snapchat_filters


def switch_filter(current_index, filters):
    current_index = (current_index + 1) % len(filters)

    if filters[current_index] == "swirl":
        face_filters = FilterManager({})
        face_filters.add_filter(face_swirl_filter)
    elif filters[current_index] == "hat":
        image_1 = cv2.imread('hat_transparant.png', cv2.IMREAD_UNCHANGED)
        image_2 = cv2.imread('sunglasses_transparant.png', cv2.IMREAD_UNCHANGED)

        if image_1 is None or image_2 is None:
            sys.exit("Error: could not load hat or sunglasses images.")


        face_filters = FilterManager({})
        face_filters.add_filter(
            lambda frame, landmarks, state: draw_snapchat_filters(frame, landmarks, image_1, image_2, state)
        )
    else:
        face_filters = FilterManager({})

    return current_index, face_filters

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open camera.")

    face_handler = FaceLandmarkerHandler()
    hand_handler = HandLandmarkerHandler()
    
    state = {
        'slider_left_value': 50,
        'slider_right_value': 50
    }
    hand_filters = FilterManager(state)

    filters = ["swirl", "hat", "none"]
    current_filter_index = 0

    hand_filters.add_filter(slider_effect)

    current_filter_index, face_filters = switch_filter(current_filter_index, filters)

    try:
        while True:
            ret, frame_bgr = cap.read()

            ### flipping frame for mirror effect
            ### disable for original camera view
            frame_bgr = cv2.flip(frame_bgr, 1)

            if not ret:
                print("Failed to capture frame")
                break

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break

            if key == ord('f'):
                current_filter_index, face_filters = switch_filter(current_filter_index, filters)
                print(f"Switched to filter: {filters[current_filter_index]}")


            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            face_handler.detect(frame_rgb, timestamp_ms)
            hand_handler.detect(frame_rgb, timestamp_ms)

            face_landmarks = face_handler.get_landmarks()
            hand_landmarks = hand_handler.get_landmarks()

            hand_filtered, hand_state = hand_filters.apply(frame_bgr, hand_landmarks)

            face_filters.state.update(hand_state)

            face_filtered, face_state = face_filters.apply(hand_filtered, face_landmarks)

            disable_right = filters[current_filter_index] == "swirl" or filters[current_filter_index] == "none"
            disable_left = filters[current_filter_index] == "none"

            final_frame, _ = draw_slider_value(face_filtered, face_state, disable_right=disable_right, disable_left=disable_left)
            #final_frame = hand_filtered

            cv2.imshow('Live Camera', final_frame)

    finally:
        face_handler.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()