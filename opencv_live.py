import cv2
import sys
from common import add_text
from face_distorsion import face_swirl_filter
from facelandmark import FaceLandmarkerHandler, FilterManager, HandLandmarkerHandler
from filters import blush_filter, draw_hand_landmarks_filter, draw_landmarks_filter
from hand_filters import slider_effect, draw_slider_value


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open camera.")

    face_handler = FaceLandmarkerHandler()
    hand_handler = HandLandmarkerHandler()
    
    state = {}
    face_filters = FilterManager(state)
    hand_filters = FilterManager(state)

    ### Add filters for faces here
    # face_filters.add_filter(blush_filter)
    # face_filters.add_filter(draw_landmarks_filter)
    face_filters.add_filter(face_swirl_filter)

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
            # final_frame = hand_filtered

            print(f"Face state: {face_state}")
            print(f"Hand state: {hand_state}")

            cv2.imshow('Live Camera', final_frame)

    finally:
        face_handler.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()