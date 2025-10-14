import cv2
import sys
from facelandmark import FaceLandmarkerHandler, FilterManager, HandLandmarkerHandler
from filters import blush_filter, draw_hand_landmarks_filter, draw_landmarks_filter

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open camera.")

    face_handler = FaceLandmarkerHandler()
    hand_handler = HandLandmarkerHandler()
    face_filters = FilterManager()
    hand_filters = FilterManager()

    ### Add filters for faces here
    face_filters.add_filter(draw_landmarks_filter)
    face_filters.add_filter(blush_filter)

    ### Add filters for hands here
    hand_filters.add_filter(draw_hand_landmarks_filter)

    try:
        while True:
            ret, frame_bgr = cap.read()
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

            face_filtered = face_filters.apply(frame_bgr, face_landmarks)
            hand_filtered = hand_filters.apply(face_filtered, hand_landmarks)

            cv2.imshow('Live Camera', hand_filtered)

    finally:
        face_handler.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()