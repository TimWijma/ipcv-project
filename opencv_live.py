import cv2
import sys

from common import detect_face, draw_detection

def main() -> None:
    cap = cv2.VideoCapture(0)  # 0 = default camera

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if not cap.isOpened():
        sys.exit("Could not open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face = detect_face(frame, face_classifier)
        for (x, y, w, h) in face:
            draw_detection(frame, x, y, w, h, "Face")

        
        cv2.imshow('Live Camera', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()