import cv2
import sys

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

        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv2.imshow('Live Camera', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()