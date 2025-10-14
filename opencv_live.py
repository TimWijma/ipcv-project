import cv2
import sys

def main() -> None:
    cap = cv2.VideoCapture(0)  # 0 = default camera

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

        cv2.imshow('Live Camera', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()