import cv2
import sys
from imutils import face_utils
import argparse
import dlib

from common import detect_face, draw_detection

def main() -> None:
    cap = cv2.VideoCapture(0)  # 0 = default camera

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--shape_predictor", required=True,
                        help="path to shape_predictor_68_face_landmarks.dat")
    args = vars(parser.parse_args())

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    if not cap.isOpened():
        sys.exit("Could not open camera.")

    while True:
        ret, image = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray_image, 1)

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray_image, rect)
            shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # face = detect_face(frame, face_classifier)
        # for (x, y, w, h) in face:
        #     draw_detection(frame, x, y, w, h, "Face")

        
        cv2.imshow('Live Camera', image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()