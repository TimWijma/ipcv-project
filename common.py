from dataclasses import dataclass

import cv2
import numpy as np

def add_text(frame, text: str) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    h, w = frame.shape[:2]

    x = (w - text_width) // 2
    y = h - 20

    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame

def create_frame(image: np.ndarray, text: str, func = None, *args) -> np.ndarray:
    if func:
        processed_image = func(image, *args)
    else:
        processed_image = image
    return add_text(processed_image, text)

def draw_detection(frame, x, y, w, h, name):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

def detect_face(image, face_classifier):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)
    )
    return faces

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((62, 2), dtype=dtype)

    for i in range(62):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords
