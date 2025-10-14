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

def create_template(template_path: str) -> tuple[np.ndarray, int, int]:
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    
    return template, w, h

def detect_template(img_gray, template, threshold=0.7):
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    points = list(zip(*loc[::-1]))
    return points

def draw_detection(frame, point, w, h, name):
    cv2.rectangle(frame, point, (point[0] + w, point[1] + h), (0,0,255), 2)
    cv2.putText(frame, name, (point[0], point[1]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def get_center(point, w, h):
    center_x = point[0] + int(round(w/2))
    center_y = point[1] + int(round(h/2))
    return center_x, center_y

def check_bounds_and_mask(center_x, center_y, mask):
    return (0 <= center_y < mask.shape[0] and 
            0 <= center_x < mask.shape[1] and 
            mask[center_y, center_x] != 255)

def overlay_image_on_frame(frame, overlay_img, x, y):
    if len(overlay_img.shape) == 2:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)
    
    h_overlay, w_overlay = overlay_img.shape[:2]
    
    frame[y:y+h_overlay, x:x+w_overlay] = overlay_img
    
    return frame