import cv2
import numpy as np




def draw_snapchat_filters(frame, landmarks_list,hat,sun,state):

    slider_right_value = state.get('slider_right_value', 50)
    slider_left_value = state.get('slider_left_value', 50)

        # Convert to scale range 0.5–2.0
    scale_right = 0.05 + (slider_right_value / 100) * (0.3 - 0.05)  # 0.05 → 0.3
    scale_left  = 0.1  + (slider_left_value  / 100) * (0.4 - 0.1)   # 0.1 → 0.4
            
    hat_resized = cv2.resize(hat, (0, 0), fx=scale_right, fy=scale_right, interpolation=cv2.INTER_AREA)
    sun_resized = cv2.resize(sun, (0, 0), fx=scale_left, fy=scale_left, interpolation=cv2.INTER_AREA)

    which_filter = [hat_resized, sun_resized]

    output = frame.copy()
    h, w, _ = frame.shape
    

    for face_landmarks in landmarks_list:


        hat_point = face_landmarks[10]
        sunglasses_pont = face_landmarks[197]

        for i, pos in enumerate([hat_point, sunglasses_pont]):
            current_filter = which_filter[i]
            fh, fw = current_filter.shape[:2]

            if  i == 0:
                cx, cy = int(pos.x * w), int(pos.y * h)
                x1, y1 = int(cx - fw / 2), int(cy - fh / 2)-50
                x2, y2 = x1 + fw, y1 + fh
            elif i == 1:
                cx, cy = int(pos.x * w), int(pos.y * h)
                x1, y1 = int(cx - fw / 2), int(cy - fh / 2)
                x2, y2 = x1 + fw, y1 + fh


            # Skip if out of bounds
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                print('out of bound')
                continue

            roi = output[y1:y2, x1:x2]
            filter_rgb = current_filter[:, :, :3]
            alpha = current_filter[:, :, 3] / 255.0

            # Alpha blend
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * filter_rgb[:, :, c]

            output[y1:y2, x1:x2] = roi
    print('done')
    print(f"Overlaying {fw}x{fh} at {(x1, y1)}")

    return output,state
