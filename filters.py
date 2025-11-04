import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

def draw_landmarks_filter(frame_bgr, landmarks_list, state=None):
    if not landmarks_list:
        return frame_bgr

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    annotated = np.copy(frame_rgb)

    for face_landmarks in landmarks_list:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

def draw_hand_landmarks_filter(frame_bgr, hand_landmarks_list, state=None):
    if not hand_landmarks_list:
        return frame_bgr

    h, w, _ = frame_bgr.shape
    annotated = frame_bgr.copy()

    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            cv2.line(
                annotated,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (0, 255, 255), 2
            )

    return annotated

def draw_snapchat_filters(frame, landmarks_list,slider, slider_resize = False,scale = 0.1,*which_filter):

    filter_number = 0

    filters, factor = load_filters(*which_filter)
    resized =  resize_filters(filters,slider_resize, slider, factor)

    filter_dict = {'hat': 10,
                   'sunglasses':197}
    
    overlay = frame.copy()
    h, w, _ = frame.shape    

    for face_landmarks in landmarks_list:
        print(f"Applying blush to face landmarks: {face_landmarks}")
        filterlandmark_positions = []

        for i,filter_name in enumerate(which_filter):
            filter_image = face_landmarks[filter_dict[filter_name]]
            filterlandmark_positions.append(filter_image)

        for pos in filterlandmark_positions:
            cx, cy = int(pos.x * w), int(pos.y * h)

            snap = resized[filter_number]
            h_hat ,w_hat = snap.shape[:2]
          
             #location
            x_hat = int(cx - (w_hat/2))
            y_hat = int(cy - w_hat)
            #hat filter
            roi_hat = frame[y_hat:y_hat+h_hat,x_hat:x_hat+w_hat]
            hat_rgb = snap[:,:,:3]
            alpha = snap[:,:,3]
            mask = alpha != 0
            roi_hat[mask] = hat_rgb[mask]
            frame[y_hat:y_hat+h_hat,x_hat:x_hat+w_hat] = roi_hat

            filter_number+=1
    return frame


def load_filters(*filters):

    filters_used = []
    factor = []

    filter_dict = {'hat': 'hat_transparant.PNG',
                   'sunglasses':'sunglasses_transparant.PNG'}
    filter_size = {'hat': 0.1,
                   'sunglasses':0.2}
    
    for name in filters:
        if name in filter_dict:
            image = cv2.imread(filter_dict[name], cv2.IMREAD_UNCHANGED)
            filters_used.append(image)
            factor.append(filter_size[name])
        else:
            print(f"Warning: '{name}' not found in filter_dict.")

    return filters_used,factor

def resize_filters(filters_used,slider_resize, slider, factor):
    resized_filters =[]

    for i,f in enumerate(filters_used):

        if slider_resize == True:
            resized = cv2.resize(f, (0, 0), fx=(slider), fy=(slider), interpolation=cv2.INTER_AREA)
            resized_filters.append(resized)
        elif slider_resize == False:
            resized = cv2.resize(f, (0, 0), fx=(factor[i]), fy=(factor[i]), interpolation=cv2.INTER_AREA)
            resized_filters.append(resized)


    return resized_filters
