import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

### EXAMPLE FILTERS ###
def blush_filter(frame, landmarks_list, state=None):
    overlay = frame.copy()
    h, w, _ = frame.shape
    for face_landmarks in landmarks_list:
        print(f"Applying blush to face landmarks: {face_landmarks}")

        left_cheek = face_landmarks[234]
        right_cheek = face_landmarks[454]
        for cheek in [left_cheek, right_cheek]:
            cx, cy = int(cheek.x * w), int(cheek.y * h)
            cv2.circle(overlay, (cx, cy), 25, (147, 20, 255), -1)
    return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

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