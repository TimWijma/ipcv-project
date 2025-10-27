import numpy as np

def face_swirl_filter(frame, landmarks_list, state):
    h, w = frame.shape[:2]
    output = frame.copy()
    
    # Process each detected face
    for face_landmarks in landmarks_list:
        face_center = face_landmarks[4]
        face_center_x = int(face_center.x * w)
        face_center_y = int(face_center.y * h)

        # Calculate face radius based on face width
        left_face = face_landmarks[234]
        right_face = face_landmarks[454]
        face_width = abs(right_face.x - left_face.x) * w
        #radius of swirl (smaller than face width)
        rad = (state.get('slider_right_value', 60)) * 0.01
        radius = int(face_width*rad)
        
        # Swirl parameters (we can use it to dynamically adjust swirl)
        strength = (state.get('slider_left_value', 50) - 50 ) * 0.1
        
        # Create swirl effect
        for y in range(max(0, face_center_y - radius), min(h, face_center_y + radius)):
            for x in range(max(0, face_center_x - radius), min(w, face_center_x + radius)):
                # Calculate distance from center

                dist_x = float(x - face_center_x)
                dist_y = float(y - face_center_y)
                dist = np.sqrt(dist_x**2 + dist_y**2)
                
                if dist < radius and dist > 0:
                    swirl_amount = (radius - dist) / radius
                    angle = strength * swirl_amount
                    
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    
                    #operate with 2D rotation matrix
                    new_x = int(dist_x * cos_a - dist_y * sin_a + face_center_x)
                    new_y = int(dist_x * sin_a + dist_y * cos_a + face_center_y)

                    # Check bounds and copy pixel
                    if 0 <= new_x < w and 0 <= new_y < h:
                        output[y, x] = frame[new_y, new_x]

    return output