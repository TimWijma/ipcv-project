import mediapipe as mp

class FaceLandmarkerHandler:
    """Handles face landmark detection using MediaPipe in live mode"""

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode


    def __init__(self, model_path='face_landmarker.task'):
        self.latest_result = None
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        """Callback to handle results from the landmarker"""
        self.latest_result = result

    def detect(self, frame_rgb, timestamp_ms):
        """Process a single frame for face landmarks"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def get_landmarks(self):
        """Return list of face landmarks or None if no face detected"""
        if self.latest_result:
            return self.latest_result.face_landmarks
        return None

    def close(self):
        """Clean up mediapipe"""
        self.landmarker.close()

import mediapipe as mp

class HandLandmarkerHandler:
    """Handles hand landmark detection using MediaPipe in live mode."""

    def __init__(self, model_path='hand_landmarker.task'):
        self.latest_result = None
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            num_hands=2
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def detect(self, frame_rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def get_landmarks(self):
        if self.latest_result:
            return self.latest_result.hand_landmarks
        return None

    def close(self):
        self.landmarker.close()

class FilterManager:
    """Simple manager for applying filters to"""
    def __init__(self):
        self.filters = []

    def add_filter(self, func):
        self.filters.append(func)

    def apply(self, frame_bgr, landmarks_list):
        if not landmarks_list:
            return frame_bgr

        output = frame_bgr.copy()
        for f in self.filters:
            output = f(output, landmarks_list)
        return output
