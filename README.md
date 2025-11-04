# ipcv-project

## How to run
1. Create a virtual environment with Python and activate it.
2. Install the required packages with `pip install -r requirements.txt`
3. Run the main file with `python opencv_live.py`
4. Press "F" to toggle between filters and "Q" to quit the application.
5. Use a pinch gesture with your thumb and index finger and move it up and down to adjust the filter parameters.


## File Descriptions
- `opencv_live.py`: Main application file that captures video from the webcam and applies filters.
- `filters.py`: Contains filters for hand detection and snapchat-like effects.
- `face_distortion.py`: Contains filters for face distortion effects.
- `common.py`: Contains common functions used across different modules.
- `requirements.txt`: Lists the required Python packages for the project.
- `facelandmark.py`: Includes classes for setting up face and hand landmark detection using MediaPipe.
- `face_landmarker.task`: Pre-trained model file for face landmark detection.
- `hand_landmarker.task`: Pre-trained model file for hand landmark detection.

## Dependencies
- OpenCV
- MediaPipe
- NumPy