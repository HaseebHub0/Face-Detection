# Face and Emotion Detection Program

This program allows you to detect faces and emotions (such as smiles) in real-time using a webcam or in a static image. It utilizes OpenCV and pre-trained Haar Cascade classifiers for detecting faces and emotions.

## Features:
- **Real-time Face Detection**: Uses the webcam to detect faces in real-time and displays them with rectangles around them.
- **Emotion Detection**: Detects smiles (or other emotions based on classifiers) within the detected faces.
- **Static Image Face Detection**: Allows you to upload an image and detect faces in it.

## Requirements:
- Python 3.x
- OpenCV (`cv2` library)

## Installation:
1. Install Python 3.x from [here](https://www.python.org/downloads/).
2. Install OpenCV by running:
    ```bash
    pip install opencv-python
    ```

## Usage:

### 1. Real-time Face Detection (Webcam):
   - Choose the first option (`1`) when prompted.
   - The program will start using your webcam and detect faces in real-time.
   - It will also detect smiles (emotions) within the faces.
   - Press `q` to exit the webcam feed.

### 2. Detect Faces in a Static Image:
   - Choose the second option (`2`) when prompted.
   - Enter the path of an image file on your system.
   - The program will display the image with detected faces and emotions marked.

## Code Explanation:
- **`detect_faces(image)`**: Detects faces in an image and draws green rectangles around them.
- **`detect_emotions(image, faces)`**: Detects smiles within the detected faces and draws blue rectangles around them.
- **`real_time_face_detection()`**: Starts the webcam and detects faces and emotions in real-time.
- **`detect_faces_in_image(image_path)`**: Detects faces and emotions in a static image file.

## Example:

For real-time detection:
```bash
Choose the mode:
1. Real-time face detection (webcam)
2. Detect faces in an image file
Enter choice (1 or 2): 1
