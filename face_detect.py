import cv2
import numpy as np

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Optional: Load pre-trained emotion classifier (for emotion detection)
emotion_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to detect faces in an image
def detect_faces(image):
    """
    Detect faces in an image and draw rectangles around them
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green rectangle

    return faces

# Optional: Function for emotion detection (like detecting smiles)
def detect_emotions(image, faces):
    """
    Detect emotions (e.g., smile) within the detected faces.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for (x, y, w, h) in faces:
        # Region of Interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        smiles = emotion_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(image, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (255, 0, 0), 2)  # Blue rectangle for smiles

# Function to handle real-time video face detection from webcam
def real_time_face_detection():
    """
    Detect faces in real-time using webcam video stream.
    """
    # Start webcam capture
    cap = cv2.VideoCapture(0)  # 0 is default webcam ID
    
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        
        # Detect faces in the frame
        faces = detect_faces(frame)
        
        # Optionally, detect emotions (e.g., smiles) in faces
        detect_emotions(frame, faces)
        
        # Display the resulting frame
        cv2.imshow('Face and Emotion Detection', frame)
        
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any windows
    cap.release()
    cv2.destroyAllWindows()

# Function to detect faces in an image file (for static images)
def detect_faces_in_image(image_path):
    """
    Detect faces in a static image file.
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Detect faces in the image
    faces = detect_faces(image)
    
    # Optionally, detect emotions (e.g., smiles) in faces
    detect_emotions(image, faces)
    
    # Display the image with detected faces and emotions
    cv2.imshow('Detected Faces and Emotions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Choose the input method: webcam for real-time or image file for static images
    choice = input("Choose the mode:\n1. Real-time face detection (webcam)\n2. Detect faces in an image file\nEnter choice (1 or 2): ")

    if choice == '1':
        real_time_face_detection()  # Real-time face detection from webcam
    elif choice == '2':
        image_path = input("Enter the image path: ")
        detect_faces_in_image(image_path)  # Face detection in a static image
    else:
        print("Invalid choice! Please enter 1 or 2.")
