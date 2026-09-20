import os
import cv2
import mediapipe as mp
from features import extract_features


# Directory configuration
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)


# MediaPipe configuration
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Create hand landmarker
with HandLandmarker.create_from_options(options) as landmarker:
    # Open webcam
    camera = cv2.VideoCapture(0)
    while True:
        # Capture frame
        success, frame = camera.read()
        if not success:
            print("Could not access the camera.")
            break
        # Convert OpenCV's BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame into a MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        # Detect hands
        result = landmarker.detect(mp_image)
        # If a hand was detected
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                # Extract our 63 numerical features
                features = extract_features(hand)
                # Print features
                #print("Features:", features)
                # Draw all 21 landmarks
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )
        # Display camera feed
        cv2.imshow("EqualVoice - Hand Detection", frame)
        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Release resources
    camera.release()
    cv2.destroyAllWindows()
