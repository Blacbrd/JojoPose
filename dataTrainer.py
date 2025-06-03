import mediapipe as mp
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Set up KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Set up for mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Utility Functions ---
def calculate_angle(a, b, c):
    a2d = np.array(a[:2])
    b2d = np.array(b[:2])
    c2d = np.array(c[:2])
    ba = a2d - b2d
    bc = c2d - b2d
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_midpoint(a, b):
    return [(a[0] + b[0])/2, (a[1] + b[1])/2, (a[2] + b[2])/2]

def get_landmark_point(landmark):
    return [int(landmark.x * frame.shape[1]),
            int(landmark.y * frame.shape[0]),
            landmark.z]

# --- Main loop for openCV ---
cap = cv2.VideoCapture(0)

try:
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
            except Exception as e:
                if str(e) == "'NoneType' object has no attribute 'landmark'":
                    pass
                else:
                    print("Error processing pose:", e)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Jojo Pose", image)
            
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

finally:
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()