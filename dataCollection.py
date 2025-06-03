import mediapipe as mp
import cv2
import csv

# Set up MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# CSV filename to store the landmark data
csv_filename = r"csvFiles/body_landmarks.csv"

# Define header for the CSV (99 values: x, y, z for each of 33 landmarks)
header = [f"Landmark_{i}_{coord}" for i in range(33) for coord in ['x', 'y', 'z']]

# Write the header to the CSV file
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)

# Instructions for the user
print(f"Press ENTER to capture the current body landmark data into {csv_filename}.")
print("Press ESC to exit.")

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Main loop with Pose processing
try:
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty frame.")
                break

            # Flip the frame horizontally for a mirror view
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw crosshair lines for alignment
            height, width, _ = image.shape
            center_x = width // 2
            center_y = height // 2
            cv2.line(image, (center_x, 0), (center_x, height), (0, 255, 0), 1)
            cv2.line(image, (0, center_y), (width, center_y), (0, 255, 0), 1)

            # Reset current landmarks for this frame
            current_landmark_values = None

            # If body landmarks are detected, extract and draw them
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                current_landmark_values = []
                for landmark in landmarks:
                    current_landmark_values.extend([landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Show the processed frame
            cv2.imshow("Jojo Pose", image)

            # Capture key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break
            elif key == 13:  # Enter key to capture
                if current_landmark_values is not None:
                    with open(csv_filename, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(current_landmark_values)
                    print("Data captured.")
                else:
                    print("No body detected to capture.")

finally:
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()