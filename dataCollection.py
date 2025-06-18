import mediapipe as mp
import cv2
import csv
import time
import math
import os

# Set up MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

LABEL = "JoDio"

# CSV filename to store the feature data
csv_filename = fr"csvFiles/body_features_{LABEL}.csv"

# Define landmarks of interest for distances and angles
# Distance pairs: (i, j)
distance_pairs = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ANKLE),
]
# Angle triplets: (a, b, c) angle at b between a-b and c-b
angle_triplets = [
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER),
]

# Build CSV header
def build_header():
    header = ["Label"]
    for (i, j) in distance_pairs:
        header.append(f"dist_{i.name.lower()}_{j.name.lower()}")
    for (a, b, c) in angle_triplets:
        header.append(f"angle_{a.name.lower()}_{b.name.lower()}_{c.name.lower()}")
    return header

# Check if file exists to decide write or append
file_exists = os.path.isfile(csv_filename)
mode = 'a' if file_exists else 'w'
with open(csv_filename, mode=mode, newline='') as csv_file:
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(build_header())

print("Press ESC to exit.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Helper functions

def get_landmark(landmarks, lm):
    """Return landmark or None if not visible enough"""
    val = landmarks[lm.value]
    return (val.x, val.y, val.z) if val.visibility >= 0.8 else None


def compute_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return math.dist(p1[:2], p2[:2])


def compute_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    # vectors ba and bc
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return None
    cos_angle = max(-1.0, min(1.0, dot/mag))
    return math.degrees(math.acos(cos_angle))

print("Get ready to pose...")
start = time.time()
while time.time() - start < 5.0:
    pass
print("Pose now!")
start2 = time.time()

try:
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened() and (time.time() - start2 < 8.0):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = pose.process(img_rgb)
            img_rgb.flags.writeable = True
            output = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Jojo Pose", output)

            # Extract features
            row = [LABEL]
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                # gather points
                pts = {lm: get_landmark(lms, lm) for lm in set([i for pair in distance_pairs for i in pair] + [b for (_, b, _) in angle_triplets] + [a for (a, _, _) in angle_triplets] + [c for (_, _, c) in angle_triplets])}
                # compute distances
                hip_width = compute_distance(pts[mp_pose.PoseLandmark.LEFT_HIP], pts[mp_pose.PoseLandmark.RIGHT_HIP]) or 1.0
                for (i, j) in distance_pairs:
                    d = compute_distance(pts[i], pts[j])
                    row.append(None if d is None else d/hip_width)
                # compute angles
                for (a, b, c) in angle_triplets:
                    row.append(compute_angle(pts[a], pts[b], pts[c]))
            else:
                row.extend([None] * (len(distance_pairs) + len(angle_triplets)))

            # Save row to CSV
            with open(csv_filename, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
