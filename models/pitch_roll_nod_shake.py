import cv2
import dlib
import numpy as np
import os
import csv
import math

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3D model points (generic)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype="double")

def get_image_points(landmarks):
    return np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),    # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
    ], dtype="double")

def get_camera_matrix(size):
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

def get_pitch_roll(rotation_vector):
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_mat[0, 0]**2 + rotation_mat[1, 0]**2)

    if sy < 1e-6:
        pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        roll = 0
    else:
        pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
        roll = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0])

    return pitch, roll  # Return in radians

def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_filename = f"{video_name}_pose_radians.csv"

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    results = []

    prev_pitch = None
    prev_roll = None
    max_expected_delta = 0.7  # Approx 40 degrees in radians for scaling the bar graph

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if faces:
            face = faces[0]
            shape = predictor(gray, face)

            image_points = get_image_points(shape)
            camera_matrix = get_camera_matrix(frame.shape)
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)

            if success:
                pitch, roll = get_pitch_roll(rotation_vector)

                if prev_pitch is not None:
                    delta_pitch = abs(pitch - prev_pitch)
                else:
                    delta_pitch = 0.0

                if prev_roll is not None:
                    delta_roll = abs(roll - prev_roll)
                else:
                    delta_roll = 0.0

                results.append([frame_id, pitch, roll, round(delta_pitch, 3), round(delta_roll, 3)])
                prev_pitch = pitch
                prev_roll = roll

                # === Visualization ===
                cv2.putText(frame, f"Pitch: {pitch:.2f} rad", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.2f} rad", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Delta Pitch: {delta_pitch:.2f} rad", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Delta Roll: {delta_roll:.2f} rad", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)



        cv2.imshow("Nod Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    # Write CSV
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "pitch", "roll", "delta_pitch", "delta_roll"])
        writer.writerows(results)

    print(f"Saved CSV: {csv_filename}")

process_video("video.avi")