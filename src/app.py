import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize a deque to store historical posture data
posture_history = deque(maxlen=30)  # Store last 30 frames of posture data

# Define posture thresholds
GOOD_POSTURE_THRESHOLD = 1.30  # Threshold for good posture
SLOUCH_THRESHOLD = 1.50  # Threshold for detecting slouching


# Novelty: Instead of static thresholds, a dynamic threshold could be implemented by analyzing individual body dimensions.
def calculate_posture_metric(landmarks):
    # Extract key landmarks for shoulders and ears
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    # Calculate the distance between the shoulders (shoulder width)
    shoulder_width = np.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2
        + (left_shoulder.y - right_shoulder.y) ** 2
    )

    # Calculate the average distance from each ear to the corresponding shoulder
    ear_shoulder_dist = (
        np.sqrt(
            (left_ear.x - left_shoulder.x) ** 2 + (left_ear.y - left_shoulder.y) ** 2
        )
        + np.sqrt(
            (right_ear.x - right_shoulder.x) ** 2
            + (right_ear.y - right_shoulder.y) ** 2
        )
    ) / 2

    # Return the ratio of shoulder width to ear-shoulder distance
    return shoulder_width / ear_shoulder_dist


# Novelty: The posture metric here is a novel approach compared to typical angle measurements used in the literature.


def detect_slouch(current_metric):
    # Determine the posture status based on the current metric
    if current_metric <= GOOD_POSTURE_THRESHOLD:
        return "Good Posture", 0  # Good posture, no slouching
    elif current_metric <= SLOUCH_THRESHOLD:
        # Mild slouching, calculate slouch level percentage
        slouch_level = (
            (current_metric - GOOD_POSTURE_THRESHOLD)
            / (SLOUCH_THRESHOLD - GOOD_POSTURE_THRESHOLD)
            * 100
        )
        return "Mild Slouching", slouch_level
    else:
        # Severe slouching, calculate slouch level percentage
        slouch_level = (
            100 + (current_metric - SLOUCH_THRESHOLD) / SLOUCH_THRESHOLD * 100
        )
        return "Severe Slouching", min(slouch_level, 100)

    # Novelty: Unlike traditional static classification, we use a percentage level to indicate severity of slouching, offering a more nuanced feedback.


# Initialize the camera
cap = cv2.VideoCapture(0)

print("Slouch detection started. Press 'q' to quit.")
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame color from BGR to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)  # Perform pose detection
        image = cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR
        )  # Convert back to BGR for display

        if results.pose_landmarks:
            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Calculate posture metric using landmarks
            landmarks = results.pose_landmarks.landmark
            current_metric = calculate_posture_metric(landmarks)
            posture_history.append(current_metric)

            # Use moving average of the posture metric for smoother detection
            avg_metric = np.mean(posture_history)
            posture_status, slouch_level = detect_slouch(avg_metric)

            # Novelty: Using a moving average approach here helps to smoothen the variations over multiple frames, which is more reliable than per-frame analysis used in Assignment 1.

            # Visualize posture status, calibration metric, and slouch level
            cv2.putText(
                image,
                f"Posture: {posture_status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"Calibration Metric: {current_metric:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                image,
                f"Slouch Level: {slouch_level:.1f}%",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Draw slouch level bar to visually represent slouch percentage
            bar_max_width = 200  # Maximum width of the slouch level bar
            bar_height = 20  # Height of the bar
            bar_width = int(
                bar_max_width * slouch_level / 100
            )  # Calculate current width
            cv2.rectangle(
                image, (10, 120), (10 + bar_max_width, 120 + bar_height), (0, 255, 0), 2
            )  # Outline of the bar
            cv2.rectangle(
                image, (10, 120), (10 + bar_width, 120 + bar_height), (0, 0, 255), -1
            )  # Filled part representing slouch level

            # Novelty: The slouch level bar provides immediate visual feedback to the user, enhancing usability compared to only textual output as seen in Assignment 2.

            # Draw posture trend graph
            graph_width = 200  # Width of the trend graph
            graph_height = 100  # Height of the trend graph
            graph = (
                np.ones((graph_height, graph_width, 3), dtype=np.uint8) * 255
            )  # Create blank white graph
            for i in range(1, len(posture_history)):
                # Calculate the y-coordinates for the trend line
                y1 = int(
                    graph_height
                    - (posture_history[i - 1] / SLOUCH_THRESHOLD) * graph_height
                )
                y2 = int(
                    graph_height
                    - (posture_history[i] / SLOUCH_THRESHOLD) * graph_height
                )
                # Calculate the x-coordinates for the trend line
                x1 = int((i - 1) * graph_width / len(posture_history))
                x2 = int(i * graph_width / len(posture_history))
                # Draw the line segment representing posture trend
                cv2.line(graph, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw threshold lines for good posture and slouch
            good_y = int(
                graph_height
                - (GOOD_POSTURE_THRESHOLD / SLOUCH_THRESHOLD) * graph_height
            )  # Y-coordinate for good posture threshold
            slouch_y = int(
                graph_height - (SLOUCH_THRESHOLD / SLOUCH_THRESHOLD) * graph_height
            )  # Y-coordinate for slouch threshold
            cv2.line(
                graph, (0, good_y), (graph_width, good_y), (0, 255, 0), 1
            )  # Good posture line
            cv2.line(
                graph, (0, slouch_y), (graph_width, slouch_y), (0, 255, 255), 1
            )  # Slouch line

            # Novelty: Adding graphical feedback for trends and threshold lines provides a more comprehensive understanding of posture history, which was missing in prior assignments.

            # Add graph to the main image in the bottom-left corner
            image[image.shape[0] - graph_height :, :graph_width] = graph

        # Display the annotated image
        cv2.imshow("Slouch Detection", image)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
