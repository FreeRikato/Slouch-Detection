import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tempfile
import os
from app import (
    mp_pose,
    mp_drawing,
    posture_history,
    GOOD_POSTURE_THRESHOLD,
    SLOUCH_THRESHOLD,
    calculate_posture_metric,
    detect_slouch,
)


def process_frame(frame, pose):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark
        current_metric = calculate_posture_metric(landmarks)
        posture_history.append(current_metric)

        avg_metric = np.mean(posture_history)
        posture_status, slouch_level = detect_slouch(avg_metric)

        # Add text overlays
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

        # Draw slouch level bar
        bar_max_width = 200
        bar_height = 20
        bar_width = int(bar_max_width * slouch_level / 100)
        cv2.rectangle(
            image, (10, 120), (10 + bar_max_width, 120 + bar_height), (0, 255, 0), 2
        )
        cv2.rectangle(
            image, (10, 120), (10 + bar_width, 120 + bar_height), (0, 0, 255), -1
        )

        # Draw posture trend graph
        graph_width = 200
        graph_height = 100
        graph = np.ones((graph_height, graph_width, 3), dtype=np.uint8) * 255
        for i in range(1, len(posture_history)):
            y1 = int(
                graph_height
                - (posture_history[i - 1] / SLOUCH_THRESHOLD) * graph_height
            )
            y2 = int(
                graph_height - (posture_history[i] / SLOUCH_THRESHOLD) * graph_height
            )
            x1 = int((i - 1) * graph_width / len(posture_history))
            x2 = int(i * graph_width / len(posture_history))
            cv2.line(graph, (x1, y1), (x2, y2), (0, 0, 255), 2)

        good_y = int(
            graph_height - (GOOD_POSTURE_THRESHOLD / SLOUCH_THRESHOLD) * graph_height
        )
        slouch_y = int(
            graph_height - (SLOUCH_THRESHOLD / SLOUCH_THRESHOLD) * graph_height
        )
        cv2.line(graph, (0, good_y), (graph_width, good_y), (0, 255, 0), 1)
        cv2.line(graph, (0, slouch_y), (graph_width, slouch_y), (0, 255, 255), 1)

        image[image.shape[0] - graph_height :, :graph_width] = graph

    return image, posture_status, slouch_level


def main():
    st.title("Slouch Detection System")
    st.write(
        "This application uses your webcam to detect and analyze your posture in real-time."
    )

    run = st.checkbox("Start Slouch Detection")
    FRAME_WINDOW = st.image([])

    posture_status_placeholder = st.empty()
    slouch_level_placeholder = st.empty()

    camera = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, posture_status, slouch_level = process_frame(frame, pose)
            FRAME_WINDOW.image(processed_frame)

            posture_status_placeholder.write(f"Current Posture: {posture_status}")
            slouch_level_placeholder.write(f"Slouch Level: {slouch_level:.1f}%")

    camera.release()


if __name__ == "__main__":
    main()
