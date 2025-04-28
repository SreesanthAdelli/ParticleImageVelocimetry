# run_piv_video.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from piv_processor import PIVProcessor

def main(video_path, win_size=32, step=1):
    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Convert to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))

    while True:
        # Read next frame
        for _ in range(step):
            ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a PIV processor using the two frames
        piv = PIVProcessor(None, None, win_size)
        piv.a = prev_frame_gray
        piv.b = frame_gray

        xs, ys, dxs, dys = piv.compute_displacement()

        ax.clear()
        norm = np.sqrt(dxs**2 + dys**2)
        ax.quiver(
            xs,
            ys[::-1],
            dxs,
            -dys,
            norm,
            cmap="plasma",
            angles="xy",
            scale_units="xy",
            scale=0.25,
        )
        ax.set_title("Live PIV Flow Field")
        ax.set_aspect('equal')
        plt.pause(0.001)  # Brief pause to allow update

        # Prepare for next iteration
        prev_frame_gray = frame_gray.copy()

    cap.release()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    video_path = "AlvinSmall.mp4"  # <-- Change to your video file
    main(video_path, win_size=32, step=1)
