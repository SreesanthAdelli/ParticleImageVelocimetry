import cv2
import os
from skimage.io import imsave

def capture_images():
    cap = cv2.VideoCapture(0)  # 0 is default webcam

    if not cap.isOpened():
        raise IOError("Cannot access webcam")

    print("Press any key to capture the first image...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Capture - Frame 1', frame)
        if cv2.waitKey(1) != -1:
            frame1 = frame
            break

    print("Press any key to capture the first image...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Capture - Frame 2', frame)

        if cv2.waitKey(1) != -1:  # if *any* key is pressed
            frame2 = frame
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Save images
    imsave("frame_1.tif", frame1_gray)
    imsave("frame_2.tif", frame2_gray)
    print("Images saved as frame_1.tif and frame_2.tif.")

if __name__ == "__main__":
    capture_images()
