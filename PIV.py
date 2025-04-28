import cv2
import numpy as np
from matplotlib import pyplot as plt

def compute_piv(image1, image2, window_size=32, step_size=16):
    """
    Function to compute PIV (Particle Image Velocimetry) using cross-correlation method.

    Args:
    - image1: First image (t)
    - image2: Second image (t + dt)
    - window_size: The size of the correlation window.
    - step_size: The step size for sliding the window.

    Returns:
    - displacement_x: Horizontal displacement field.
    - displacement_y: Vertical displacement field.
    """
    # Convert images to grayscale if they are not
    

    # Initialize displacement arrays
    displacement_x = []
    displacement_y = []

    # Loop through the images in steps, sliding the window
    for y in range(0, image1.shape[0] - window_size, step_size):
        for x in range(0, image1.shape[1] - window_size, step_size):
            # Extract the sub-images (windows) from both images
            window1 = image1[y:y+window_size, x:x+window_size]
            window2 = image2[y:y+window_size, x:x+window_size]

            # Compute the cross-correlation
            correlation = cv2.matchTemplate(window2, window1, method=cv2.TM_CCOEFF_NORMED)

            # Find the location of the maximum correlation
            _, _, _, max_loc = cv2.minMaxLoc(correlation)

            # Calculate the displacement (difference between max location and top-left corner)
            displacement_x.append(max_loc[0] - window_size // 2)
            displacement_y.append(max_loc[1] - window_size // 2)

    return np.array(displacement_x), np.array(displacement_y)

def visualize_piv(displacement_x, displacement_y, image_shape, step_size=16):
    """
    Visualize the PIV results as vector field plot.

    Args:
    - displacement_x: Horizontal displacement.
    - displacement_y: Vertical displacement.
    - image_shape: Shape of the image.
    - step_size: Step size of the grid.
    """
    # Create a meshgrid of the points where displacements were calculated
    X, Y = np.meshgrid(np.arange(0, image_shape[1], step_size),
                       np.arange(0, image_shape[0], step_size))

    # Plot the displacements as a quiver plot
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, displacement_x, displacement_y, scale=10)
    plt.gca().invert_yaxis()
    plt.title("PIV Displacement Field")
    plt.show()

def capture_images(num_images=2, width=640, height=480):
    """
    Capture a few images using the webcam.

    Args:
    - num_images: Number of images to capture
    - width: Width of the video frame
    - height: Height of the video frame

    Returns:
    - images: List of captured images
    """
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return []

    cap.set(3, width)  # Set video frame width
    cap.set(4, height)  # Set video frame height

    images = []
    for i in range(num_images):
        print(f"Capturing image {i+1}/{num_images}...")
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        images.append(frame)
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(500)  # Wait for 500ms to capture next frame

    cap.release()
    cv2.destroyAllWindows()
    return images

# Capture a few images from the webcam
captured_images = capture_images(num_images=2)

# Ensure we have enough images to run PIV
if len(captured_images) < 2:
    print("Error: Need at least 2 images to run PIV.")
else:
    # Compute PIV between the two captured images
    image1, image2 = captured_images[0], captured_images[1]
    displacement_x, displacement_y = compute_piv(image1, image2)

    # Visualize the PIV results
    visualize_piv(displacement_x, displacement_y, image1.shape)
