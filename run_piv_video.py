import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse
from scipy.signal import correlate
import tempfile
from matplotlib.animation import FuncAnimation

# Configuration variables
# Adjust these variables to customize the PIV analysis

# Maximum vector size filter: vectors with magnitude larger than this will be removed
# Set to None to disable filtering
MAX_VECTOR_SIZE = 13  # Default value, adjust based on your specific video and flow characteristics

class VideoFramePIVProcessor:
    def __init__(self, win_size=32, max_vector_size=None):
        """
        Initialize the PIV processor for video frames
        
        Parameters:
        -----------
        win_size : int
            Size of interrogation window
        max_vector_size : float or None
            Maximum allowed vector magnitude for filtering, None for no filtering
        """
        self.win_size = win_size
        self.max_vector_size = max_vector_size
        
    def process_frame_pair(self, frame1, frame2):
        """
        Process a pair of frames to compute velocity field
        
        Parameters:
        -----------
        frame1 : numpy.ndarray
            First frame
        frame2 : numpy.ndarray
            Second frame
            
        Returns:
        --------
        xs, ys, dxs, dys, norm_drs : PIV vector field components
        """
        a, b = frame1, frame2
        win_size = self.win_size
        
        # Ensure frames are grayscale
        if len(a.shape) > 2:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        if len(b.shape) > 2:
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            
        ys = np.arange(0, a.shape[0] - win_size, win_size)
        xs = np.arange(0, a.shape[1] - win_size, win_size)
        dys = np.zeros((len(ys), len(xs)))
        dxs = np.zeros((len(ys), len(xs)))
        
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                int_win = a[y : y + win_size, x : x + win_size]
                search_win = b[y : y + win_size, x : x + win_size]
                
                cross_corr = correlate(
                    search_win - search_win.mean(), 
                    int_win - int_win.mean(), 
                    method="fft"
                )
                
                peak_y, peak_x = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
                dy, dx = (peak_y - win_size + 1), (peak_x - win_size + 1)
                dys[iy, ix], dxs[iy, ix] = dy, dx
                
        # Position vectors at window centers
        xs_mesh = xs + win_size / 2
        ys_mesh = ys + win_size / 2
        
        # Calculate magnitude
        norm_drs = np.sqrt(dxs**2 + dys**2)
        
        # Filter out vectors that exceed the maximum size if specified
        if self.max_vector_size is not None:
            mask = norm_drs > self.max_vector_size
            dxs[mask] = np.nan
            dys[mask] = np.nan
            norm_drs[mask] = np.nan
            
        return xs_mesh, ys_mesh, dxs, dys, norm_drs

def process_video_file(video_path, win_size=32, skip_frames=1, save_output=None, display=True, max_vector_size=None):
    """
    Process a video file using PIV analysis
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    win_size : int
        Size of the interrogation window
    skip_frames : int
        Number of frames to skip between analyzed pairs
    save_output : str or None
        Path to save output video, or None to not save
    display : bool
        Whether to display the results in real time
    max_vector_size : float or None
        Maximum allowed vector magnitude for filtering, None for no filtering
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
    
    # Initialize PIV processor
    piv_processor = VideoFramePIVProcessor(win_size=win_size, max_vector_size=max_vector_size)
    
    # Initialize output video writer if requested
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_output, fourcc, fps/(skip_frames+1), (frame_width, frame_height))
    
    # Initialize display
    if display:
        plt.figure(figsize=(10, 8))
        plt.ion()  # Enable interactive mode
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        if out:
            out.release()
        raise IOError("Failed to read first frame")
    
    frame_count = 0
    processed_count = 0
    
    while True:
        # Skip frames if necessary
        for _ in range(skip_frames):
            ret, _ = cap.read()
            frame_count += 1
            if not ret:
                break
                
        # Read next frame to process
        ret, curr_frame = cap.read()
        frame_count += 1
        
        if not ret:
            break
            
        # Process the frame pair
        xs, ys, dxs, dys, norm_drs = piv_processor.process_frame_pair(prev_frame, curr_frame)
        processed_count += 1
        
        # Create a visualization frame
        vis_frame = curr_frame.copy()
        
        # For display and video saving
        if display or save_output:
            # Clear previous plot
            if display:
                plt.clf()
                
            # Display original frame with vector field overlay
            plt.imshow(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))
            
            # Fix the coordinate system to match the image
            # Don't reverse the y-coordinates, but adjust for image coordinate system
            # In image coordinates, y increases downward, but in the plot, y increases upward
            q = plt.quiver(
                xs,
                ys,
                dxs,
                dys,
                color='red',  # All vectors colored red
                angles="xy",
                scale_units="xy",
                scale=0.25
            )
            
            # Add filtering information to title
            filter_info = f" (Filtered: max magnitude = {max_vector_size})" if max_vector_size is not None else ""
            plt.title(f"PIV Result - Frame {frame_count}{filter_info}")
            plt.xlabel("X")
            plt.ylabel("Y")
            
            if display:
                plt.draw()
                plt.pause(0.001)  # Small pause to update display
                
            # For video saving, convert the matplotlib figure to an image
            if save_output:
                # Convert the matplotlib figure to an image
                fig = plt.gcf()
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.resize(img, (frame_width, frame_height))
                out.write(img)
                
        # Update for next iteration
        prev_frame = curr_frame
        
        # Print progress
        print(f"\rProcessed: {processed_count}/{total_frames//(skip_frames+1)} frames", end="")
    
    # Clean up
    cap.release()
    if out:
        out.release()
    if display:
        plt.ioff()
        plt.show()
        
    print(f"\nFinished processing {processed_count} PIV frame pairs.")
    if max_vector_size is not None:
        print(f"Vectors with magnitude > {max_vector_size} were filtered out.")

# Define a variable for maximum vector size filter
# Adjust this value to filter out vectors with magnitude greater than this threshold
# Set to None to disable filtering
MAX_VECTOR_SIZE = 15  # Example value, adjust based on your specific needs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run PIV analysis on video file')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--win_size', type=int, default=32, help='Interrogation window size')
    parser.add_argument('--skip', type=int, default=0, help='Number of frames to skip between analyzed pairs')
    parser.add_argument('--save', default=None, help='Path to save output video')
    parser.add_argument('--no-display', action='store_true', help='Disable real-time display')
    parser.add_argument('--max-vector', type=float, default=MAX_VECTOR_SIZE, 
                        help='Maximum vector magnitude to display (filters larger vectors)')
    
    args = parser.parse_args()
    
    # Process the video
    process_video_file(
        args.video_path, 
        win_size=args.win_size,
        skip_frames=args.skip,
        save_output=args.save,
        display=not args.no_display,
        max_vector_size=args.max_vector
    )

if __name__ == "__main__":
    main()