# Lava Lamp PIV (Particle Image Velocimetry)

![Lava Lamp PIV Demo](/api/placeholder/800/400 "Lava Lamp PIV in action")

## Overview

This project implements a simple, educational Particle Image Velocimetry (PIV) workflow using lava lamps as flow visualization examples. PIV is a technique used in fluid dynamics to measure velocity fields by analyzing the motion of particles between sequential image frames.

The system captures images from a webcam, processes them using OpenPIV algorithms, and visualizes the resulting velocity fields - all within a clean, modular Python framework designed for demonstrations and educational purposes.

## Features

- **Real-time image capture** from webcam
- **PIV processing** to calculate velocity fields between image pairs
- **Vector field visualization** with customizable display options
- **Modular architecture** for easy extension and modification
- **Educational tool** for fluid dynamics demonstrations

## Project Structure

```
PARTICLEIMAGEVELOCIMETRY/
├── capture_images.py      # Script to capture image pairs from webcam
├── piv_processor.py       # Class for PIV computation and visualization
├── run_piv.py             # Main script that runs the entire workflow
├── requirements.txt       # Dependencies
├── examples/              # Example images and results
│   ├── frame1.jpg         # Sample first frame
│   ├── frame2.jpg         # Sample second frame
│   └── vector_field.png   # Sample vector field output
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lava-lamp-piv.git
   cd lava-lamp-piv
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- OpenPIV

## Usage

### Capturing Images

To capture a pair of images from your webcam:

```bash
python capture_images.py
```

This will save two sequential frames (each one as you press a key) as `frame1.jpg` and `frame2.jpg` in the current directory.

### Running PIV Analysis

To run the complete PIV workflow on captured images:

```bash
python run_piv.py
```

This will:
1. Load the image pair
2. Process the images using PIV algorithms
3. Display and save the resulting vector field


## Example Output

### Input Image Pair

First frame captured from lava lamp:

![First Frame](/examples/images/lava1 "First frame from lava lamp")

Second frame captured immediately after:

![Second Frame](/examples/images/lava2 "Second frame from lava lamp")

### Resulting Vector Field

The computed velocity field:

![Vector Field Result](/examples/plots/lava "PIV vector field visualization")
