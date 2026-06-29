# Panorama Stitching using ORB and Homography

A computer vision project that creates a panoramic image by stitching multiple overlapping images without using OpenCV's built-in `Stitcher` class.

The application detects and matches image features using ORB, estimates the transformation between images through Homography and RANSAC, and generates a final cropped panorama.

## Features

* Feature detection using **ORB (Oriented FAST and Rotated BRIEF)**
* Feature matching using **Brute Force Hamming Matcher**
* Homography estimation using **RANSAC**
* Perspective transformation and image warping
* Sequential stitching of multiple images
* Removal of black borders from the final panorama

## Project Workflow

1. Load the input images.
2. Detect keypoints and descriptors using ORB.
3. Match descriptors between image pairs.
4. Keep only the best matches (top 15%).
5. Compute the homography matrix using RANSAC.
6. Apply perspective transformation and stitch the images together.
7. Crop the final image to remove black borders.

## Technologies

* Python
* OpenCV
* NumPy
* Matplotlib

## Project Structure

```text
panorama-stitching/
│
├── panorama_stitching.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/panorama-stitching.git
cd panorama-stitching
pip install -r requirements.txt
```

## Requirements

```text
opencv-python
numpy
matplotlib
```

## Example Result

The original project was developed using three overlapping images and generated a seamless panoramic image by estimating the geometric transformation between consecutive images.

> The repository currently contains the source code and documentation of the project. Input images and generated outputs can be reproduced by running the application with any set of overlapping images.

## Key Concepts

* Feature Detection
* Feature Matching
* Image Registration
* Homography Estimation
* Perspective Transformation
* RANSAC
* Image Stitching
* Computer Vision

## Future Improvements

* Automatic border detection and cropping.
* Support for an arbitrary number of images.
* Multi-band blending to reduce visible seams.
* GPU acceleration for large image sets.
* Automatic image ordering before stitching.

Faculty of Electronics, Telecommunications and Information Technology (TST)
National University of Science and Technology POLITEHNICA Bucharest
