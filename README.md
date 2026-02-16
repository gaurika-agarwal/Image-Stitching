## Overview
This project implements image stitching using the Scale-Invariant Feature Transform (SIFT) algorithm. It detects keypoints in overlapping images, matches features, computes homography, and stitches images together to create a panoramic output.

The project demonstrates fundamental computer vision concepts such as feature detection, feature matching, and geometric transformation using OpenCV.

## Features
- SIFT feature detection
- Keypoint matching using BFMatcher / FLANN
- Homography estimation
- Image warping and stitching
- Panoramic image generation

## Project Workflow
1. Load input images
2. Detect SIFT keypoints and descriptors
3. Match features between images
4. Compute homography matrix
5. Warp and stitch images
6. Display final panorama


## Technologies Used
- Python
- OpenCV
- NumPy



## Author
Gaurika Agarwal
