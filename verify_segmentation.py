"""
Verify that the segmentation images work correctly by applying them to capture images (RGB data) as masks.

For each image, both the original and the masked image are shown for comparison.

Usage:
1) Put capture images to `./data/verify-seg/Captures/` and segmentation images to `./data/verify-seg/Segmentation/` ;
2) Run this script.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

CAP_IMG_DIR = './data/verify-seg/Captures/'
SEG_IMG_DIR = './data/verify-seg/Segmentation/'

if __name__ == '__main__':
    # Scan the directories for capture and segmentation images
    cap_files = os.listdir(CAP_IMG_DIR)
    seg_files = os.listdir(SEG_IMG_DIR)

    # Make sure the number of capture and segmentation images are equal
    assert len(cap_files) == len(seg_files)
    # For simplicity, here we sort these files in string order, which may differ from numerical order (e.g. '2' > '10')
    # As long as the numbering scheme are consistent in both sets of images, the program will work correctly
    cap_files.sort()
    seg_files.sort()

    # Iterate over each pair of images
    for i in range(len(cap_files)):
        # Load capture image
        capture_img = cv2.imread(CAP_IMG_DIR + cap_files[i])

        # Load segmentation image as the mask
        mask = cv2.imread(SEG_IMG_DIR + seg_files[i], cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)  # Convert to black (0) and white (255)
        mask_inv = cv2.bitwise_not(mask)  # Inverted mask

        # Apply the mask to the original image
        body = cv2.bitwise_and(capture_img, capture_img, mask=mask)

        # Create a green screen for better representation of results
        green = np.zeros_like(capture_img)
        green[:] = (0, 255, 0)  # Fill it with green (B=0, G=255, R=0)
        green = cv2.bitwise_and(green, green, mask=mask_inv)  # Use the inverted mask to remove body area

        # Combine the body and the green screen
        res = cv2.bitwise_or(body, green)

        # Display the masked result alongside the capture image
        plt.imshow(cv2.cvtColor(np.hstack((capture_img, res)), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
