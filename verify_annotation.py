"""
Verify annotation data (mask) by rendering the masked RGB data image.

For each image, both the original and the masked image are shown for comparison.

Usage:
1) Put RGB images and JSON data (annotation.json and rgbImage.json) to `./data/verify-annot/` ;
2) Run this script.
"""
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

ALPHA = 0.3

if __name__ == '__main__':
    # Hide the axes
    plt.axis('off')

    # Read JSON data
    with open('data/verify-annot/rgbImage.json', 'r') as img_info_f:
        img_info_list = json.load(img_info_f)
    with open('data/verify-annot/annotation.json', 'r') as annot_f:
        annot_info_list = json.load(annot_f)
    # Make sure the data is not empty
    assert img_info_list is not None and annot_info_list is not None

    # Iterate over each image
    for img_info in img_info_list:
        # Load the image
        file_name = img_info['file_name']
        img = cv2.imread(f'data/verify-annot/{file_name}')
        if img is None:
            print(f'Cannot open image "{file_name}"')
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find annotation info whose id matches the current image id
        id_ = img_info['id']
        annot_info = next(e for e in annot_info_list if e['id'] == id_)
        if annot_info is None:
            print(f'Cannot find annotation info for image id={id_}')
            continue

        # Reshape raw keypoints data to a list of 2D tuples
        pts_raw = annot_info['keypoints2d']
        pts_mat = np.array(pts_raw).reshape(-1, 2)
        poly = [tuple(pt_v) for pt_v in pts_mat]

        # Draw the polygon mask
        mask = np.zeros_like(img_gray)  # Empty mask
        cv2.fillPoly(mask, np.array([poly]), 255)

        # Apply the mask to the original image
        res = cv2.bitwise_and(img, img, mask=mask)

        # Display the masked result alongside the original
        plt.imshow(cv2.cvtColor(np.vstack((img, res)), cv2.COLOR_BGR2RGB))
        plt.show()
