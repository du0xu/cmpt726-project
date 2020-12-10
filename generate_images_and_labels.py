"""
Generate images (body + background) and labels for our model, using captured images and segmentation data.

Usage:
1. Put capture images to `./data/captures/`, segmentation images to `./data/segmentation/`,
   image JSON files to `./data/image-json/`, annotation JSON files to `./data/annotation-json/`,
   and background images to `./data/backgrounds/`.
2. Run this script. Generated images and labels will be saved to `./data/generated/`.
"""
import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMG_JSON_DIR = './data/image-json/'
ANNOT_JSON_DIR = './data/annotation-json/'
BACKGROUNDS_DIR = './data/backgrounds/'
CAPTURES_DIR = './data/captures/'
SEGMENTATION_DIR = './data/segmentation/'
GENERATED_DIR = './data/generated/'

CAP_FILE_PREFIX = 'rgb_'
SEG_FILE_PREFIX = 'segmentation_'


def preload_backgrounds():
    """
    Preload background images to a list of NumPy arrays.

    :return: a list of NumPy arrays, each representing an image.
    """
    bg_files = os.listdir(BACKGROUNDS_DIR)
    # Make sure there exist background images in the specified directory
    assert len(bg_files) > 0

    ret = []
    # Iterate over background images
    for bg_filename in bg_files:
        # Read the current file as a NumPy array and then add it to the result
        bg_img = cv2.imread(BACKGROUNDS_DIR + bg_filename)
        ret.append(bg_img)
    return ret


def calculate_crop_area(kp):
    """
    Calculates a crop area according to the provided keypoint coordinates.

    :param kp: keypoint data, a 2D NumPy array, each row representing a point.
    :return: top left and bottom right points (in tuple form) of the rectangular crop area.
    """
    [x_min, y_min, _] = np.min(kp, axis=0).astype(int)
    [x_max, y_max, _] = np.max(kp, axis=0).astype(int)
    # We need some paddings when cropping the image
    return (x_min - 50, y_min - 50), (x_max + 25, y_max + 25)


def draw_keypoints(img, kp):
    """
    Debugging function. Draw keypoints onto an image to check if the coordinates are correct.

    :param img: the image where keypoints will be drawn on.
    :param kp: keypoint data, a 2D NumPy array, each row representing a point.
    """
    for p in kp:
        p = p.astype(int)
        cv2.rectangle(img, (p[0], p[1]), (p[0], p[1]), (0, 255, 0), 5)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    # Preload background images to the memory since these images will be reused for every data point
    bg_imgs = preload_backgrounds()

    # Scan the directories for image and annotation JSON files
    img_json_files = os.listdir(IMG_JSON_DIR)
    annot_json_files = os.listdir(ANNOT_JSON_DIR)

    # Make sure the numbers of JSON files in these two directories are equal
    assert len(img_json_files) == len(annot_json_files)
    # Sort these files in string order
    img_json_files.sort()
    annot_json_files.sort()

    # Iterate over each pair of JSON files
    for i in range(len(img_json_files)):
        # Load a pair of image and annotation JSON files
        with open(IMG_JSON_DIR + img_json_files[i], 'r') as img_json:
            img_info = json.load(img_json)
        with open(ANNOT_JSON_DIR + annot_json_files[i], 'r') as annot_json:
            annot_info = json.load(annot_json)
        # Make sure the JSON data is not empty, and the number of elements matches each other
        assert (img_info is not None and len(img_info) > 0) and (
                annot_info is not None and len(annot_info) > 0) and len(img_info) == len(annot_info)

        # Save the label data as a text file, each line listing the recalculated keypoint data for an image
        with open(GENERATED_DIR + f"{i}.dat", "w") as label_file:
            # Iterate over images specified in the image JSON file
            for j in range(len(img_info)):
                # File names of the capture image and the segmentation image
                cap_file_name = img_info[j]['file_name']
                img_height = img_info[j]['height']
                seg_file_name = cap_file_name.replace(CAP_FILE_PREFIX, SEG_FILE_PREFIX)

                # Load capture image
                cap_img = cv2.imread(CAPTURES_DIR + cap_file_name)

                # Read keypoint coordinates of the current image from the annotation JSON
                keypoints = np.array(annot_info[j]['keypoints2d']).reshape(-1, 3)
                # Flip the y coordinates of keypoint data from Unity, whose origin is at the bottom left
                keypoints[:, 1] = img_height - keypoints[:, 1]

                # (For debugging only) Draw keypoints
                # draw_keypoints(cap_img, keypoints)

                # Load segmentation image as the mask
                mask = cv2.imread(SEGMENTATION_DIR + seg_file_name, cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)  # Convert to black (0) and white (255)
                mask_inv = cv2.bitwise_not(mask)  # Inverted mask

                # Apply the mask to the original image
                masked_img = cv2.bitwise_and(cap_img, cap_img, mask=mask)

                # Calculate the crop area within the masked capture image
                crop_pt1, crop_pt2 = calculate_crop_area(keypoints)
                crop_x_min, crop_y_min = crop_pt1
                crop_x_max, crop_y_max = crop_pt2
                crop_width, crop_height = crop_x_max - crop_x_min + 1, crop_y_max - crop_y_min + 1

                # Recalculate keypoint coordinates after cropping, will be saved later as label
                keypoints_cropped = np.zeros((keypoints.shape[0], 2))
                keypoints_cropped[:, 0] = keypoints[:, 0] - crop_x_min
                keypoints_cropped[:, 1] = keypoints[:, 1] - crop_y_min

                # Crop the masked image and the inverted mask (will be used later for the cropped background)
                masked_img_cropped = masked_img[crop_y_min:crop_y_max + 1, crop_x_min:crop_x_max + 1]
                mask_inv_cropped = mask_inv[crop_y_min:crop_y_max + 1, crop_x_min:crop_x_max + 1]

                # (For debugging only) Draw keypoints after cropping
                # draw_keypoints(masked_img_cropped, keypoints_cropped)

                # Randomly choose a background
                bg_img = random.choice(bg_imgs)

                # Crop the background to an area with the same size as the previously cropped masked image
                # The position of this area is chosen randomly
                bg_height, bg_width, _ = bg_img.shape
                # Randomize the position of crop area within predefined margins
                bg_crop_x_min, bg_crop_y_min = np.random.randint([100, bg_height - 50 - crop_height],
                                                                 [bg_width - 50 - crop_width,
                                                                  bg_height - 10 - crop_height])
                bg_img_cropped = bg_img[bg_crop_y_min:bg_crop_y_min + crop_height,
                                 bg_crop_x_min:bg_crop_x_min + crop_width]

                # Use the cropped inverted mask to remove body area from the cropped background
                bg_img_cropped = cv2.bitwise_and(bg_img_cropped, bg_img_cropped, mask=mask_inv_cropped)

                # Combine the body and the cropped background
                res = cv2.bitwise_or(masked_img_cropped, bg_img_cropped)

                # (For debugging only) Display the generated image
                # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                # plt.show()

                # Save the generated image
                gen_file_name = f"{i}_{j}.jpg"
                cv2.imwrite(GENERATED_DIR + gen_file_name, res)

                # Save recalculated keypoint coordinates as label
                label_file.write(" ".join(map(str, keypoints_cropped[:, 0:2].flatten().tolist())) + "\n")
