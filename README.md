# CMPT 726 course project

## Work in progress

This project is not finished yet. More details will be added later.

## Overview

This repository includes the training part. For the data collection part, check
out [this repository](https://github.com/chenjshihchieh/Unity-data-collection).

Before training our model, we need to generate images and labels based on the data we collected using Unity.

The diagram below illustrates the process of generating the images:
![](./doc/images/data-generation.png)

Each label is composed of (x, y) coordinates of the 21 keypoints after cropping the capture image (a total of 42
numbers), ultimately derived from annotation data captured in Unity.

## Milestones

1. (Fall 2020) ...
2. ...

## How to run

To verify segmentation images:

- Put capture images to `./data/verify-seg/Captures/` and segmentation images to `./data/verify-seg/Segmentation/`;
- Run `verify_segmentation.py`.

Example result:

![](./doc/images/seg-verification.png)

How to generate images and labels for the model:

1. Put capture images to `./data/captures/`, segmentation images to `./data/segmentation/`, image JSON files
   to `./data/image-json/`, annotation JSON files to `./data/annotation-json/`, and background images
   to `./data/backgrounds/`.
2. Run `./generate_images_and_labels.py`. Generated images and labels will be saved to `./data/generated/`.

To train the model:

- ...
