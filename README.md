# Action Recognition Zoo
Codes for popular action recognition models, verified on the [something-something](https://www.twentybn.com/datasets/something-something) dataset. This code is built on top of the [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch).

**Note**: always use git clone --recursive https://github.com/coderSkyChen/Action_Recognition_Zoo to clone this project Otherwise you will not be able to use the inception series CNN architecture.

## Data preparation
### Dataset
Download the [something-something](https://www.twentybn.com/datasets/something-something) dataset. Decompress them into some folder. Note that this dataset contains 108,499 videos and each video is presented in JPG images. The JPG images were extracted from the orginal videos at 12 frames per seconds.
### Get optical flow using TVL1 algorithm
This requires Opencv-2.4.13.
