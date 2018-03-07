# Action Recognition Zoo
Codes for popular action recognition models, written based on pytorch, verified on the [something-something](https://www.twentybn.com/datasets/something-something) dataset. This code is built on top of the [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch).

**Note**: always use git clone --recursive https://github.com/coderSkyChen/Action_Recognition_Zoo to clone this project Otherwise you will not be able to use the inception series CNN architecture.

## Depencies
- Opencv-2.4.13 or some greater version that has tvl1 api for the computing of optical flow.
- Pytorch-0.2.0_3
- Tensorflow-1.3.1，this is only for the using of tensorboard, it's ok without this, but you need to comment the corresponding codes.

## Install
git clone --recursive https://github.com/coderSkyChen/Action_Recognition_Zoo

## Data preparation
### Dataset
- **Download** the [something-something](https://www.twentybn.com/datasets/something-something) dataset. Decompress them into some folder.
- Note that this dataset contains 108,499 videos and each video is presented in JPG images. The JPG images were extracted from the orginal videos at 12 frames per seconds.
- The temporal evolution in videos is important for this dataset, so it's hard for some classic models such as: Two-Stream Convolutional Networks for Action Recognition in Videos, NIPS 2014.
### Prepare optical flow using Opencv
Note that optical flow is an important modal feature in two-stream series methods, which contains the motion information of videos.

Since there only rgb frames in the official dataset, we need compute optical flow by ourselves.

I apply a TV-L1 optical flow algorithm, pixel values are truncated to the range \[-20, 20\], then rescaled between 0 and 255, each optical flow has two channels representing horizontal and vertical components. Note that the fps in original dataset is 12, which is too fast for optical flow computing in practice, so i sample frame at 6fps.

```
cd optical_flow
make bin                          #for cpu
make -f gpu_makefile gpu_bin      #for gpu

./bin          #for cpu
./gpu_bin      #for gpu
```
Before using the code you should modify the path in main.cpp or gpu_main.cpp.

