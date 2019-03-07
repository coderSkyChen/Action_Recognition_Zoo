# Action Recognition Zoo
Codes for popular action recognition models, written based on pytorch, verified on the [something-something](https://www.twentybn.com/datasets/something-something) dataset. This code is built on top of the [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch).

**Note** The main purpose of this repositoriy is to go through several methods and get familiar with their pipelines.

**Note**: always use git clone --recursive https://github.com/coderSkyChen/Action_Recognition_Zoo to clone this project Otherwise you will not be able to use the inception series CNN architecture.

## Depencies
- Opencv-2.4.13 or some greater version that has tvl1 api for the computing of optical flow.
- Pytorch-0.2.0_3
- Tensorflow-1.3.1，this is only for the using of tensorboard, it's ok without this, but you need to comment the corresponding codes.

## Data preparation
### Dataset
- **Download** the [something-something](https://www.twentybn.com/datasets/something-something) dataset. Decompress them into some folder.
- Note that this dataset contains 108,499 videos and each video is presented in JPG images. The JPG images were extracted from the orginal videos at 12 frames per seconds.
- The temporal evolution in videos is important for this dataset, so it's hard for some classic models which pay attention to short motion such as: Two-Stream Convolutional Networks for Action Recognition in Videos, NIPS 2014.
### Prepare optical flow using Opencv
Note that optical flow is an important modal feature in two-stream series methods, which contains the motion information of videos.

Since there only rgb frames in the official dataset, we need compute optical flow by ourselves.

I apply a TV-L1 optical flow algorithm, pixel values are truncated to the range \[-20, 20\], then rescaled between 0 and 255, each optical flow has two channels representing horizontal and vertical components. Note that the fps in original dataset is 12, which is too fast for optical flow computing in practice, so i sample frame at 6fps.

- The command to compute optical flow:
```
cd optical_flow
make bin                          #for cpu
make -f gpu_makefile gpu_bin      #for gpu

./bin          #for cpu
./gpu_bin      #for gpu
```
Before using the code you should modify the path in main.cpp or gpu_main.cpp.

### Generate the meta files
```
python process_dataset.py
```

# Models
Before using the code you should modify the path as your own. The test time for one video is measured on one K80.
## Two stream action recognition
**Main Reference Paper**: [Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)

- Base CNN: BN-Inception pretrained on ImageNet.
- Partical BN and cross-modality tricks have been used in the code.
- Spatial stream: it's input is single rgb frame.
- Temporal stream: it's input is stacked optical flows.
### Training
- Spatial CNN: A single rgb frame is randomly selected for a video, which equals to image classification，input channel is 3.
- Temporal CNN: 5 consequent stacked optical flows are selected for a video, input channel is 5*2(2 channels:x and y).

- The command to train models:
```
train for spatial stream:
python main.py TwoStream RGB two-stream-rgb --arch BNInception --batch_size 256 --lr 0.002
train for temporal stream:
python main.py TwoStream Flow two-stream-flow --arch BNInception --batch_size 256 --lr 0.0005
```
### Testing on validation set
At test time, given a video, i sample a fixed number of frames (25 for spatial stream and 8 for temporal stream in my experiments) with equal temporal spacing between them. From each of the frames i then obtain 10 ConvNet
inputs by cropping and flipping four corners and the center of the frame. The class scores for the
whole video are then obtained by averaging the scores across the sampled frames and crops therein.

- The command to test models:
```
test for spatial stream:
python test_models.py --model TwoStream --modality RGB --weights TwoStream_RGB_BNInception_best.pth.tar --train_id two-stream-rgb --save_scores rgb.npz --arch BNInception --test_segments 25

test for temporal stream；
python test_models.py --model TwoStream --modality Flow --weights TwoStream_Flow_BNInception_best.pth.tar --train_id two-stream-flow --save_scores flow.npz --arch BNInception --test_segments 25

```
After running the test code, we get the precision scores on validation set and the probability for all class is saved in npz files which is useful in late fusion.

```
fusion: combine spatial stream and temporal stream results.
python average_scores.py
```

## Temporal Segment Networks
**Main Reference Paper**: [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1611.05267)


- Base CNN: BN-Inception pretrained on ImageNet.
- Partical BN and cross-modality tricks have been used in the code.
- Spatial stream: it's input is k rgb frames, k is the segment number.
- Temporal stream: it's input is k stacked optical flows.
- The consensus function i've implemented is average function.

### Training

```
train spatial stream:
python main.py TSN RGB tsn-rgb --arch BNInception --batch_size 128 --lr 0.001 --num_segments 3

train temporal stream:
python main.py TSN Flow tsn-flow --arch BNInception --batch_size 128 --lr 0.0007 --num_segments 3
```
### Testing on validation set
Note that in testing phrase the k equals 1 according to the paper and it's offical code. So the segment mechanism is only used in training phrase.
```
test spatial stream:
python test_models.py --model TSN --modality RGB --weights TSN_RGB_BNInception_best.pth.tar --train_id tsn-rgb --save_scores rgb.npz --arch BNInception --test_segments 25

test temporal stream:
python test_models.py --model TSN --modality Flow --weights TSN_Flow_BNInception_best.pth.tar --train_id tsn-flow --save_scores flow.npz --arch BNInception --test_segments 25

fusion:
python average_scores.py   # need modify the path to your own
```

## Pretrained-C3D :3D Convolutional Networks
**Main Reference Paper**: [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767

- finetune the model pretrained on sports-1M, the pretrained model is upload to Baidu Cloud: [link](https://pan.baidu.com/s/1A-iAn4x45CHFgs7caOAFZw)

### Training
```
python main.py C3D RGB c3d-rgb --arch BNInception --batch_size 32 --lr 0.0001 --num_segments 1 --lr_steps 2 5 10 20 --factor 0.5
```

### Testing
```
python test_models.py --model C3D --modality RGB --weights C3D_RGB_BNInception_best.pth.tar --train_id c3d-rgb --save_scores rgb.npz --test_segments 5 --test_crops 1
```
### Results on validation set
- It seems like the C3D is faster than previous methods, but the input size for C3D is `112*112` vs `224*224` for Two-Stream models.
- The result is not good. I've found that it's hard to traing 3D CNN on this difficult dataset. This is mainly due to the poor GPU which slows the training phrase, so it's hard to choose proper hyperparameters with my machine, but this code works and it'll give you a **quick start**.

## I3D
**Main Reference Paper**: [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)

- The code for I3D model is based on [hassony2](https://github.com/hassony2/kinetics_i3d_pytorch)
- Training is too slow to report the results on Something-Something, but this code is useful
- Kinetics pretrained model is uploaded to Baidu Cloud: [link](https://pan.baidu.com/s/18pfAM2fYVsA6KxhX4A_pMQ)
### Training
```
python main.py I3D RGB i3d-rgb --arch I3D --batch_size 32 --lr 0.002 --num_segments 1 --lr_steps 2 10 20 --factor 0.5
```

