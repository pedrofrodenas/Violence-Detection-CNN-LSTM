# Violence-Detection-CNN-LSTM
Violence Detection tutorial using pre-trained CNN and LSTM

This is a tutorial to see a keras code architecture to train a violence video classifier and view the flowchart. To see a detailed explanation open de Jupyter Notebook (violence_detection.ypynb).

## Introduction

Today, the amount of public violence has increased dramatically. As much in high schools as in the street. This has resulted in the ubiquitous use of surveillance cameras. This has helped the authorities to identify these events and take the necessary measures. But almost all systems today require the human-inspection of these videos to identify such events, which is virtually inefficient. It is therefore necessary to have such a practical system that can automatically monitor and identify the surveillance videos. The development of various deep learning techniques, thanks to the availability of large data sets and computational resources, has resulted in a historic change in the community of computer vision. Various techniques have been developed to address problems such as object detection, recognition, tracking, action recognition, legend generation, etc.

## Flowchart

The method consists of extracting a set of frames belonging to the video, sending them to a pretrained network called VGG16, obtaining the output of one of its final layers and from these outputs train another network architecture with a type of special neurons called LSTM. These neurons have memory and are able to analyze the temporal information of the video, if at any time they detect violence, it will be classified as a violent video.

## Prerequisites
* [Python3](https://www.python.org/)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](http://www.numpy.org)
* [matplotlib](https://matplotlib.org/users/installing.html)
* [keras](https://pypi.org/project/Keras/)
* [h5py](http://docs.h5py.org/en/stable/build.html)

## How to use
Make sure python3 and pip is installed. Then, install cv2 and numpy, matplotlib, keras and h5py.
```bash
# install opencv-python
pip install cv2
# install PyWavelets
pip install numpy
# install matplotlib
python -m pip install -U matplotlib 
# install keras
pip install Keras
# install h5py
pip install h5py
```

Let's train the violence classifier. Type on shell in project directory: 

```bash
python Violence_Detection.py
```

