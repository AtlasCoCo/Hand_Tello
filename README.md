# Hand_Tello

a simple use Hand Gesture to control Tello UAV based on tensorflow. 
Gesture recognition have two step: 1. Find Hand joints; 2. Gesture Classification.

@danquxunhuan130
MIT LICENSE

### dependency:

```
tensorflow-gpu == 1.11.0
Anaconda 5.2.0
Python 3.6.5
ffmpeg == 4.0.2
av == 6.0.0
cuda V8.0.1
opencv-python == 3.4.3.18
tellopy == 0.6.0.dev0
```

### trained model:
Two tensorflow model (`classify` and `joint`) in `models` folder.

`joint` to find 21 hand joints, and generating Joint Map for Gesture Classification.

`classify` a simple model to classify Gesture Classification from Joint Map.

Not upload due to their large size.


### run:

```shell
python test.py # test hand gesture classification.

python telloCV.py # test control tello. (not test yet)
```

### ref:

Tello Control: https://github.com/Ubotica/telloCV.git , https://github.com/hanyazou/TelloPy.git

Hand Gesture: https://github.com/yyyerica/HandGestureClassify , https://github.com/timctho/convolutional-pose-machines-tensorflow


Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh, "Convolutional Pose Machines", CVPR 2016.
