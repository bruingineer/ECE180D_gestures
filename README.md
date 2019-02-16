ECE180D_gestures
======
Gestures portion of the Synchro ECE180 Project

Code used in the python gesture scripts:
[CMU OpenPose - Python api](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### Potential bugs:
* single player game seems to be working correctly.

### Improvements:
* normalize pose dimensions to skeleton size in frame
* add more dynamic gestures

FILES
------
### openpose-process.py
1. handles the gesture processing and sends the results back to unity through mqtt
2. utilizes the [OpenPose API](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
3. waits for unity to request a gesture, then processes images in search of that gesture

### openpose-multithreaded.py
1. used to test if a multithreaded process would increase openpose processing FPS for two cameras. It did not.

### OpenPose_installation.md
1. directions on how to compile and install openpose and the python api on Windows 7 / 10
2. written by Josh while he went through the process multiple times

FOLDERS
------
### op_cpu_only
1. used for a virtual windows environment on a MacBook Pro
2. contains compiled openpose library and python packages

### op_cuda_jose
1. used on Jose's Windows laptop that has a dGPU (MX530)
2. openpose library and python packages compiled on Jose's laptop

### op_cuda_josh
1. used on Josh's Windows desktop that has a dGPU (GTX 980)
2. openpose library and python packages compiled on Josh's desktop

### models
1. contains the models and scripts necessary for openpose to work
2. getModels_poseOnly.bat is a windows script to download the large caffe model used by openpose
3. wget is included so windows can used this in the bat script

