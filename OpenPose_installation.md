# HOW TO INSTALL OPENPOSE PYTHON API - Windows 7, 10

This is a log Josh created while following this link:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#operating-systems
FAQ: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/faq.md

* installed GitHub desktop
* cloned most recent open pose master commit
  * want the added python functionality
* install Visual Studio 2015 update 3
* install cmake
* install python 3.7 in default location
  * add to path
  * pip install numpy opencv-python
* download cuda 8.0 and patch 2
  * https://developer.nvidia.com/cuda-80-ga2-download-archive 
  * install cuda toolkit 8.0
  * install patch 2
  * if CUDA 8 does not work because of the type of graphics card, try CUDA 9
* download cuDNN 5.1
  * unzip folder
  * copy include and lib contents into CUDA\v8.0 folder
* cmake will downloads the models
* cmake also checks for the zip files of opencv, caffe, and its dependencies in the 3rd party folder
* configure and generate with cmake
  * When asked what to use as the generator, pick “Visual Studio 14 2015 win64”
  * the config option ‘gpu mode’ is CUDA or CPU_ONLY depending on what you want
* open the VS solution, openpose.sln, in the build directory
  * select the Release and x64 configuration, not debug or x86 (32bit) ones
  * in the right hand pane, right click the top folder and select build
  * also build the python wrapper (pyopenpose)
    * right click the pyopenpose folder at the bottom of the listed files and select build
* In order to use OpenPose outside Visual Studio, and assuming you have not unchecked the BUILD_BIN_FOLDER flag in CMake, copy all DLLs from {build_directory}/bin into the folder where the generated openpose.dll and *.exe demos are, e.g., {build_directory}/x64/Release for the 64-bit release version
  * do this after every build
