# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/left.png "Left Image"
[image2]: ./examples/center.png "Center Image"
[image3]: ./examples/right.png "Right Image"
[image4]: ./examples/nvidia_arch.png "Nvidia Model architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


#### 4. Use the simulator to collect data of good driving behavior 

I collected data using the Simulator in training mode. I did 2 laps, one clockwise and the other counter-clockwise, to have a balanced dataset.

|Left Camera Image          | Center Camera Image       | Right Camera Image |
|:-------------------------:|:-------------------------:|:------------------:|
![alt text][image1] |       ![alt text][image2] |      ![alt text][image3] 

#### 5. Design, train and validate a model that predicts a steering angle from image data.

I followed the architecure structure provided by NVIDIA in their paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf).
I added additional Dropout layers between convolutional layers to prevent overfitting.
I trained the model with 9600 samples and valided with 2400 samples with a split of 80%-20%.

|Architecture based on Nvidia Architecture|         
|:---------------------------------------:|
| ![alt text][image4]                     |

#### 6. Conclusion

The model can drive autonomously through the first track without any crashes. We can improve the model by collecting more data, driving on different tracks, and performing more than 1 lap per direction.

#### 7. Output

Here is the output video of the project: [Watch the Video](./video.mp4)
