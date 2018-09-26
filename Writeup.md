
## Behavior Cloning Project

### Network: 


### 1. An appropriate model architecture has been employed

My Network architecture is similar to the architecture used by NVIDIA as shown: 

![Archi.JPG](attachment:Archi.JPG)

The Network used 5 convolutional layers and 3 fully connected layers. 

The layers have the same depth and filter sizes as suggested by the end-to-end neural network used by Nvidia. 

The activation used in this case was the ReLu acitvation. 


### 2. Attempts to reduce overfitting in the model

The model was tested with many datasets with varying image characteristics and number of images. 

Hence, there was no requirement to apply dropout layers for this network. 

This similar network was sufficient to do very well on the 2nd track as well. 


### 3. Model parameter tuning

The model used the adam optimizer and the learning rate was not tuned manually


### 4. Appropriate training data

The training data was altered in the following ways: 

1. The right, left and center camera images were considered as shown: 

Center camera: 
![center_2016_12_01_13_31_13_177.jpg](attachment:center_2016_12_01_13_31_13_177.jpg)

Left camera: 
![left_2016_12_01_13_31_13_177.jpg](attachment:left_2016_12_01_13_31_13_177.jpg)

Right camera: 
![right_2016_12_01_13_31_13_177.jpg](attachment:right_2016_12_01_13_31_13_177.jpg)

2. The images were flipped to get an augmented dataset 

To make sure the vehicle does not steer off the track, recovery data was recorded as shown: 

The car was recorded while recovering from the side of the road to the center. 

![center_2018_08_07_13_32_35_023.jpg](attachment:center_2018_08_07_13_32_35_023.jpg)

The dataset was shuffled and the split of training set vs validation set was 80 percent / 20 percent 

The 'Data Generator' was used to process batches of data once at a time and avoid taxing the available Random Access Memory. 

I am currently working on training the network to work perfectly on the second track. 
