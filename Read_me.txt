---------------------------------------
Package requirement:

Python = 3.10
Pytorch == 2.3.1

---------------------------------------
PET Image:
All images used in this research will be converted from DICOM to a 3D tensor. During this process, any values exceeding 100,000 will be capped at 100,000. Each voxel corresponds to a number, with the three dimensions represented as zxy. In our study, the z dimension can vary between different examinations, but the x and y dimensions remain consistent at 256.

For example, if a whole-body image is reconstructed as 256*256 and has 411 slices, the resulting matrix size will be 411*256*256.


#####################
p_Training_444

Training the network, and two datasets used as testing. 

Input:
1. Training input and label
   All patches first generated from the training patient, then we selected 0.7 M patches with large variance and 0.7 M patches randomly selected. They are combined together to create the training input and label.
2. Testing input and label

Output:
Trained model


#####################
post_processing_64_64

The post-processing for the testing examination. All possilbe patches will be generated and go through the network in this process. Multi-processing has been used in testing process.

Input:
1. Testing input images
   The combination of 20 sec/bed Q.Clear reconstruction with beta = 700, and TOFOSEM-PSF reconstrution


Output:
1. Testing result

#####################
Subfunction

Network structure we used in our study