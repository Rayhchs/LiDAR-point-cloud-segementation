# LiDAR Dataset and Pre-processing

LiDAR point cloud data is derived from ISPRS Benchmarks: https://www.isprs.org/education/benchmarks.aspx
This repository applies two dataset: Toronto and Vaihingen
First of all, both data formats of point cloud datasets are transformed to csv file.
Since the 3D computation cost too much of GPU and CPU space, we remove some redundant points.
Above processes are conducted by using ArcGIS and QGIS.
For Toronto dataset, we focus on detecting building and ground points.
The DSM and building footprint of training dataset in Tornoto are shown as:

<img src="https://github.com/Rayhchs/Demo/blob/main/PC/3.jpg">

As for Vaihingen dataset, Although there are more than three types in the dataset, we only focus on building, ground and tree.
We select 9 regions for training and 1 region for testing
Here shows one of the training region:

<img src="https://github.com/Rayhchs/Demo/blob/main/PC/5.jpg">


## 3D Convolution Neural Network

Secondly, we put the point data into many 3D space with size of (50,50,400). 
Precision of height is 0.01m, but we only consider to 0.1m.
The spaces are created according to the coordinate of point cloud.
On the other hand, the label data is 2D segmentation map.
For training, we adopt a 3D convolution neural network shown as:

<img src="https://github.com/Rayhchs/Demo/blob/main/PC/flow.png">

## Result (Building & Ground)

In the result, we use the detected segmentation map to back-project the detected label to each point.
Back-projected point clouds, DSM and building footprint(True data) are shown as follows:

<img src="https://github.com/Rayhchs/Demo/blob/main/PC/4.jpg">

The evaluation table is shown as:

<img src="https://github.com/Rayhchs/Demo/blob/main/PC/1.jpg">


## Result (Building & Ground & Tree)

As same as previous one, we also use the detected segmentation map to back-project the detected label to each point.
The back-projected point clouds are shown as follows:

<img src="https://github.com/Rayhchs/Demo/blob/main/PC/6.jpg">

The evaluation table is shown as:

<img src="https://github.com/Rayhchs/Demo/blob/main/PC/2.jpg">

## Discussion

In the Vaihigen dataset, the DSM data seems have some sysmetic errors. (regular grey level change from up to down and the lowest value is not zero)
In this case, the point cloud data may have some errors resulting in bad segmentation result.
On the other hand, in the result, there are also some commission errors in the boundary of the clipped 3D space.
Solving this requires better preprocessing approach to put the point into 3D space.
