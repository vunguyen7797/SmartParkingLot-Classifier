# Smart Parking Lot Classifier
This is the classifier script used to deploy a Deep Learning model on Raspberry Pi 4 to detect whether a parking space is empty or occupied.

This is a part of the Smart Parking Lot System - Capstone Senior Project.

## Model Description
* The model is trained using Transfer  Learning method with TensorFlow Lite to be used on Raspberry Pi 4.
* Dataset: **PKLot - A robust dataset for parking lot classification** consisting of 695,899 images of occupied and empty parking spaces captured from two parking lots with different camera views and under different weather conditions.
* This model only works with pre-labeled spaces.

## Model Benchmark
 * When being tested at a parking lot with 222 spaces being observed:
	 * The model achieved an accuracy of 85%.
	 * Precision is 73%
	 * Recall is 88% which is the proportion of actual vacant slots identified correctly.
	 
	 ![enter image description here](https://github.com/vunguyen7797/SmartParkingLot-Classifier/blob/main/screenshots/Picture1.png?raw=true)

## Built With:
* Python
* OpenCV
* TensorFlow Lite
