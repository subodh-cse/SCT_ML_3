# Cats vs Dogs Image Classification using HOG and Support Vector Machine (SVM)

# Overview

This project implements an image classification system that distinguishes between cats and dogs using Histogram of Oriented Gradients (HOG) feature extraction and a Support Vector Machine (SVM) classifier.

The model is trained on the Dogs vs Cats dataset and uses computer vision techniques to extract meaningful image features before classification.

# Features

* Image preprocessing using OpenCV
* HOG feature extraction
* Support Vector Machine (SVM) classification
* Dataset visualization
* Confusion Matrix visualization
* Accuracy, Precision, Recall, and F1-Score evaluation
* Prediction on new images
* Trained model saving using Joblib

# Dataset

Dataset: Dogs vs Cats Classification Dataset

Dataset Structure:

dogs-vs-cats-classification

Dataset Structure/

train/
├── cats/
└── dogs/

validation/
├── cats/
└── dogs/

test/
├── cats/
└── dogs/

# Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-Learn
* Scikit-Image
* Matplotlib
* Joblib

# Machine Learning Workflow

1. Load cat and dog images
2. Convert images to grayscale
3. Resize images to a fixed size
4. Extract HOG features
5. Split dataset into training and testing sets
6. Train Linear SVM classifier
7. Evaluate model performance
8. Generate confusion matrix
9. Save trained model
10. Predict new images

# HOG Feature Extraction

Histogram of Oriented Gradients (HOG) is a feature descriptor used in computer vision and image processing for object detection.

HOG captures:

* Edge information
* Shape information
* Object structure
* Gradient orientation patterns

These features help improve classification performance compared to raw pixel values.

# Support Vector Machine (SVM)

Support Vector Machine is a supervised machine learning algorithm used for classification tasks.

Advantages:

* Effective in high-dimensional spaces
* Works well with image classification problems
* Good generalization performance
* Suitable for medium-sized datasets

# Project Structure

Project Structure

SCT_ML_3/
│
├── train.py
├── predict.py
├── README.md
├── requirements.txt
│
└── cat_vs_dog/
    └── dogs-vs-cats-classification/
        ├── train/
        │   ├── cats/
        │   └── dogs/
        │
        ├── validation/
        │   ├── cats/
        │   └── dogs/
        │
        ├── test/
        │   ├── cats/
        │   └── dogs/
        │
        └── dataset_info.csv

# Installation

Install all required dependencies:

pip install -r requirements.txt

# Training the Model

Run:

python train.py

The script will:

* Load images
* Extract HOG features
* Train the SVM classifier
* Display dataset distribution
* Display confusion matrix
* Save the trained model

# Making Predictions

Place an image named:

test_image.jpg

inside the project directory.

Run:

python predict.py

The model will display:

* Input image
* Predicted class (Cat or Dog)

# Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

# Results

The project achieves classification performance using HOG feature extraction combined with a Linear Support Vector Machine.

Performance may vary depending on:

* Number of training images
* Image resolution
* HOG parameters
* System configuration

# Future Improvements

* Hyperparameter tuning
* Cross-validation
* Data augmentation
* CNN-based classification
* Streamlit web application deployment
* Real-time image classification

# Author

Subodh

Note:Dataset is not included in this repository due to size limitations. Download it from Kaggle and place it in the cat_vs_dog directory
