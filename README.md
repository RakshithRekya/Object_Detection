# Object Detection with ResNet-50 on CIFAR-10

# Overview
This project implements an object detection pipeline using a custom convolutional neural network with ResNet-50 as the backbone. The task focuses on object classification and localization within the CIFAR-10 dataset, leveraging PyTorch for training and evaluation.

# Key Features
Dataset: CIFAR-10, comprising 60,000 32x32 color images across 10 classes (e.g., airplanes, cars, animals).

# Model Architecture:
Backbone: ResNet-50 pre-trained on ImageNet.
Classifier: Fully connected layers for object classification.
Bounding Box Regressor: Fully connected layers for bounding box predictions.
Data Augmentation: Includes random horizontal flips, color jitter, and rotations to enhance generalization.
Metrics: Evaluates performance using Accuracy, Precision, Recall, F1 Score, and Mean Average Precision (mAP).

# Methodology
# Data Preprocessing:

Applied normalization and augmentations for the training dataset.
CIFAR-10 dataset was automatically downloaded and split for training and testing.

# Model Implementation:

Custom object detection model with separate classifiers for class prediction and bounding box regression.
Loss functions:
Classification: Cross-Entropy Loss.
Bounding Box Regression: Mean Squared Error Loss.
Optimizers:
Adam with learning rate 0.001 and weight decay 0.0005.
SGD for comparison experiments.

# Training Pipeline:

Trained for 25 epochs with batch sizes of 32 and 64 for both optimizers.
Monitored training and validation losses alongside performance metrics.

# Performance:

The Adam optimizer with a batch size of 32 achieved the best results, with an accuracy of 86.31% and mAP of 93.99%.
Training and validation losses were visualized to track convergence.

# Results
Models were tested across different optimizers (Adam and SGD) and batch sizes.
Smaller batch sizes (32) yielded better performance due to more frequent updates.
The Adam optimizer outperformed SGD in accuracy and precision.

# Scope for Improvement
Implementation of advanced object detection algorithms like Faster R-CNN.
Exploration of additional optimizers and hyperparameter tuning.
Leveraging larger datasets or transfer learning to enhance accuracy further.
