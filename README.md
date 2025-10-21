# Face-Mask-Detection

This project detects whether a person is wearing a face mask or not in real-time images using a Convolutional Neural Network (CNN) based on MobileNetV2 architecture.
It was developed to contribute towards health safety applications, especially during the COVID-19 pandemic, by automating mask compliance checks in public areas.

**Overview**

The project uses a pre-trained MobileNetV2 model as the backbone for feature extraction and adds a custom classification head to distinguish between two categories:

With Mask

Without Mask

To improve model generalization, extensive data augmentation techniques are applied. The model is trained using transfer learning, where the convolutional base of MobileNetV2 remains frozen during initial training.

**Key Features**

Uses MobileNetV2 pre-trained on ImageNet for efficient feature extraction

Employs transfer learning for faster convergence and higher accuracy

Supports data augmentation (rotation, zoom, shift, shear, flip)

Trains with binary cross-entropy loss and Adam optimizer

Generates visualizations of training accuracy and loss curves

**Architecture**

Input Image (224x224x3)
    ↓
Preprocessing (Normalization + Augmentation)
    ↓
MobileNetV2 Base (frozen)
    ↓
AveragePooling2D → Flatten → Dense(128, ReLU) → Dropout(0.5)
    ↓
Dense(2, Softmax)
