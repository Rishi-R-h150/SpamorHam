# Spam or Ham Email Classifier

A **Neural Network-based email classifier** that automatically identifies whether an email is **Spam** or **Ham (Not Spam)**. This project demonstrates text preprocessing, feature extraction, and deep learning-based classification for email messages.

---

## Project Overview

Email spam is a major problem affecting communication efficiency and security. This project implements a **neural network model** to classify emails accurately as spam or legitimate messages.

The workflow includes:

- Text preprocessing (tokenization, stopword removal, vectorization)  
- Neural network-based classification  
- Model evaluation using standard metrics  

---

## Features

- **Email Classification**: Automatically predict spam or ham.  
- **Text Preprocessing**: Converts raw email text into features suitable for modeling.  
- **Neural Network Model**: Multi-layer architecture for robust predictions.  
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix.  

---

## Dataset

- Dataset of labeled spam and ham emails.  
- Text preprocessing converts emails into numerical representations for the neural network.  

---

## Model Architecture

- **Input Layer**: Accepts preprocessed text features.  
- **Hidden Layers**: Fully connected layers with **ReLU activation** and **Dropout** for regularization.  
- **Output Layer**: Single neuron with **Sigmoid activation** to predict spam (1) or ham (0).

**Example in PyTorch:**

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
