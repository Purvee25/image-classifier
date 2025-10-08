# Image Classifier

A lightweight image classification framework supporting custom model architectures and transfer learning.

## Features
- Custom CNN architecture builder
- Data augmentation pipeline
- Model evaluation metrics
- Confusion matrix visualization

## Quick Start
```python
from classifier import ImageClassifier
model = ImageClassifier(num_classes=10)
model.train(train_data, epochs=20)
```
