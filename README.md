# Histopathologic Cancer Detection

## [Overview](notebooks/histopathologic_cancer.ipynb)
This project focuses on identifying metastatic cancer in small image patches taken from larger digital pathology scans. The task involves creating a binary image classification model using convolutional neural networks (CNNs) to accurately classify whether a given image contains metastatic cancer cells or not. The dataset used is a slightly modified version of the PatchCamelyon (PCam) benchmark dataset, which is known for its clinical relevance and approachability.

## [Dataset](notebooks/histopathologic_cancer.ipynb)
The dataset comprises small pathology images with corresponding labels indicating the presence of tumor tissue. The images are of size 96x96 pixels and are provided as part of the Kaggle competition.

- **Training Images**: 220,000
- **Validation Images**: 57,000
- **Image Size**: 96x96 pixels

## [Project Structure](notebooks/histopathologic_cancer.ipynb)
- `train.py`: Script to train the CNN model.
- `infer.py`: Script to perform inference using the trained model.
- `HCDNetwork.py`: Definition of the CNN model architecture.
- `utils.py`: Utility functions for data processing and visualization.
- `data/`: Directory containing the dataset. ([link](https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview))
- `output/`: Directory where model weights and results are saved.

## [Model Architecture](notebooks/histopathologic_cancer.ipynb)
The CNN model, `HCDNetwork`, is configurable with different numbers of convolutional layers and dropout rates. The architecture includes:
- Convolutional layers followed by ReLU activation and max pooling
- Fully connected layers with dropout for regularization
- Softmax output layer for classification

### Example Model Configuration
```python
params_model = {
    "shape_in": (3, 96, 96),
    "initial_filters": 8,
    "num_fc1": 100,
    "num_classes": 2,
    "dropout_rate": 0.75,  # Dropout rate
    "num_conv_layers": 4   # Number of convolutional layers
}
```

## [Training and Evaluation](notebooks/histopathologic_cancer.ipynb)
The training process involves hyperparameter tuning, trying different architectures, and applying various techniques to improve performance. The model's performance is evaluated using the area under the ROC curve (AUC).

### Training Results
| Model | Dropout Rate | Conv Layers | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|-------|--------------|-------------|---------------|-------------------|-----------------|---------------------|
| A     | 0.10         | 4           | 0.2453        | 0.9022            | 0.4719          | 0.7888              |
| B     | 0.50         | 4           | 0.3539        | 0.8538            | 0.4424          | 0.8037              |
| C     | 0.90         | 4           | 0.4535        | 0.8072            | 0.4922          | 0.7812              |
| D     | 0.75         | 3           | 0.4226        | 0.8209            | 0.4533          | 0.8250              |
| E     | 0.75         | 4           | 0.4340        | 0.8091            | 0.4873          | 0.7913              |
| F     | 0.75         | 5           | 0.1433        | 0.9603            | 0.6798          | 0.7712              |

## [Inference](notebooks/histopathologic_cancer.ipynb)
The `infer.py` script allows for performing inference on new images using the trained model. The script loads the trained model, preprocesses the input image, and outputs the predicted label and class probabilities.

### Example Usage
```python
from infer import infer

# Load the model and perform inference
model_path = 'output/trained_hcd_model.pth'
image_path = 'test/sample_image.tif'
pred_label, pred_probs = infer(model, image_path, device='cuda')

print(f'Predicted Label: {pred_label}')
print(f'Class Probabilities: {pred_probs}')
```

## [Future Work](notebooks/histopathologic_cancer.ipynb)
- **Regularization**: Implement L2 regularization (weight decay) to improve model generalization.
- **Batch Normalization**: Add batch normalization to stabilize and accelerate training.
- **Data Augmentation**: Increase data augmentation techniques to generate diverse training examples.
- **Early Stopping**: Implement early stopping to prevent overfitting and save computational resources.
- **Learning Rate Scheduling**: Use learning rate scheduling to adjust the learning rate during training.
- **Model Architecture Experimentation**: Test various architectures for potential performance improvements.
- **Transfer Learning**: Apply transfer learning using pre-trained models on similar tasks.

## [References](notebooks/histopathologic_cancer.ipynb)
- [Histopathologic Cancer Detection - Kaggle Competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview)
- [PyTorch Optim Documentation](https://pytorch.org/docs/stable/optim.html)
- [PyTorch Vision Transforms Documentation](https://pytorch.org/vision/stable/transforms.html)
- [PyTorch Learning Rate Scheduler Documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
- [Understanding Convolutional Neural Networks](https://arxiv.org/abs/1603.07285)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [Data Augmentation Techniques in Deep Learning](https://arxiv.org/abs/1712.04621)
- [Early Stopping - A Simple Way to Prevent Overfitting](https://en.wikipedia.org/wiki/Early_stopping)
- [Transfer Learning with Convolutional Neural Networks](https://arxiv.org/abs/1409.1556)

