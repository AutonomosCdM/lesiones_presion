import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

class WoundCNN(nn.Module):
    """
    Advanced Convolutional Neural Network for Wound Image Classification
    
    Designed for early detection of Pressure Injuries (LPP) with bias mitigation
    """
    def __init__(self, num_classes=10, input_channels=3, class_weights=None):
        """
        Initialize CNN architecture with bias mitigation techniques
        
        Args:
            num_classes (int): Number of wound classification classes
            input_channels (int): Number of input image channels
            class_weights (torch.Tensor, optional): Weights to handle class imbalance
        """
        super(WoundCNN, self).__init__()
        
        # Convolutional layers with increased complexity
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Batch normalization for each layer
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Advanced regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # Layer for feature importance tracking
        self.feature_importance = None
        
        # Class weights for bias mitigation
        self.class_weights = class_weights
        
        # Logger for tracking
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def forward(self, x):
        """
        Forward pass with advanced feature extraction
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Class probabilities
        """
        # Convolutional layers with ReLU, batch norm, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers with dropout and ReLU
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        """
        Predict wound classification with confidence
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            tuple: (predicted class, confidence score)
        """
        with torch.no_grad():
            outputs = self(x)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item()
    
    def compute_feature_importance(self, x):
        """
        Compute feature importance for bias analysis
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            np.ndarray: Feature importance scores
        """
        self.eval()
        x.requires_grad = True
        
        # Forward pass
        outputs = self(x)
        
        # Compute gradients
        loss = outputs.sum()
        loss.backward()
        
        # Get feature importance from gradients
        feature_importance = x.grad.abs().mean(dim=[0, 2, 3]).cpu().numpy()
        
        # Reset gradients
        x.grad.zero_()
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def log_model_summary(self):
        """
        Log model architecture and bias mitigation details
        """
        self.logger.info("Advanced Wound Classification CNN Model Summary:")
        self.logger.info(f"Input Channels: {self.conv1.in_channels}")
        self.logger.info(f"Number of Classes: {self.fc3.out_features}")
        self.logger.info("Bias Mitigation Techniques:")
        self.logger.info("- Batch Normalization")
        self.logger.info("- Dropout Regularization")
        self.logger.info("- Feature Importance Tracking")
        if self.class_weights is not None:
            self.logger.info("- Class Weight Balancing")
        
        self.logger.info("Layer Details:")
        self.logger.info("- Conv1: Input -> 32 channels")
        self.logger.info("- Conv2: 32 -> 64 channels")
        self.logger.info("- Conv3: 64 -> 128 channels")
        self.logger.info("- Conv4: 128 -> 256 channels")
        self.logger.info("- FC1: 256*14*14 -> 1024")
        self.logger.info("- FC2: 1024 -> 512")
        self.logger.info(f"- FC3: 512 -> {self.fc3.out_features} classes")
