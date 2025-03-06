import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class WoundCNN(nn.Module):
    """
    Convolutional Neural Network for Wound Image Classification
    
    Designed for early detection of Pressure Injuries (LPP)
    """
    def __init__(self, num_classes=4, input_channels=3):
        """
        Initialize CNN architecture
        
        Args:
            num_classes (int): Number of wound severity/stage classes
            input_channels (int): Number of input image channels
        """
        super(WoundCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Logger for tracking
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Class probabilities
        """
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Predict wound classification
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            int: Predicted wound class
        """
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()
    
    def log_model_summary(self):
        """
        Log model architecture details
        """
        self.logger.info("Wound Classification CNN Model Summary:")
        self.logger.info(f"Input Channels: 3")
        self.logger.info(f"Number of Classes: 4")
        self.logger.info("Layers:")
        self.logger.info("- Conv1: 3 -> 32 channels")
        self.logger.info("- Conv2: 32 -> 64 channels")
        self.logger.info("- Conv3: 64 -> 128 channels")
        self.logger.info("- FC1: 128*28*28 -> 512")
        self.logger.info("- FC2: 512 -> 4 classes")
