import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), normalize=True):
        """
        Initialize image preprocessing pipeline
        
        Args:
            target_size (tuple): Desired image size for CNN input
            normalize (bool): Whether to apply normalization
        """
        self.transforms = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet standard normalization
                std=[0.229, 0.224, 0.225]
            ) if normalize else transforms.Lambda(lambda x: x)
        ])
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def preprocess(self, image_path):
        """
        Preprocess an image from file path
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Open image with PIL to support multiple formats
            with Image.open(image_path) as img:
                # Convert to RGB if not already
                img = img.convert('RGB')
                
                # Apply transformations
                processed_image = self.transforms(img)
                
                self.logger.info(f"Preprocessed image: {image_path}")
                return processed_image
        
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def batch_preprocess(self, image_paths):
        """
        Preprocess multiple images
        
        Args:
            image_paths (list): List of image file paths
        
        Returns:
            torch.Tensor: Batch of preprocessed images
        """
        return torch.stack([self.preprocess(path) for path in image_paths])

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']
