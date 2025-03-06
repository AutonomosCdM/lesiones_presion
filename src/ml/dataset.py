import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
import pandas as pd
import numpy as np

class WoundDataset(Dataset):
    """
    Custom Dataset for Wound Images
    Manages medical image data for Pressure Injury detection
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Initialize dataset
        
        Args:
            csv_file (str): Path to CSV with image metadata
            root_dir (str): Root directory of image files
            transform (callable, optional): Optional transform to be applied
        """
        self.wound_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Validate dataset
        self._validate_dataset()
    
    def _validate_dataset(self):
        """
        Validate dataset integrity
        """
        missing_files = []
        for idx, row in self.wound_frame.iterrows():
            img_path = os.path.join(self.root_dir, row['filename'])
            if not os.path.exists(img_path):
                missing_files.append(img_path)
        
        if missing_files:
            self.logger.warning(f"Missing {len(missing_files)} image files")
            for file in missing_files[:5]:  # Log first 5 missing files
                self.logger.warning(f"Missing file: {file}")
    
    def __len__(self):
        """
        Return total number of samples
        """
        return len(self.wound_frame)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (image, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, 
                                self.wound_frame.iloc[idx]['filename'])
        
        # Read image
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            self.logger.error(f"Error loading image {img_name}: {e}")
            raise
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.wound_frame.iloc[idx]['wound_stage']
        
        return image, label
    
    def get_class_distribution(self):
        """
        Analyze class distribution in the dataset
        
        Returns:
            dict: Distribution of wound stages
        """
        distribution = self.wound_frame['wound_stage'].value_counts()
        self.logger.info("Wound Stage Distribution:")
        for stage, count in distribution.items():
            self.logger.info(f"Stage {stage}: {count} images")
        
        return distribution.to_dict()
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            train_ratio (float): Proportion of training data
            val_ratio (float): Proportion of validation data
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        # Ensure reproducibility
        np.random.seed(42)
        
        # Shuffle the dataset
        shuffled_indices = np.random.permutation(len(self))
        
        # Calculate split indices
        train_end = int(len(self) * train_ratio)
        val_end = train_end + int(len(self) * val_ratio)
        
        # Create splits
        train_indices = shuffled_indices[:train_end]
        val_indices = shuffled_indices[train_end:val_end]
        test_indices = shuffled_indices[val_end:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(self, train_indices)
        val_dataset = torch.utils.data.Subset(self, val_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)
        
        self.logger.info(f"Dataset Split:")
        self.logger.info(f"Training set: {len(train_indices)} samples")
        self.logger.info(f"Validation set: {len(val_indices)} samples")
        self.logger.info(f"Test set: {len(test_indices)} samples")
        
        return train_dataset, val_dataset, test_dataset

# Example usage
def create_data_loaders(dataset, batch_size=32, num_workers=4):
    """
    Create DataLoaders for a given dataset
    
    Args:
        dataset (Dataset): Input dataset
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes
    
    Returns:
        DataLoader: Configured DataLoader
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
