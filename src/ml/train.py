import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import mlflow
import os
from typing import Dict, Any

from models.wound_cnn import WoundCNN
from dataset import WoundDataset, create_data_loaders
from preprocessing import ImagePreprocessor

class ModelTrainer:
    """
    Training pipeline for Wound Classification CNN
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training configuration
        
        Args:
            config (dict): Training configuration
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Configuration
        self.config = config
        
        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Model, loss, and optimizer placeholders
        self.model = None
        self.criterion = None
        self.optimizer = None
    
    def prepare_data(self) -> tuple:
        """
        Prepare dataset for training
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # Create dataset
        dataset = WoundDataset(
            csv_file=self.config['dataset_csv'],
            root_dir=self.config['dataset_root']
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = dataset.split_dataset()
        
        # Create data loaders
        train_loader = create_data_loaders(train_dataset, 
                                           batch_size=self.config.get('batch_size', 32))
        val_loader = create_data_loaders(val_dataset, 
                                         batch_size=self.config.get('batch_size', 32))
        test_loader = create_data_loaders(test_dataset, 
                                          batch_size=self.config.get('batch_size', 32))
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self):
        """
        Initialize model, loss function, and optimizer
        """
        # Initialize model
        self.model = WoundCNN(
            num_classes=self.config.get('num_classes', 4)
        ).to(self.device)
        
        # Log model summary
        self.model.log_model_summary()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 0.001)
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for a single epoch
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            dict: Training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Optional: Log batch progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Batch {batch_idx}, Loss: {loss.item():.4f}'
                )
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': accuracy
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model performance
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                # Compute metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Compute validation metrics
        val_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total_samples
        
        return {
            'loss': val_loss,
            'accuracy': accuracy
        }
    
    def train(self):
        """
        Full training pipeline
        """
        # Start MLflow tracking
        mlflow.set_experiment('LPP_Wound_Classification')
        
        with mlflow.start_run():
            # Prepare data
            train_loader, val_loader, test_loader = self.prepare_data()
            
            # Initialize model
            self.initialize_model()
            
            # Training loop
            for epoch in range(self.config.get('epochs', 10)):
                self.logger.info(f'Epoch {epoch+1}/{self.config.get("epochs", 10)}')
                
                # Train epoch
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }, step=epoch)
                
                # Optional: Early stopping or model checkpointing logic
                self._save_checkpoint(epoch)
            
            # Final model evaluation
            self._evaluate_final_model(test_loader)
    
    def _save_checkpoint(self, epoch: int):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current training epoch
        """
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'wound_cnn_epoch_{epoch}.pth'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        self.logger.info(f'Saved checkpoint: {checkpoint_path}')
    
    def _evaluate_final_model(self, test_loader: DataLoader):
        """
        Evaluate final model performance
        
        Args:
            test_loader (DataLoader): Test data loader
        """
        test_metrics = self.validate(test_loader)
        
        # Log final test metrics to MLflow
        mlflow.log_metrics({
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy']
        })
        
        self.logger.info("Final Model Evaluation:")
        self.logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")

# Example configuration
DEFAULT_CONFIG = {
    'dataset_csv': 'data/wound_dataset.csv',
    'dataset_root': 'data/images',
    'num_classes': 4,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 10,
    'checkpoint_dir': 'checkpoints'
}

def main():
    """
    Main training script
    """
    trainer = ModelTrainer(DEFAULT_CONFIG)
    trainer.train()

if __name__ == '__main__':
    main()
