import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
import logging
from typing import List, Dict, Any

class ModelEvaluator:
    """
    Comprehensive model evaluation framework for medical image classification
    
    Focuses on performance, generalization, and fairness assessment
    """
    def __init__(self, model, device='cuda'):
        """
        Initialize model evaluator
        
        Args:
            model (torch.nn.Module): Trained neural network model
            device (str): Computation device (cuda/cpu)
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def multi_class_performance(self, dataloader) -> Dict[str, Any]:
        """
        Evaluate multi-class classification performance
        
        Args:
            dataloader (torch.utils.data.DataLoader): Test dataset
        
        Returns:
            Dict containing performance metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute detailed performance metrics
        performance = {
            'accuracy': metrics.accuracy_score(all_labels, all_preds),
            'precision': metrics.precision_score(all_labels, all_preds, average='weighted'),
            'recall': metrics.recall_score(all_labels, all_preds, average='weighted'),
            'f1_score': metrics.f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': metrics.confusion_matrix(all_labels, all_preds)
        }
        
        self.logger.info("Multi-Class Performance Metrics:")
        for metric, value in performance.items():
            if metric != 'confusion_matrix':
                self.logger.info(f"{metric.capitalize()}: {value:.4f}")
        
        return performance
    
    def generalization_test(self, test_dataloaders: List[torch.utils.data.DataLoader]) -> Dict[str, float]:
        """
        Assess model performance across different datasets
        
        Args:
            test_dataloaders (List): Multiple test datasets
        
        Returns:
            Dict of performance across different datasets
        """
        generalization_results = {}
        
        for i, dataloader in enumerate(test_dataloaders, 1):
            performance = self.multi_class_performance(dataloader)
            generalization_results[f'Dataset_{i}'] = performance['accuracy']
            
            self.logger.info(f"Generalization Test - Dataset {i} Accuracy: {performance['accuracy']:.4f}")
        
        # Compute generalization stability
        accuracy_std = np.std(list(generalization_results.values()))
        generalization_results['stability'] = accuracy_std
        
        self.logger.info(f"Generalization Stability (Std Dev): {accuracy_std:.4f}")
        
        return generalization_results
    
    def bias_fairness_assessment(self, dataloader, sensitive_attributes: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Comprehensive bias and fairness analysis
        
        Args:
            dataloader (torch.utils.data.DataLoader): Test dataset
            sensitive_attributes (Dict): Mapping of sensitive attributes to class indices
        
        Returns:
            Dict containing bias metrics
        """
        self.model.eval()
        bias_results = {}
        
        with torch.no_grad():
            for attribute, classes in sensitive_attributes.items():
                # Filter dataset for specific attribute/classes
                attribute_performance = self._compute_attribute_performance(
                    dataloader, attribute, classes
                )
                bias_results[attribute] = attribute_performance
        
        # Compute overall bias disparity
        disparities = [
            result['performance_gap'] 
            for result in bias_results.values() 
            if 'performance_gap' in result
        ]
        
        bias_results['total_bias_disparity'] = np.mean(disparities) if disparities else 0
        
        self.logger.info("Bias and Fairness Assessment:")
        for attr, result in bias_results.items():
            self.logger.info(f"{attr} Performance Gap: {result.get('performance_gap', 'N/A')}")
        
        return bias_results
    
    def _compute_attribute_performance(self, dataloader, attribute, target_classes):
        """
        Compute performance for specific sensitive attributes
        
        Args:
            dataloader (torch.utils.data.DataLoader): Test dataset
            attribute (str): Sensitive attribute name
            target_classes (List[int]): Classes to analyze
        
        Returns:
            Dict of performance metrics for the attribute
        """
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Filter for specific classes
                mask = torch.isin(labels, torch.tensor(target_classes).to(self.device))
                
                all_preds.extend(predicted[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        
        # Compute performance
        overall_accuracy = metrics.accuracy_score(all_labels, all_preds)
        
        return {
            'accuracy': overall_accuracy,
            'performance_gap': 1 - overall_accuracy  # Potential bias indicator
        }
    
    def generate_comprehensive_report(self, 
                                      test_dataloader, 
                                      generalization_dataloaders=None, 
                                      sensitive_attributes=None):
        """
        Generate a comprehensive model evaluation report
        
        Args:
            test_dataloader (torch.utils.data.DataLoader): Primary test dataset
            generalization_dataloaders (List, optional): Additional test datasets
            sensitive_attributes (Dict, optional): Sensitive attributes for bias analysis
        
        Returns:
            Dict containing full evaluation report
        """
        report = {
            'multi_class_performance': self.multi_class_performance(test_dataloader)
        }
        
        if generalization_dataloaders:
            report['generalization_test'] = self.generalization_test(generalization_dataloaders)
        
        if sensitive_attributes:
            report['bias_fairness'] = self.bias_fairness_assessment(
                test_dataloader, 
                sensitive_attributes
            )
        
        return report
