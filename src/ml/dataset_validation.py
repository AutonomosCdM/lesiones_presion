import os
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

class DatasetValidator:
    def __init__(self, 
                 original_dir: str = 'data/images/wound_classification/',
                 processed_dir: str = 'data/processed_images/'):
        """
        Initialize dataset validator
        
        Args:
            original_dir (str): Path to original dataset
            processed_dir (str): Path to processed dataset
        """
        self.original_dir = original_dir
        self.processed_dir = processed_dir
        
    def validate_image_transformations(self) -> Dict[str, Dict]:
        """
        Validate image transformations across wound types
        
        Returns:
            Dict containing validation metrics for each wound type
        """
        validation_results = {}
        
        for wound_type in os.listdir(self.processed_dir):
            if not os.path.isdir(os.path.join(self.processed_dir, wound_type)):
                continue
            
            # Skip the preprocessing report
            if wound_type == 'preprocessing_report.json':
                continue
            
            original_images = [f for f in os.listdir(os.path.join(self.original_dir, wound_type)) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            processed_images = [f for f in os.listdir(os.path.join(self.processed_dir, wound_type)) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Metrics calculation
            type_metrics = {
                'original_count': len(original_images),
                'processed_count': len(processed_images),
                'augmented_count': len([f for f in processed_images if f.startswith('augmented_')]),
                'size_consistency': [],
                'pixel_range_check': []
            }
            
            # Sample images for detailed checks
            sample_size = min(10, len(processed_images))
            sample_images = np.random.choice(processed_images, sample_size, replace=False)
            
            for image_file in sample_images:
                image_path = os.path.join(self.processed_dir, wound_type, image_file)
                
                # Read image
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Size check
                height, width = image.shape[:2]
                type_metrics['size_consistency'].append((width, height))
                
                # Pixel range check
                pixel_min = float(image_rgb.min())
                pixel_max = float(image_rgb.max())
                type_metrics['pixel_range_check'].append((pixel_min, pixel_max))
            
            # Aggregate size consistency
            sizes = type_metrics['size_consistency']
            type_metrics['size_stats'] = {
                'unique_sizes': len(set(sizes)),
                'target_size_met': all(size == (224, 224) for size in sizes)
            }
            
            # Pixel range analysis
            pixel_ranges = type_metrics['pixel_range_check']
            type_metrics['pixel_range_stats'] = {
                'min_range': min(r[0] for r in pixel_ranges),
                'max_range': max(r[1] for r in pixel_ranges),
                'normalized_range_check': all(0 <= r[0] and r[1] <= 1 for r in pixel_ranges)
            }
            
            validation_results[wound_type] = type_metrics
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict) -> Dict:
        """
        Generate a comprehensive validation report
        
        Args:
            validation_results (Dict): Validation metrics
        
        Returns:
            Dict containing overall validation assessment
        """
        overall_report = {
            'total_wound_types': len(validation_results),
            'wound_type_metrics': {},
            'overall_assessment': {
                'size_consistency': True,
                'pixel_normalization': True,
                'augmentation_coverage': True
            }
        }
        
        for wound_type, metrics in validation_results.items():
            type_assessment = {
                'original_count': metrics['original_count'],
                'processed_count': metrics['processed_count'],
                'augmented_count': metrics['augmented_count'],
                'size_consistent': metrics['size_stats']['target_size_met'],
                'pixel_normalized': metrics['pixel_range_stats']['normalized_range_check']
            }
            
            overall_report['wound_type_metrics'][wound_type] = type_assessment
            
            # Update overall assessment
            overall_report['overall_assessment']['size_consistency'] &= type_assessment['size_consistent']
            overall_report['overall_assessment']['pixel_normalization'] &= type_assessment['pixel_normalized']
            overall_report['overall_assessment']['augmentation_coverage'] &= (metrics['augmented_count'] > 0)
        
        return overall_report
    
    def run_validation(self) -> None:
        """
        Run full dataset validation and generate report
        """
        # Validate transformations
        validation_results = self.validate_image_transformations()
        
        # Generate comprehensive report
        validation_report = self.generate_validation_report(validation_results)
        
        # Save detailed validation results
        with open(os.path.join(self.processed_dir, 'dataset_validation_results.json'), 'w') as f:
            json.dump(validation_results, f, indent=2, cls=NumpyEncoder)
        
        # Save overall validation report
        with open(os.path.join(self.processed_dir, 'dataset_validation_report.json'), 'w') as f:
            json.dump(validation_report, f, indent=2, cls=NumpyEncoder)
        
        # Print summary
        print("Dataset Validation Complete")
        print(f"Total Wound Types Processed: {validation_report['total_wound_types']}")
        print("Overall Assessment:")
        for key, value in validation_report['overall_assessment'].items():
            print(f"- {key.replace('_', ' ').title()}: {'PASSED' if value else 'FAILED'}")

def main():
    validator = DatasetValidator()
    validator.run_validation()

if __name__ == '__main__':
    main()
