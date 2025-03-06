import os
import numpy as np
import cv2
from typing import List, Dict, Any
import json

class WoundImagePreprocessor:
    def __init__(self, 
                 input_dir: str = 'data/images/wound_classification/', 
                 output_dir: str = 'data/processed_images/'):
        """
        Initialize wound image preprocessing pipeline
        
        Args:
            input_dir (str): Source directory for wound images
            output_dir (str): Destination directory for processed images
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_augmentation_counts(self, wound_types: Dict[str, int]) -> Dict[str, int]:
        """
        Calculate augmentation counts based on current dataset distribution
        
        Args:
            wound_types (Dict): Dictionary with wound types and their image counts
        
        Returns:
            Dict with augmentation counts per wound type
        """
        min_images = min(wound_types.values())
        target_images = int(min_images * 1.3)  # 30% increase
        
        augmentation_counts = {}
        for wound_type, count in wound_types.items():
            augmentation_counts[wound_type] = max(0, target_images - count)
        
        return augmentation_counts
    
    def preprocess_image(self, image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image_path (str): Path to the image
            target_size (tuple): Target size for resizing
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply augmentations to an image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        # Horizontal flip
        flip_h = cv2.flip(image, 1)
        augmented_images.append(flip_h)
        
        # Vertical flip
        flip_v = cv2.flip(image, 0)
        augmented_images.append(flip_v)
        
        # Rotation (90 degrees)
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        augmented_images.append(rotated)
        
        # Brightness adjustment
        brightness = np.clip(image * 1.2, 0, 1)
        augmented_images.append(brightness)
        
        # Contrast adjustment
        contrast = np.clip((image - 0.5) * 1.5 + 0.5, 0, 1)
        augmented_images.append(contrast)
        
        return augmented_images
    
    def preprocess_and_augment(self):
        """
        Preprocess and augment wound images
        """
        # Get wound types from directory structure
        wound_types = {}
        for wound_type in os.listdir(self.input_dir):
            wound_type_path = os.path.join(self.input_dir, wound_type)
            if os.path.isdir(wound_type_path):
                # Count images in directory
                image_files = [f for f in os.listdir(wound_type_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                wound_types[wound_type] = len(image_files)
                print(f"Found {len(image_files)} images in {wound_type}")
        
        # Calculate augmentation counts
        augmentation_counts = self.calculate_augmentation_counts(wound_types)
        
        # Process each wound type
        for wound_type, augment_count in augmentation_counts.items():
            wound_type_path = os.path.join(self.input_dir, wound_type)
            output_type_path = os.path.join(self.output_dir, wound_type)
            os.makedirs(output_type_path, exist_ok=True)
            
            # Get original images
            original_images = [f for f in os.listdir(wound_type_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Process original images
            for image_file in original_images:
                image_path = os.path.join(wound_type_path, image_file)
                
                try:
                    # Preprocess image
                    processed_image = self.preprocess_image(image_path)
                    
                    # Save processed image
                    output_path = os.path.join(output_type_path, f"processed_{image_file}")
                    cv2.imwrite(output_path, cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            
            # Augment images if needed
            if augment_count > 0:
                # Select random images for augmentation
                images_to_augment = np.random.choice(original_images, min(augment_count, len(original_images)), replace=True)
                
                for i, image_file in enumerate(images_to_augment):
                    if i >= augment_count:
                        break
                    
                    image_path = os.path.join(wound_type_path, image_file)
                    
                    try:
                        # Preprocess image
                        processed_image = self.preprocess_image(image_path)
                        
                        # Apply augmentations
                        augmented_images = self.augment_image(processed_image)
                        
                        # Save augmented images
                        for j, aug_image in enumerate(augmented_images):
                            if i * len(augmented_images) + j >= augment_count:
                                break
                            
                            output_filename = f"augmented_{i}_{j}_{image_file}"
                            output_path = os.path.join(output_type_path, output_filename)
                            cv2.imwrite(output_path, cv2.cvtColor((aug_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"Error augmenting {image_path}: {e}")
        
        # Generate report
        report = {
            "original_counts": wound_types,
            "augmentation_counts": augmentation_counts,
            "total_counts": {wt: wound_types[wt] + augmentation_counts[wt] for wt in wound_types}
        }
        
        # Save report
        with open(os.path.join(self.output_dir, 'preprocessing_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Preprocessing and augmentation complete.")
        print(f"Report saved to {os.path.join(self.output_dir, 'preprocessing_report.json')}")

def main():
    preprocessor = WoundImagePreprocessor()
    preprocessor.preprocess_and_augment()

if __name__ == '__main__':
    main()
