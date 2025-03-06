import os
import sys
import PIL.Image
from typing import Dict, List

def scan_wound_classification_dataset(base_path: str = 'data/images/wound_classification/') -> Dict[str, Dict]:
    """
    Catalog and validate the wound classification dataset.
    
    Args:
        base_path (str): Base path to the wound classification images
    
    Returns:
        Dict containing dataset inventory and validation results
    """
    # Use absolute path from current working directory
    full_base_path = os.path.join(os.getcwd(), base_path)
    
    # Print current working directory and base path for debugging
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Base Path: {full_base_path}")
    print(f"Base Path Exists: {os.path.exists(full_base_path)}")
    
    # List contents of the base path
    print("Base Path Contents:")
    try:
        print(os.listdir(full_base_path))
    except Exception as e:
        print(f"Error listing base path contents: {e}")

    # Verify base path exists
    if not os.path.exists(full_base_path):
        print(f"Error: Base path {full_base_path} does not exist.")
        sys.exit(1)

    dataset_report = {
        'total_wound_types': 0,
        'wound_types': {},
        'validation_results': {
            'min_images_per_category': float('inf'),
            'max_images_per_category': 0,
            'resolution_check': {
                'passed': True,
                'details': []
            }
        }
    }
    
    # Scan wound type directories
    for wound_type in os.listdir(full_base_path):
        wound_type_path = os.path.join(full_base_path, wound_type)
        
        # Skip if not a directory
        if not os.path.isdir(wound_type_path):
            continue
        
        # Get image files
        image_files = [f for f in os.listdir(wound_type_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Skip empty directories
        if not image_files:
            continue
        
        # Update wound type statistics
        dataset_report['wound_types'][wound_type] = {
            'total_images': len(image_files),
            'image_files': image_files,
            'resolution_details': []
        }
        
        # Update min/max images per category
        dataset_report['validation_results']['min_images_per_category'] = min(
            dataset_report['validation_results']['min_images_per_category'], 
            len(image_files)
        )
        dataset_report['validation_results']['max_images_per_category'] = max(
            dataset_report['validation_results']['max_images_per_category'], 
            len(image_files)
        )
        
        # Check image resolutions
        for image_file in image_files:
            image_path = os.path.join(wound_type_path, image_file)
            try:
                with PIL.Image.open(image_path) as img:
                    width, height = img.size
                    resolution_detail = {
                        'filename': image_file,
                        'width': width,
                        'height': height,
                        'meets_requirement': width >= 224 and height >= 224
                    }
                    
                    dataset_report['wound_types'][wound_type]['resolution_details'].append(resolution_detail)
                    
                    if not resolution_detail['meets_requirement']:
                        dataset_report['validation_results']['resolution_check']['passed'] = False
                        dataset_report['validation_results']['resolution_check']['details'].append(resolution_detail)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        dataset_report['total_wound_types'] += 1
    
    return dataset_report

def generate_dataset_report(dataset_report: Dict) -> str:
    """
    Generate a human-readable report from the dataset inventory.
    
    Args:
        dataset_report (Dict): Dataset inventory dictionary
    
    Returns:
        str: Formatted report
    """
    report = "Wound Classification Dataset Inventory Report\n"
    report += "=" * 50 + "\n\n"
    
    report += f"Total Wound Types: {dataset_report['total_wound_types']}\n\n"
    
    report += "Wound Type Statistics:\n"
    for wound_type, details in dataset_report['wound_types'].items():
        report += f"- {wound_type}: {details['total_images']} images\n"
    
    report += "\nValidation Results:\n"
    report += f"- Minimum images per category: {dataset_report['validation_results']['min_images_per_category']}\n"
    report += f"- Maximum images per category: {dataset_report['validation_results']['max_images_per_category']}\n"
    
    resolution_check = dataset_report['validation_results']['resolution_check']
    report += f"- Resolution Check: {'PASSED' if resolution_check['passed'] else 'FAILED'}\n"
    
    if not resolution_check['passed']:
        report += "  Problematic Images:\n"
        for detail in resolution_check['details']:
            report += f"  - {detail['filename']}: {detail['width']}x{detail['height']}\n"
    
    return report

def main():
    dataset_report = scan_wound_classification_dataset()
    report_text = generate_dataset_report(dataset_report)
    
    print(report_text)

if __name__ == '__main__':
    main()
