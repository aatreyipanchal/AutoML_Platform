import pandas as pd
import numpy as np
import os
from PIL import Image
from typing import Union, List, Tuple, Dict

class DataLoader:
    """
    Handles loading of CSV and Image datasets.
    """
    def __init__(self):
        pass

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Loads a CSV file into a pandas DataFrame."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at {file_path}")
        return pd.read_csv(file_path)

    def load_images(self, directory_path: str, target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """Loads images from a directory and resizes them."""
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found at {directory_path}")
        
        images = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(directory_path, filename)
                try:
                    with Image.open(img_path) as img:
                        img = img.resize(target_size).convert('RGB')
                        images.append(np.array(img))
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
        return images

    def load_images_with_labels(self, directory_path: str, target_size: Tuple[int, int] = (224, 224)) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
        """
        Loads images from subdirectories, where each subdirectory is a class/label.
        Returns:
            - List of image arrays
            - List of integer labels
            - Dictionary mapping integer labels to folder names
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found at {directory_path}")
        
        images = []
        labels = []
        label_map = {}
        
        # Identify subdirectories as classes
        classes = sorted([d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))])
        
        for idx, cls_name in enumerate(classes):
            label_map[idx] = cls_name
            cls_dir = os.path.join(directory_path, cls_name)
            
            for filename in os.listdir(cls_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, filename)
                    try:
                        with Image.open(img_path) as img:
                            img = img.resize(target_size).convert('RGB')
                            images.append(np.array(img))
                            labels.append(idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        return images, labels, label_map

    def detect_task_type(self, data: Union[pd.DataFrame, List[np.ndarray]]) -> str:
        """Detects if the task is tabular (CSV) or computer vision (Images)."""
        if isinstance(data, pd.DataFrame):
            return "tabular"
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray):
            return "cv"
        else:
            return "unknown"
