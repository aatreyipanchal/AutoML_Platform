from .data_loader import DataLoader
from .eda import EDAEngine
from .preprocessing import PreprocessingEngine
from .model_selection import ModelSelector
from .training import Trainer
import os
import pandas as pd
import numpy as np
import joblib

class AutoMLPipeline:
    """
    The orchestrator that runs the entire AutoML flow.
    """
    def __init__(self, data_path: str, target_col: str = None, task_type: str = "tabular", sub_task: str = "classification"):
        self.data_path = data_path
        self.target_col = target_col
        self.task_type = task_type
        self.sub_task = sub_task
        
        self.loader = DataLoader()
        self.eda = EDAEngine()
        self.preprocessor = PreprocessingEngine()
        self.selector = ModelSelector(task_type=task_type, sub_task=sub_task)
        self.trainer = Trainer()

    def run(self):
        print(f"Starting AutoML Pipeline for {self.task_type} task...")
        
        # 1. Load Data
        if self.task_type == "tabular":
            df = self.loader.load_csv(self.data_path)
            
            # 2. EDA
            self.eda.analyze_tabular(df, target_col=self.target_col)
            
            # 3. Preprocessing
            X, y, feature_names = self.preprocessor.preprocess_tabular(df, self.target_col)
            
            # 4. Model Selection
            candidates = self.selector.get_tabular_candidates()
            
            # 5. Training
            best_model, results = self.trainer.train_tabular(X, y, candidates, self.sub_task)
            
            # 6. Save Preprocessor State for later inference
            if self.preprocessor.column_transformer:
                joblib.dump(self.preprocessor.column_transformer, os.path.join("src/models", "preprocessor.pkl"))
            
            print("AutoML Pipeline Finished Successfully!")
            return best_model, results

        elif self.task_type == "cv":
            print("Starting CV Pipeline...")
            # 1. Load Images and Labels (folder names are labels)
            images, labels, label_map = self.loader.load_images_with_labels(self.data_path)
            
            # 2. EDA
            self.eda.analyze_cv(images, labels=labels)
            
            # 3. Preprocessing
            X = self.preprocessor.preprocess_images(images)
            y = np.array(labels)
            
            # 4. Model Selection (Pre-trained ResNet)
            num_classes = len(np.unique(y))
            model = self.selector.get_cv_model(num_classes=num_classes)
            
            # 5. Training
            best_model = self.trainer.train_cv(model, X, y, epochs=5)
            
            # 6. Save label map for later inference
            joblib.dump(label_map, os.path.join("src/models", "cv_label_map.pkl"))
            
            print("CV Pipeline Finished Successfully!")
            return best_model, {"status": "completed", "num_classes": num_classes}
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
