from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import torch
import torch.nn as nn
import torchvision.models as models

class ModelSelector:
    """
    Selects best candidate models based on task and data.
    """
    def __init__(self, task_type: str = "tabular", sub_task: str = "classification"):
        self.task_type = task_type
        self.sub_task = sub_task

    def get_tabular_candidates(self) -> list:
        """Returns a list of candidate models for tabular data."""
        if self.sub_task == "classification":
            return [
                LogisticRegression(max_iter=1000),
                RandomForestClassifier(n_estimators=100),
                XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                LGBMClassifier()
            ]
        else:
            return [
                LinearRegression(),
                RandomForestRegressor(n_estimators=100),
                XGBRegressor(),
                LGBMRegressor()
            ]

    def get_cv_model(self, num_classes: int) -> nn.Module:
        """Returns a pre-trained CNN model for image classification."""
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
