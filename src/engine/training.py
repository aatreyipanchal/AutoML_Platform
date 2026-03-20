import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

class Trainer:
    """
    Orchestrates training and evaluation.
    """
    def __init__(self, model_dir: str = "src/models"):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train_tabular(self, X: np.ndarray, y: np.ndarray, models: list, sub_task: str):
        """Trains multiple models and picks the best one."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_model = None
        best_score = -np.inf if sub_task == "classification" else np.inf
        results = {}

        for model in models:
            model_name = model.__class__.__name__
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            if sub_task == "classification":
                score = f1_score(y_test, y_pred, average='weighted')
                print(f"{model_name} F1 Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model = model
            else:
                score = mean_squared_error(y_test, y_pred)
                print(f"{model_name} MSE: {score:.4f}")
                if score < best_score:
                    best_score = score
                    best_model = model
            
            results[model_name] = score

        # Save best model
        import joblib
        joblib.dump(best_model, os.path.join(self.model_dir, "best_tabular_model.pkl"))
        print(f"Best model {best_model.__class__.__name__} saved.")
        return best_model, results

    def train_cv(self, model: nn.Module, images: np.ndarray, labels: np.ndarray, epochs: int = 5):
        """Trains a PyTorch model on image data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Simple data loader
        images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        print(f"Starting Training on {device}...")
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        torch.save(model.state_dict(), os.path.join(self.model_dir, "best_cv_model.pth"))
        print("CV Model saved.")
        return model
