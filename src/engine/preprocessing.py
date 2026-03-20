import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List

class PreprocessingEngine:
    """
    Automated preprocessing for tabular and image data.
    """
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.column_transformer = None

    def preprocess_tabular(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Cleans and transforms tabular data."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Define transformations
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        X_processed = self.column_transformer.fit_transform(X)
        
        # Handle target encoding if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        feature_names = numeric_cols + list(self.column_transformer.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)) if categorical_cols else numeric_cols
        
        return X_processed, np.array(y), feature_names

    def preprocess_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Normalizes images (rescaling 0-255 to 0-1)."""
        return np.array(images) / 255.0
