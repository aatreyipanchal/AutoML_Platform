import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

class EDAEngine:
    """
    Automated EDA and visual report generator.
    """
    def __init__(self, report_dir: str = "src/reports"):
        self.report_dir = report_dir
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def analyze_tabular(self, df: pd.DataFrame, target_col: str = None) -> str:
        """Performs EDA on tabular data and saves plots."""
        print("Generating Tabular EDA...")
        
        # 1. Summary statistics
        summary = df.describe(include='all')
        summary.to_csv(os.path.join(self.report_dir, "summary.csv"))
        
        # 2. Correlation Heatmap (numeric only)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            plt.figure(figsize=(12, 10))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.savefig(os.path.join(self.report_dir, "correlation_heatmap.png"))
            plt.close()

        # 3. Distribution of target variable
        if target_col and target_col in df.columns:
            plt.figure(figsize=(8, 6))
            if df[target_col].dtype in [np.number]:
                sns.histplot(df[target_col], kde=True)
            else:
                sns.countplot(x=target_col, data=df)
            plt.title(f"Distribution of {target_col}")
            plt.savefig(os.path.join(self.report_dir, f"{target_col}_distribution.png"))
            plt.close()

        # 4. Outliers check (box plots)
        for col in numeric_df.columns:
            if col != target_col:
                plt.figure(figsize=(8, 4))
                sns.boxplot(x=df[col])
                plt.title(f"Boxplot for {col}")
                plt.savefig(os.path.join(self.report_dir, f"boxplot_{col}.png"))
                plt.close()

        return self.report_dir

    def analyze_cv(self, images: list, labels: list = None) -> str:
        """Visualizes sample images and label distributions for CV."""
        print("Generating CV EDA...")
        
        # Visualize first 9 images
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                img = images[i]
                ax.imshow(img)
                if labels:
                    ax.set_title(f"Label: {labels[i]}")
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, "sample_images.png"))
        plt.close()

        if labels:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=labels)
            plt.title("Class Distribution")
            plt.savefig(os.path.join(self.report_dir, "cv_class_distribution.png"))
            plt.close()

        return self.report_dir
