import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.engine.pipeline import AutoMLPipeline
from src.engine.data_loader import DataLoader
from src.engine.eda import EDAEngine
import glob
import sys
import os

# Robust Path Handling
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

REPORT_DIR = os.path.join(ROOT_DIR, "src", "reports")
MODEL_DIR = os.path.join(ROOT_DIR, "src", "models")

# Ensure directories exist
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Page Config
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .report-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def st_download_file(file_path, label, file_name):
    """Helper to create a download button for a file if it exists."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label=label,
                data=f,
                file_name=file_name,
                mime="application/octet-stream"
            )
    else:
        st.warning(f"File {file_name} not found. Ensure the pipeline ran successfully.")

def main():
    st.title("🚀 AutoML Platform")
    st.markdown("### Accelerate your Machine Learning workflow with ease.")

    # Sidebar
    st.sidebar.title("Settings")
    task_type = st.sidebar.selectbox("Task Type", ["tabular", "cv"])
    
    if task_type == "tabular":
        sub_task = st.sidebar.selectbox("Sub-task", ["classification", "regression"])
    else:
        sub_task = "classification" # Default for CV

    # Main Content
    if task_type == "tabular":
        st.header("📂 Data Upload")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            st.write("### Dataset Info")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.index.size)
            col2.metric("Columns", df.columns.size)
            col3.metric("Missing Values", df.isna().sum().sum())

            target_col = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns)-1)
            
            # Save uploaded file temporarily for the pipeline
            temp_path = os.path.join(ROOT_DIR, "tmp_uploaded_data.csv")
            df.to_csv(temp_path, index=False)

            # EDA Section
            st.divider()
            st.header("📊 Exploratory Data Analysis")
            if st.button("Run Automated EDA"):
                try:
                    with st.spinner("Analyzing data..."):
                        eda_engine = EDAEngine(report_dir=REPORT_DIR)
                        report_dir = eda_engine.analyze_tabular(df, target_col=target_col)
                        
                        st.success("EDA Completed!")
                        
                        # Display Plots
                        st.subheader("Visualizations")
                        
                        # Correlation Heatmap
                        heatmap_path = os.path.join(REPORT_DIR, "correlation_heatmap.png")
                        if os.path.exists(heatmap_path):
                            st.image(heatmap_path, caption="Correlation Heatmap", use_column_width=True)
                        
                        # Target Distribution
                        dist_path = os.path.join(REPORT_DIR, f"{target_col}_distribution.png")
                        if os.path.exists(dist_path):
                            st.image(dist_path, caption=f"Distribution of {target_col}", use_column_width=True)
                        
                        # Boxplots
                        st.write("### Boxplots (Outlier Detection)")
                        boxplots = glob.glob(os.path.join(REPORT_DIR, "boxplot_*.png"))
                        if boxplots:
                            cols = st.columns(2)
                            for i, bp in enumerate(boxplots):
                                cols[i % 2].image(bp, use_column_width=True)
                except Exception as e:
                    st.error("EDA Failed!")
                    st.exception(e)

            # AutoML Section
            st.divider()
            st.header("⚙️ AutoML Pipeline")
            if st.button("Run AutoML"):
                try:
                    with st.spinner("Running AutoML Pipeline... This may take a minute."):
                        pipeline = AutoMLPipeline(
                            data_path=temp_path,
                            target_col=target_col,
                            task_type=task_type,
                            sub_task=sub_task,
                            model_dir=MODEL_DIR,
                            report_dir=REPORT_DIR
                        )
                        best_model, results = pipeline.run()
                        
                        st.success("AutoML Pipeline Finished!")
                        
                        st.write("### Best Model Performance")
                        st.json(results)
                        
                        # Assuming results contains a 'metrics' key or similar
                        # Let's show it nicely if it's there
                        if isinstance(results, dict):
                            st.write("#### Detailed Metrics")
                            metrics_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
                            st.table(metrics_df)

                        # Download Buttons
                        st.divider()
                        st.write("### 📥 Download Trained Assets")
                        st.info("Download the trained model and preprocessor for later use in production.")
                        col1, col2 = st.columns(2)
                        with col1:
                            st_download_file(os.path.join(MODEL_DIR, "best_tabular_model.pkl"), "Model (.pkl)", "best_tabular_model.pkl")
                        with col2:
                            st_download_file(os.path.join(MODEL_DIR, "preprocessor.pkl"), "Preprocessor (.pkl)", "preprocessor.pkl")

                except Exception as e:
                    st.error("AutoML Pipeline Failed!")
                    st.exception(e)

    elif task_type == "cv":
        st.header("🖼️ Computer Vision")
        st.info("CV tasks require a directory structure where each folder name is a label.")
        dir_path = st.text_input("Enter path to image dataset directory", placeholder="e.g., d:/data/images")
        
        if dir_path and os.path.isdir(dir_path):
            st.success(f"Directory located: {dir_path}")
            
            if st.button("Run CV AutoML"):
                try:
                    with st.spinner("Running CV Pipeline..."):
                        pipeline = AutoMLPipeline(
                            data_path=dir_path,
                            task_type="cv",
                            model_dir=MODEL_DIR,
                            report_dir=REPORT_DIR
                        )
                        best_model, results = pipeline.run()
                        st.success("CV AutoML Finished!")
                        st.json(results)
                        
                        # Show sample images if EDA was run
                        sample_images_path = os.path.join(REPORT_DIR, "sample_images.png")
                        if os.path.exists(sample_images_path):
                            st.image(sample_images_path, caption="Sample Dataset Images")

                        # Download Buttons
                        st.divider()
                        st.write("### 📥 Download Trained Assets")
                        st.info("Download the trained CV model and label mapping.")
                        col1, col2 = st.columns(2)
                        with col1:
                            st_download_file(os.path.join(MODEL_DIR, "best_cv_model.pth"), "CV Model (.pth)", "best_cv_model.pth")
                        with col2:
                            st_download_file(os.path.join(MODEL_DIR, "cv_label_map.pkl"), "Label Map (.pkl)", "cv_label_map.pkl")

                except Exception as e:
                    st.error("CV AutoML Failed!")
                    st.exception(e)
        elif dir_path:
            st.error("Invalid directory path.")

if __name__ == "__main__":
    main()
