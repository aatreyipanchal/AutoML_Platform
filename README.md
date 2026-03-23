# 🤖 AutoML Platform

An end-to-end Automated Machine Learning platform built with Python and Streamlit. This platform allows users to perform automated Exploratory Data Analysis (EDA) and train high-performance models for both Tabular and Computer Vision tasks with zero coding.

## 🚀 Features

### 📊 Tabular AutoML
- **Data Upload**: Support for CSV files with instant data preview.
- **Automated EDA**: Generate correlation heatmaps, target distributions, and outlier boxplots automatically.
- **Model Training**: Automatically preprocess data, select model candidates, and train the best model for Classification or Regression.
- **Model Downloads**: Download the trained model (`.pkl`) and preprocessor (`.pkl`) for production use.

### 🖼️ Computer Vision AutoML
- **Image Classification**: Train a PyTorch-based image classification model by providing a directory of images organized by folders (labels).
- **CV EDA**: Visualize sample images and class distributions.
- **Model Downloads**: Download the trained model (`.pth`) and label mappings (`.pkl`).

### 🎨 Premium UI
- Clean and modern Streamlit interface with a responsive sidebar for configuration.
- Real-time progress tracking with spinners and status updates.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd AutoML_Platform
    ```

2.  **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

To start the Streamlit application, run:

```bash
streamlit run streamlit_app.py
```

Open your browser and navigate to `http://localhost:8501`.

## 📂 Project Structure

- `src/engine/`: Core AutoML logic (Data loading, EDA, Preprocessing, Training, etc.)
- `src/models/`: Directory where trained models and preprocessors are saved.
- `src/reports/`: Directory where EDA plots and summaries are generated.
- `streamlit_app.py`: Main Streamlit application file.
- `requirements.txt`: Python dependencies.
- `dummy_data.csv`: A sample dataset for testing tabular AutoML.

## ⚖️ License

MIT License. See `LICENSE` for details.
