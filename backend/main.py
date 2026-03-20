from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import shutil
import uuid
from typing import Dict
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.engine.pipeline import AutoMLPipeline
import pandas as pd
import joblib
import zipfile
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np

app = FastAPI(title="AutoML Platform API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "backend/uploads"
RESULTS_DIR = "src/reports" # Points to engine's reports
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app.mount("/reports", StaticFiles(directory=RESULTS_DIR), name="reports")

# In-memory task status (for demo/simplicity)
tasks: Dict[str, Dict] = {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # If ZIP, extract it
    if file_extension == ".zip":
        extract_dir = os.path.join(UPLOAD_DIR, f"{file_id}_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return {"file_id": file_id, "filename": file.filename, "file_path": extract_dir, "is_zip": True}
        
    return {"file_id": file_id, "filename": file.filename, "file_path": file_path, "is_zip": False}

@app.get("/columns/{file_id}")
async def get_columns(file_id: str):
    # Find file by ID in upload dir
    for filename in os.listdir(UPLOAD_DIR):
        if filename.startswith(file_id):
            file_path = os.path.join(UPLOAD_DIR, filename)
            
            # If it's the extracted directory
            if os.path.isdir(file_path) and filename.endswith("_extracted"):
                # Return subfolders as labels
                labels = [d for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]
                return {"columns": labels, "type": "cv"}
            
            # If it's a CSV file
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path, nrows=0)
                return {"columns": df.columns.tolist(), "type": "tabular"}
                
    return {"error": "File not found"}

def run_automl_task(task_id: str, data_path: str, target: str, type: str, task: str):
    tasks[task_id]["status"] = "running"
    try:
        pipeline = AutoMLPipeline(
            data_path=data_path,
            target_col=target,
            task_type=type,
            sub_task=task
        )
        pipeline.run()
        tasks[task_id]["status"] = "completed"
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

@app.post("/run-pipeline")
async def start_pipeline(
    file_path: str, 
    target: str, 
    type: str = "tabular", 
    task: str = "classification",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "type": type}
    background_tasks.add_task(run_automl_task, task_id, file_path, target, type, task)
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        return {"error": "Task not found"}
    
    # If completed, add list of reports
    if tasks[task_id]["status"] == "completed":
        reports = [f for f in os.listdir(RESULTS_DIR) if os.path.isfile(os.path.join(RESULTS_DIR, f))]
        tasks[task_id]["reports"] = reports
        
    return tasks[task_id]

@app.get("/download/{filename}")
async def download_file(filename: str):
    # Search in models or reports
    for folder in ["src/models", "src/reports"]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return FileResponse(path, filename=filename)
    return {"error": "File not found"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: dict):
    task_type = data.get("type", "tabular")
    
    if task_type == "tabular":
        model_path = "src/models/best_tabular_model.pkl"
        prep_path = "src/models/preprocessor.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(prep_path):
            return {"error": "Tabular model not found"}
        
        try:
            model = joblib.load(model_path)
            preprocessor = joblib.load(prep_path)
            input_df = pd.DataFrame([data["input"]])
            X_processed = preprocessor.transform(input_df)
            prediction = model.predict(X_processed)
            return {"prediction": str(prediction[0])}
        except Exception as e:
            return {"error": f"Tabular prediction failed: {str(e)}"}
            
    elif task_type == "cv":
        model_path = "src/models/best_cv_model.pth"
        label_map_path = "src/models/cv_label_map.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(label_map_path):
            return {"error": "CV model not found"}
            
        try:
            # Load metadata
            label_map = joblib.load(label_map_path)
            num_classes = len(label_map)
            
            # Reconstruct model (ResNet18 as used in selector)
            from src.engine.model_selection import ModelSelector
            selector = ModelSelector(task_type="cv")
            model = selector.get_cv_model(num_classes=num_classes)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
            # Process image from base64
            img_data = base64.b64decode(data["image"].split(",")[-1])
            img = Image.open(BytesIO(img_data)).convert('RGB').resize((224, 224))
            img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                class_idx = predicted.item()
                return {"prediction": label_map[class_idx]}
        except Exception as e:
            return {"error": f"CV prediction failed: {str(e)}"}
            
    return {"error": "Invalid task type"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
