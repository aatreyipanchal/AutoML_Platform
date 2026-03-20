import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Play, CheckCircle, BarChart3, Database, Layers, ArrowRight, Settings, Image as ImageIcon, Send, Download, ExternalLink } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = "http://localhost:8001";

function App() {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [columns, setColumns] = useState([]);
  const [target, setTarget] = useState("");
  const [taskType, setTaskType] = useState("tabular");
  const [subTask, setSubTask] = useState("classification");
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState("idle");
  const [reports, setReports] = useState([]);
  const [predictionData, setPredictionData] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);

  const handleFileUpload = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      setFileData(res.data);
      
      const colRes = await axios.get(`${API_BASE}/columns/${res.data.file_id}`);
      setColumns(colRes.data.columns);
      setTaskType(colRes.data.type || "tabular");
      
      // Auto-set target for CV (first class is fine as a placeholder, though not strictly needed for pipeline)
      if (colRes.data.type === "cv") {
        setTarget("inferred_labels");
        setSubTask("classification");
      }
      
      setStep(2);
    } catch (err) {
      console.error(err);
      alert("Upload failed");
    }
  };

  const startPipeline = async () => {
    if (!target) return;
    setStep(3);
    setStatus("running");

    try {
      const res = await axios.post(`${API_BASE}/run-pipeline`, null, {
        params: {
          file_path: fileData.file_path,
          target: target,
          type: taskType,
          task: subTask
        }
      });
      setTaskId(res.data.task_id);
      startPolling(res.data.task_id);
    } catch (err) {
      console.error(err);
      setStatus("failed");
    }
  };

  const startPolling = (id) => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/status/${id}`);
        if (res.data.status === "completed") {
          setStatus("completed");
          setReports(res.data.reports || []);
          clearInterval(interval);
        } else if (res.data.status === "failed") {
          setStatus("failed");
          clearInterval(interval);
        }
      } catch (err) {
        clearInterval(interval);
      }
    }, 2000);
  };

  const handlePredict = async () => {
    try {
      const payload = taskType === "tabular" 
        ? { type: "tabular", input: predictionData } 
        : { type: "cv", image: predictionData.image };
        
      const res = await axios.post(`${API_BASE}/predict`, payload);
      setPredictionResult(res.data.prediction);
    } catch (err) {
      console.error(err);
    }
  };

  const handleImageInference = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      setPredictionData({ image: reader.result });
      setPreviewImage(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const downloadAsset = (filename) => {
    window.open(`${API_BASE}/download/${filename}`);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-100 selection:text-blue-900">
      <nav className="border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-blue-600 rounded-xl flex items-center justify-center shadow-sm shadow-blue-200">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-lg tracking-tight text-slate-900">AutoML <span className="text-blue-600">Platform</span></span>
          </div>
          <div className="flex items-center gap-1">
            <NavButton active={step === 1} onClick={() => setStep(1)} label="Data" />
            <NavButton active={step === 2} onClick={() => step > 1 && setStep(2)} label="Config" />
            <NavButton active={step === 3} onClick={() => step > 2 && setStep(3)} label="Results" />
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-6 py-16">
        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div 
              key="step1"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center text-center space-y-12 max-w-3xl mx-auto"
            >
              <div className="space-y-4">
                <h1 className="text-5xl md:text-6xl font-bold text-slate-900 tracking-tight">
                  Intelligent <span className="text-blue-600">AutoML</span>.
                </h1>
                <p className="text-slate-500 text-lg md:text-xl leading-relaxed">
                  Streamline your machine learning workflow. Upload your dataset and let our engine handle preprocessing, EDA, and model optimization.
                </p>
              </div>

              <div className="w-full group">
                <div className="relative bg-white border-2 border-dashed border-slate-200 rounded-3xl p-16 flex flex-col items-center gap-6 cursor-pointer hover:border-blue-400 hover:bg-blue-50/30 transition-all duration-300 group">
                  <input type="file" id="file" className="hidden" onChange={handleFileUpload} accept=".csv,.zip" />
                  <label htmlFor="file" className="flex flex-col items-center gap-5 cursor-pointer w-full">
                    <div className="w-20 h-20 bg-slate-50 rounded-2xl flex items-center justify-center group-hover:bg-blue-100 transition-colors duration-300">
                      <Upload className="w-8 h-8 text-slate-400 group-hover:text-blue-600 transition-colors" />
                    </div>
                    <div className="space-y-1">
                      <span className="text-2xl font-semibold text-slate-900">Choose CSV or ZIP</span>
                      <p className="text-slate-400 text-sm">Drag and drop your dataset here</p>
                    </div>
                  </label>
                </div>
              </div>
            </motion.div>
          )}

          {step === 2 && (
            <motion.div 
              key="step2"
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              className="grid grid-cols-1 lg:grid-cols-12 gap-10"
            >
              <div className="lg:col-span-8 space-y-8">
                <div className="card-minimal p-10 space-y-10">
                  <div className="flex items-center gap-4">
                    <div className="p-2.5 bg-blue-50 rounded-xl"><Settings className="w-6 h-6 text-blue-600" /></div>
                    <div className="space-y-1">
                      <h2 className="text-2xl font-bold text-slate-900">Configuration</h2>
                      <p className="text-sm text-slate-500">Fine-tune your model parameters</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                    <div className="space-y-4">
                      <label className="text-xs font-bold text-slate-400 uppercase tracking-widest">{taskType === "cv" ? "Detected Classes" : "Target Variable"}</label>
                      {taskType === "cv" ? (
                        <div className="flex flex-wrap gap-2">
                          {columns.map(cls => (
                            <span key={cls} className="px-3 py-1.5 bg-blue-50 text-blue-600 rounded-lg text-xs font-bold border border-blue-100">{cls}</span>
                          ))}
                        </div>
                      ) : (
                        <select 
                          className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3.5 outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-slate-700 transition-all font-medium cursor-pointer"
                          value={target}
                          onChange={(e) => setTarget(e.target.value)}
                        >
                          <option value="">Select column...</option>
                          {columns.map(col => <option key={col} value={col}>{col}</option>)}
                        </select>
                      )}
                    </div>
                    <div className="space-y-4">
                      <label className="text-xs font-bold text-slate-400 uppercase tracking-widest">Model Objective</label>
                      <div className="flex p-1 bg-slate-100/50 rounded-xl border border-slate-200">
                        <button 
                          onClick={() => setSubTask("classification")} 
                          className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all ${subTask === "classification" ? "bg-white text-blue-600 shadow-sm border border-slate-200" : "text-slate-500 hover:text-slate-900"}`}
                        >Classification</button>
                        <button 
                          onClick={() => setSubTask("regression")} 
                          className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all ${subTask === "regression" ? "bg-white text-blue-600 shadow-sm border border-slate-200" : "text-slate-500 hover:text-slate-900"}`}
                        >Regression</button>
                      </div>
                    </div>
                  </div>

                  <div className="pt-4">
                    <button 
                      onClick={startPipeline}
                      disabled={!target}
                      className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-100 disabled:text-slate-400 disabled:cursor-not-allowed text-white px-8 py-4.5 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-3 shadow-lg shadow-blue-200"
                    >
                      <Play className="w-5 h-5 fill-current" />
                      Start AutoML Engine
                    </button>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-4 space-y-6">
                 <div className="card-minimal p-8 space-y-6">
                    <h3 className="font-bold text-slate-400 flex items-center gap-2 uppercase tracking-widest text-[10px]">
                      <Database className="w-3.5 h-3.5" /> Data Profile
                    </h3>
                    <div className="space-y-5">
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-slate-500">File Name</span>
                        <span className="font-medium text-slate-900 truncate max-w-[150px]">{file?.name}</span>
                      </div>
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-slate-500">Variables</span>
                        <span className="text-slate-900 font-semibold">{columns.length} Columns</span>
                      </div>
                    </div>
                 </div>
                 <div className="p-8 bg-blue-50 border border-blue-100 rounded-2xl space-y-4">
                    <div className="w-10 h-10 bg-white rounded-xl flex items-center justify-center shadow-sm">
                      <Layers className="w-5 h-5 text-blue-600" />
                    </div>
                    <div className="space-y-1">
                      <h4 className="font-bold text-slate-900 text-sm uppercase tracking-wide">Automatic Pipeline</h4>
                      <p className="text-xs text-slate-500 leading-relaxed font-medium">
                        One-Hot Encoding, Standard Scaling, and Median Imputation applied automatically.
                      </p>
                    </div>
                 </div>
              </div>
            </motion.div>
          )}

          {step === 3 && (
            <motion.div 
              key="step3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-12"
            >
              <div className="flex items-end justify-between border-b border-slate-200 pb-10">
                <div className="space-y-2">
                  <h2 className="text-4xl font-bold text-slate-900 tracking-tight">Engine Monitor</h2>
                  <p className="text-slate-500 text-sm font-medium">Monitoring pipeline for Task: <span className="font-mono text-blue-600">{taskId?.slice(0,8) || "..."}_ID</span></p>
                </div>
                <div className={`px-6 py-2.5 rounded-full text-xs font-bold border transition-all duration-500 ${status === "completed" ? "bg-emerald-50 text-emerald-600 border-emerald-100" : "bg-blue-50 text-blue-600 border-blue-100 animate-pulse"}`}>
                  {status.toUpperCase()}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
                <ProcessStep title="Ingestion" label="Success" done={status !== "idle"} current={status === "running"} icon={<Database />} />
                <ProcessStep title="EDA" label="Analyzed" done={status === "completed"} current={status === "running"} icon={<BarChart3 />} />
                <ProcessStep title="Encoding" label="Mapped" done={status === "completed"} current={status === "running"} icon={<Layers />} />
                <ProcessStep title="Training" label="Best Fit" done={status === "completed"} current={status === "running"} icon={<ImageIcon />} />
              </div>

              {status === "completed" && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                  <div className="lg:col-span-8 space-y-10">
                    <section className="card-minimal p-10 space-y-8">
                       <div className="flex items-center justify-between">
                         <h3 className="text-xl font-bold text-slate-900 flex items-center gap-3"><ImageIcon className="w-5 h-5 text-blue-600" /> Insight Gallery</h3>
                         <div className="flex gap-2">
                            {reports.map(r => r.endsWith('.csv') && (
                              <button key={r} onClick={() => downloadAsset(r)} className="p-2 bg-slate-50 border border-slate-200 rounded-lg hover:bg-white hover:shadow-sm transition-all">
                                <Download className="w-4 h-4 text-slate-500 hover:text-blue-600" />
                              </button>
                            ))}
                         </div>
                       </div>
                       <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                         {reports.filter(r => r.endsWith('.png')).map(img => (
                           <div key={img} className="space-y-4 group">
                              <div className="relative aspect-video bg-slate-50 rounded-2xl border border-slate-200 overflow-hidden cursor-pointer" onClick={() => downloadAsset(img)}>
                                <img src={`${API_BASE}/reports/${img}`} alt={img} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                                <div className="absolute inset-0 bg-white/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-all">
                                  <div className="p-3 bg-white rounded-full shadow-lg"><ExternalLink className="text-blue-600 w-5 h-5" /></div>
                                </div>
                              </div>
                              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block text-center">{img.replace(/_/g, ' ').replace('.png', '')}</span>
                           </div>
                         ))}
                       </div>
                    </section>

                    <section className="card-minimal p-10 space-y-8">
                       <h3 className="text-xl font-bold text-slate-900">Export Assets</h3>
                       <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                         {taskType === "tabular" ? (
                           <>
                             <ExportButton name="best_tabular_model.pkl" label="Main Model" sub="Optimized Estimator" color="blue" onClick={downloadAsset} />
                             <ExportButton name="preprocessor.pkl" label="Preprocessor" sub="Fitted Transformers" color="slate" onClick={downloadAsset} />
                             <ExportButton name="summary.csv" label="Training Log" sub="Full Metrics" color="slate" onClick={downloadAsset} />
                           </>
                         ) : (
                           <>
                             <ExportButton name="best_cv_model.pth" label="PyTorch Model" sub="Weights & Biases" color="blue" onClick={downloadAsset} />
                             <ExportButton name="cv_label_map.pkl" label="Label Map" sub="Class Mapping" color="slate" onClick={downloadAsset} />
                             <ExportButton name="sample_images.png" label="Sample Batch" sub="Dataset Preview" color="slate" onClick={downloadAsset} />
                           </>
                         )}
                       </div>
                    </section>
                  </div>

                  <div className="lg:col-span-4 space-y-8">
                    <div className="card-minimal p-8 space-y-8 h-fit sticky top-24">
                      <div className="space-y-2">
                         <h3 className="text-xl font-bold text-slate-900">Inference Lab</h3>
                         <p className="text-sm text-slate-500 leading-relaxed font-medium">Test the model with custom inputs. Transformations applied automatically.</p>
                      </div>
                      
                      <div className="space-y-6">
                        <div className="max-h-[350px] overflow-y-auto pr-2 space-y-5 custom-scrollbar">
                          {taskType === "tabular" ? (
                            columns.filter(c => c !== target).map(col => (
                              <div key={col} className="space-y-2">
                                <label className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">{col}</label>
                                <input 
                                  type="text" 
                                  placeholder={`Value for ${col}...`}
                                  className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-blue-100 focus:border-blue-400 outline-none transition-all font-medium"
                                  onChange={(e) => setPredictionData({...predictionData, [col]: e.target.value})}
                                />
                              </div>
                            ))
                          ) : (
                            <div className="space-y-5">
                              <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block">Upload Sample Image</label>
                              <div className="relative aspect-square bg-slate-50 border-2 border-dashed border-slate-200 rounded-2xl flex flex-col items-center justify-center gap-4 group hover:bg-blue-50-30 transition-all overflow-hidden">
                                {previewImage ? (
                                  <img src={previewImage} alt="Preview" className="w-full h-full object-cover" />
                                ) : (
                                  <>
                                    <ImageIcon className="w-8 h-8 text-slate-300" />
                                    <span className="text-xs text-slate-400 font-medium">Click to pick image</span>
                                  </>
                                )}
                                <input type="file" accept="image/*" onChange={handleImageInference} className="absolute inset-0 opacity-0 cursor-pointer" />
                              </div>
                            </div>
                          )}
                        </div>

                        <button 
                          onClick={handlePredict}
                          className="w-full bg-blue-600 hover:bg-blue-700 text-white py-4 rounded-xl font-bold uppercase text-xs tracking-widest transition-all shadow-lg shadow-blue-100 flex items-center justify-center gap-2 group"
                        >
                           <Send className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                           Predict Now
                        </button>

                        {predictionResult !== null && (
                           <motion.div 
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              className="p-8 bg-blue-600 rounded-2xl text-center relative overflow-hidden shadow-xl shadow-blue-200"
                           >
                              <div className="relative">
                                <span className="text-[10px] text-blue-100 uppercase font-bold block mb-2 tracking-widest opacity-80">Prediction Result</span>
                                <span className="text-4xl font-bold text-white tracking-tight">{predictionResult}</span>
                              </div>
                           </motion.div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function NavButton({ active, onClick, label }) {
  return (
    <button 
      onClick={onClick} 
      className={`px-5 py-2 rounded-xl text-sm font-semibold transition-all ${active ? "bg-blue-50 text-blue-600 shadow-sm border border-blue-100" : "text-slate-500 hover:text-slate-900 hover:bg-slate-50"}`}
    >
      {label}
    </button>
  );
}

function ProcessStep({ title, label, done, current, icon }) {
  return (
    <div className={`p-6 rounded-2xl border transition-all duration-500 ${done ? "bg-emerald-50 border-emerald-100" : (current ? "bg-blue-50 border-blue-100" : "bg-white border-slate-200")}`}>
      <div className={`w-10 h-10 rounded-xl mb-4 flex items-center justify-center ${done ? "bg-emerald-100 text-emerald-600" : (current ? "bg-blue-100 text-blue-600 shadow-sm" : "bg-slate-100 text-slate-400")}`}>
        {React.cloneElement(icon, { size: 18 })}
      </div>
      <div className="space-y-1">
        <span className="text-[10px] font-bold uppercase text-slate-400 block tracking-widest">{title}</span>
        <span className={`text-lg font-bold block ${done ? "text-emerald-600" : (current ? "text-blue-600" : "text-slate-300")}`}>{label}</span>
      </div>
    </div>
  );
}

function ExportButton({ name, label, sub, color, onClick }) {
  const bgClass = color === "blue" ? "bg-blue-50 border-blue-100 hover:bg-blue-100" : "bg-slate-50 border-slate-200 hover:bg-white hover:shadow-sm";
  const iconClass = color === "blue" ? "text-blue-600" : "text-slate-400 group-hover:text-blue-600";
  
  return (
    <button onClick={() => onClick(name)} className={`p-6 ${bgClass} border rounded-2xl text-left transition-all group`}>
       <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-bold text-slate-400 tracking-tight uppercase">{sub}</span>
          <Download className={`w-4 h-4 ${iconClass} transition-colors`} />
       </div>
       <span className="text-base font-bold text-slate-900 block">{label}</span>
    </button>
  );
}

export default App;
