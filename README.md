# 🌲 Sentinel-2 Multi-Temporal Tree Species Classification in Germany

## 🧭 Background  
This project aims to support **forest resource monitoring in Germany** by developing a **multi-temporal Sentinel-2 tree species classification system**.  
Based on **37,907 sampling points**, the system performs a **three-level hierarchical classification** (“leaf type → genus → species”) covering **19 species and 9 genera**.  
It integrates temporal, spectral, and spatial information to provide fine-scale forest type mapping and supports ecological monitoring and biodiversity assessment.  

---

## ⚙️ Objectives  
- Build an **automated preprocessing pipeline** for multi-temporal Sentinel-2 imagery (cloud masking, temporal interpolation, vegetation indices).  
- Design a **hierarchical classification framework** (leaf → genus → species) combining spectral and temporal features.  
- Compare the performance of **five models** (ViT, XGBoost, GNN, CNN, Random Forest).  
- Improve **model generalization** through temporal interpolation and SMOTE oversampling.  
- Generate **spatially explicit species distribution maps** for Germany.  

---

## 🧩 Technical Workflow  

### 1️⃣ Data Acquisition & Preprocessing  
- **Source:** Sentinel-2 Level-2A imagery (Germany).  
- **Pipeline:**  
  - Cloud masking (threshold = 20%).  
  - Dual-stage spatio-temporal interpolation (spatial kernel = 3–5 pixels).  
  - Computation of vegetation indices: NDVI, EVI, SAVI, NDWI, MSI.  
  - Automated preprocessing pipeline using `GDAL`, `Rasterio`, and `NumPy`.  

### 2️⃣ Feature Extraction & Model Design  
- Constructed temporal–spectral feature vectors for each sampling point.  
- Defined hierarchical classification levels:  
  - **Level 1:** Leaf Type (Coniferous / Broadleaf)  
  - **Level 2:** Genus  
  - **Level 3:** Species  
- **Models:**  
  - ViT (Vision Transformer)  
  - XGBoost / LightGBM  
  - GNN (Graph Neural Network)  
  - CNN / Random Forest (Baselines)  

### 3️⃣ Data Optimization & Evaluation  
- **Enhancements:**  
  - SMOTE oversampling for class balance.  
  - Temporal interpolation for improved feature stability.  
- **Metrics:**  
  - Accuracy, F1-score, Precision, Recall at all three levels.  
  - Confusion matrices and inter-class error analysis.  
- **Visualization:**  
  - Species distribution maps (GeoTIFF / shapefile).  
  - Interactive visualization using `Folium` or `Kepler.gl`.  

---

## 📅 Project Timeline  

| Phase | Weeks | Tasks | Deliverables |
|-------|--------|--------|---------------|
| **Phase 1: Data Preparation** | Week 1–2 | Collect Sentinel-2 imagery, perform cloud masking & temporal interpolation | Preprocessed imagery & vegetation indices |
| **Phase 2: Feature Engineering** | Week 3–4 | Build temporal feature vectors & hierarchical labels | Feature dataset |
| **Phase 3: Model Training** | Week 5–7 | Train ViT, XGBoost, GNN; perform hyperparameter tuning | Trained models & results |
| **Phase 4: Evaluation** | Week 8 | Evaluate model accuracy & analyze confusion matrices | Model performance report |
| **Phase 5: Integration & Visualization** | Week 9–10 | Build automated classification pipeline & visualize outputs | Forest species maps |
| **Phase 6: Documentation & Deployment** | Week 11–12 | Write documentation & publish GitHub repository | Open-source code & report |

---

## 🛠️ Tech Stack  
- **Languages:** Python  
- **Frameworks:** PyTorch, XGBoost, Scikit-learn  
- **Remote Sensing Tools:** GDAL, Rasterio, SentinelHub API, Google Earth Engine  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn, Folium, Kepler.gl  
- **Version Control:** GitHub, DVC, Weights & Biases  

---

## 📈 Expected Outcomes  
- ✅ Automated multi-temporal Sentinel-2 preprocessing pipeline  
- ✅ Reproducible modeling and evaluation scripts  
- ✅ Three-level (leaf–genus–species) classification framework  
- ✅ Germany-wide tree species distribution maps  
- ✅ Complete open-source repository with documentation and results  

---

## 🌟 Highlights & Innovations  
- Dual-stage **spatio-temporal interpolation** to handle cloud coverage and temporal gaps.  
- Hierarchical classification framework combining **spectral and temporal dynamics**.  
- Integration of **deep (ViT, GNN)** and **shallow (XGBoost)** models for performance comparison.  
- Generalizable pipeline applicable to **other regions or ecosystems**.  

---

## 📁 Repository Structure  
