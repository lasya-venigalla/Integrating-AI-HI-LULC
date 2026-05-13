# Integrated Human & Artificial Intelligence for Object-Based LULC
### Ph.D. Dissertation Project | Dallas County, Texas

## 📌 Project Overview
This research addresses a critical gap in High Spatial Resolution (HSR) imagery classification. While **Human Intelligence** (Curve Matching) excels at spectral detail via histograms, it lacks generalization and speed. Conversely, **Artificial Intelligence** (2D CNNs) generalizes well but suffers from "geometric mismatching" when fitting square chips to irregular land objects.

I developed a **novel integrated framework** that utilizes a **1D Convolutional Neural Network (CNN)** to process **object-level spectral histograms**. This approach eliminates geometric errors and significantly reduces processing time while maintaining a **98% overall accuracy**.

## 🚀 Key Innovations & Results
* **Geometric Precision:** Replaced fixed-dimension "image chips" with actual image objects, eliminating edge-mismatch errors.
* **Efficiency Gain:** Drastically reduced training and classification time compared to 2D CNNs and traditional Curve-Matching.
* **Enhanced Generalization:** Utilized CNN feature extraction to correctly classify spectral patterns not explicitly present in the reference samples.
* **High-Dimensional Integration:** Successfully fused **10 data channels** (8 WorldView-2 bands + NDVI + LiDAR-derived nDSM).

## 🛠 Tech Stack
* **Deep Learning:** Python, TensorFlow, Keras (Custom 1D CNN Architecture)
* **Geospatial Analysis:** ArcPy (MeanShiftSegmentation), ArcGIS Pro
* **Data Science:** NumPy, SciPy (Spectral Histogram Modeling)
* **Sensors:** WorldView-2 (1.84m Multispectral), LiDAR (nDSM)

## 🔬 Methodology
1.  **Segmentation:** Generated irregularly shaped image objects using `MeanShiftSegmentation` in ArcPy.
2.  **Feature Engineering:** Modeled internal spectral heterogeneity by generating 1D histograms for every object (bridging human-expert logic with automated AI).
3.  **Model Architecture:** Designed a 1D CNN to extract "deep features" directly from spectral signatures.
4.  **Inferencing:** Performed high-speed classification via model inferencing rather than direct 1-to-1 comparison.

## 📡 Data Channels
| Channel Type | Source | Purpose |
| :--- | :--- | :--- |
| **Spectral (8 Bands)** | WorldView-2 | Visible, Coastal, Yellow (seasonal change), Red-Edge, and NIR. |
| **Vegetation Index** | NDVI | Analyzes biomass and plant health. |
| **Topography** | nDSM (LiDAR) | Separates vertical structures (buildings/trees) from ground features. |
