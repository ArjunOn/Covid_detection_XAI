# 🧠 COVID-19 Chest X-Ray Classification using EfficientNet and Explainable AI (Grad-CAM)

This repository presents a **Deep Learning project** for automatic **COVID-19 detection from chest X-ray images**, leveraging multiple **EfficientNet architectures** (B0, V2B0, and B7) for high-accuracy and resource-efficient classification.  
Additionally, the project integrates **Explainable AI (XAI)** tools like **Grad-CAM** to highlight the regions that influence the model’s decision-making process.  
All experiments are configured to run seamlessly on **cluster (HPC) environments**.

---

## 📘 Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Model Architectures](#model-architectures)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
- [Cluster Environment Setup](#cluster-environment-setup)
- [Results and Metrics](#results-and-metrics)
- [Performance Comparison](#performance-comparison)
- [Repository Structure](#repository-structure)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Author](#author)

---

## 🧩 Overview

The **COVID-19 pandemic** emphasized the need for rapid, accurate, and automated diagnostic systems.  
This project applies **state-of-the-art convolutional neural networks (CNNs)** to classify **chest X-ray images** into:

- 🦠 **COVID-19 Positive**
- 💓 **Normal / Non-COVID**

It also focuses on **explainability**, offering transparency in AI-based healthcare models through visualizations that show *where* the model focuses during its decision-making process.

---

## 🎯 Objectives

- Build a robust COVID-19 detection system using **EfficientNet** models.  
- Compare the performance of **EfficientNetB0**, **EfficientNetV2B0**, and **EfficientNetB7**.  
- Ensure fairness through **K-Fold Cross Validation**.  
- Provide **interpretability** using Grad-CAM visual explanations.  
- Optimize training and inference for **High-Performance Computing (HPC)** environments.

---

## 🏗️ Model Architectures

| Model | Parameters | Input Size | Description |
|:------|:-----------:|:-----------:|:-------------|
| **EfficientNetB0** | ~5.3M | 224×224 | Lightweight and fast baseline |
| **EfficientNetV2B0** | ~7.1M | 224×224 | Enhanced training efficiency and accuracy |
| **EfficientNetB7** | ~66M | 600×600 | Large capacity, higher accuracy but GPU-intensive |

Each model uses identical data pipelines and metrics for consistent evaluation.

---

## 🩺 Dataset

- **Dataset:** COVID-QU-Ex  
- **Classes:** `COVID-19` and `Normal`  
- **Images per class:** 2000 (balanced subset)  
- **Split:**  
  - 80% Training  
  - 10% Validation  
  - 10% Testing  

### Preprocessing

- Resize all images to the input resolution expected by the model.  
- Normalize pixel values to [0,1].  
- (Optional) Random augmentations like flip, rotation, and zoom.  
  *(Disabled as it did not improve accuracy in this dataset.)*

---

## ⚙️ Methodology

1. **Data Loading:** Efficient `tf.data` pipelines for high-speed input streaming.  
2. **Model Construction:** Modular design allowing quick swaps between EfficientNet variants.  
3. **Cross Validation:** Stratified 5-Fold CV ensures class balance.  
4. **Training Configuration:**
   - Optimizer: `Adam (2e-4)`  
   - Loss: `Categorical Crossentropy`  
   - Batch Size: `16`  
   - Epochs: `30`
5. **Callbacks:**
   - `ModelCheckpoint` for saving best weights.  
   - `EarlyStopping` for preventing overfitting.  
6. **Metrics:** Accuracy, Precision, Recall, F1-score, AUC.

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM visualizes **which regions of an X-ray** contribute the most to a model’s classification decision.  
For each class, a representative image is selected from the test set to generate a heatmap overlay.

**Example Outputs:**
- `outputs/gradcam_COVID.png`  
- `outputs/gradcam_Normal.png`  

The **red/yellow regions** indicate areas of highest importance (attention) for the model.

---

## 🖥️ Cluster Environment Setup

The project was executed on an **HPC cluster** using **SLURM** for job scheduling.

### Environment Setup

```bash
# Load modules
module load python/3.9
module load tensorflow/2.12

# Create virtual environment
python -m venv covid_env
source covid_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Example SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=covid_effnet
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source covid_env/bin/activate
python efficientnet_covid_xai_stream.py
```

---

## 📊 Results and Metrics

Average metrics across all folds and test set evaluation:

| Model | Accuracy | Precision | Recall | F1-score | AUC |
|:------|:----------:|:----------:|:---------:|:----------:|:------:|
| **EfficientNetB0** | 0.93 | 0.91 | 0.92 | 0.91 | 0.94 |
| **EfficientNetV2B0** | 0.95 | 0.94 | 0.94 | 0.94 | 0.96 |
| **EfficientNetB7** | 0.97 | 0.96 | 0.97 | 0.96 | 0.98 |

Confusion matrices and accuracy/loss curves are automatically saved in the `outputs/` directory.

---

## 📈 Performance Comparison

Below is a visual comparison of the average model metrics:

```
EfficientNetB0   ████████████████▌ 93%
EfficientNetV2B0 ██████████████████▎ 95%
EfficientNetB7   ████████████████████▉ 97%
```

**Observation:** EfficientNetB7 consistently achieves the highest accuracy but requires the most memory (64GB+).  
EfficientNetV2B0 provides the best balance between performance and compute cost.

---

## 📁 Repository Structure

```
📂 COVID-Detection-Project/
├── efficientnet_covid_xai_stream.py     # Main script: training + Grad-CAM visualization
├── train_job.sh                         # SLURM batch script for HPC execution
├── requirements.txt                     # Python dependencies
├── outputs/                             # Generated results and plots
│   ├── fold_1_acc_loss.png
│   ├── fold_1_cm.png
│   ├── gradcam_COVID.png
│   ├── gradcam_Normal.png
│   ├── final_report.txt
│   └── folds_summary.csv
├── README.md                            # This file
```

---

## 🚀 Future Improvements

- Integrate **Transfer Learning** from pre-trained ImageNet weights.  
- Experiment with **Vision Transformers (ViT)** or **ConvNeXt** models.  
- Add **LIME** and **SHAP** explainability visualizations.  
- Develop a **Streamlit-based web interface** for live inference.  
- Extend to **3D CNNs** for CT-scan classification.  

---

## 📚 References

- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*  
- Kaggle COVID-QU-Ex Dataset  
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*  
- TensorFlow, Keras, and Scikit-learn documentation.

---

## 👨‍💻 Author

**Mallikarjuna Mannem**  
🎓 *M.S. Data Science and Analytics, Grand Valley State University*  
📧 mannemm@mail.gvsu.edu  
🔗 [LinkedIn](https://www.linkedin.com/in/mallikarjuna-mannem/)  

> *This repository demonstrates a full pipeline — from data loading to model explanation — showing how AI and interpretability can support healthcare diagnostics.*
