# 🥂 White Wine Quality Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Overview
> “Wine is the most healthful and most hygienic of beverages.” – *Louis Pasteur*  

This is an **end-to-end Machine Learning and Deep Learning project** that analyzes and predicts the **quality of white wine** based on its **physicochemical properties**.  
The project uses both **Random Forest Classification** and **Deep Neural Network (DNN) Regression** models to evaluate and predict wine quality scores.

---

## 📊 Dataset
**Source:** [Kaggle - Wine Quality Red & White Dataset](https://www.kaggle.com/datasets/abdullah0a/wine-quality-red-white-analysis-dataset)

### Features (Input Variables)
| Feature | Description |
|----------|--------------|
| Fixed Acidity | Tartaric, succinic, citric, and malic acids |
| Volatile Acidity | Gaseous acids present in wine |
| Citric Acid | Weak organic acid, adds freshness |
| Residual Sugar | Sugar remaining after fermentation |
| Chlorides | Amount of salt present |
| Free SO₂ | Protects wine from oxidation & microbes |
| Total SO₂ | Combined free + bound sulfur dioxide |
| Density | Density of wine |
| pH | Acidity level |
| Sulphates | Adds preservatives and flavor |
| Alcohol | Percentage of alcohol |
| **Quality** | Target variable (score between 0–10) |

---
⚙️ Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras  
- **Environment:** Google Colab  

---

## 🧠 Machine Learning Approach

### 1️⃣ Data Preprocessing
- Checked for missing values (`isnull().sum()`)
- Descriptive statistics using `describe()` and `info()`
- Label binarization:  
  - Quality ≥ 7 → **Good (1)**  
  - Quality < 7 → **Not Good (0)**  
- Split data: 80% Training / 20% Testing

### 2️⃣ Visualization
- Count plots for quality distribution  
- Correlation heatmap between features  
- Feature relationships with wine quality  

### 3️⃣ Model Training – **Random Forest Classifier**
- Built and trained using Scikit-learn  
- Achieved **Accuracy: 87.75%**  
- Model generalized well without overfitting  

### 4️⃣ Predictive System
- Takes physicochemical inputs (from dataset or user)
- Outputs **“Good”** or **“Bad”** wine quality prediction

---

## 🤖 Deep Learning Approach

### 1️⃣ Model Architecture
- **Input Layer:** 11 features  
- **Hidden Layers:** 3 layers × 512 neurons, ReLU activation  
- **Output Layer:** 1 neuron (continuous output)  

### 2️⃣ Training Setup
- Normalized data using `StandardScaler`
- Optimizer: **Adam**  
- Loss Function: **Mean Absolute Error (MAE)**  
- Batch Size: 256 | Epochs: 10–100  

### 3️⃣ Enhancements
- Implemented **Early Stopping** to prevent overfitting  
- Added **Dropout (0.3)** and **Batch Normalization** for stability  
- Used validation set to monitor performance  

### 4️⃣ Results
| Model | Metric | Result | Remarks |
| Random Forest | Accuracy | **87.75%** | Strong baseline |
| DNN Regression | MAE | **0.10** | Excellent precision |

---

## 📈 Results & Insights
- **Random Forest**: Quick, interpretable, and effective classification  
- **DNN Regression**: Captured complex nonlinear patterns for continuous prediction  
- Combining ML & DL shows how hybrid approaches can provide both interpretability and high performance.

---

## 📂 Project Structure
```
├── data/
│   └── winequality-white.csv
├── notebooks/
│   └── white_wine_quality.ipynb
├── models/
│   ├── random_forest.pkl
│   └── dnn_model.h5
├── visuals/
│   ├── correlation_heatmap.png
│   └── loss_curve.png
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/kiruthika-ai-eng/white-wine-quality-prediction.git
   cd white-wine-quality-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   Google collab notebooks/white_wine_quality.ipynb
   ```
4. Run all cells to reproduce results.

---

## 🏆 Key Learnings
- Data preprocessing and feature scaling significantly affect model performance.  
- Random Forests are strong baselines for tabular data.  
- Deep Learning (DNN) can capture nonlinear relationships if tuned properly.  
- Early stopping and dropout help avoid overfitting.

---

## 👩‍💻 Author
**Developed by:** [kiruthika-ai-eng](https://github.com/kiruthika-ai-eng)  

If you like this project, please ⭐ it on GitHub — it really helps!

---

## 📜 License
This project is licensed under the **MIT License** — feel free to use, modify, and share.
