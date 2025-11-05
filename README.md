# heart_disease_prediction
Predicting and diagnosing heart disease
This project aims to **analyze and predict the presence of heart disease** in patients using **Machine Learning**. The dataset contains various health indicators such as age, cholesterol, blood pressure, and chest pain type. The notebook includes data visualization, feature analysis, and model training for accurate prediction.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Technologies Used](#technologies-used)
- [Visualizations Explained](#visualizations-explained)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🧠 Overview
The notebook walks through the entire process of **heart disease prediction**, including:
1. Importing and cleaning the dataset.
2. Exploring relationships between health factors and heart disease.
3. Visualizing distributions and correlations.
4. Building machine learning models to classify patients.

---

## 📊 Dataset
- The dataset used is commonly known as the **Heart Disease UCI dataset**.
- Each record represents patient data with features like:
  - `age`, `sex`, `chest_pain`, `resting_bp`, `cholesterol`, etc.
  - `target`: 1 = heart disease, 0 = no heart disease.

---

## 🔍 Exploratory Data Analysis (EDA)
EDA includes:
- Distribution of features using histograms and bar plots.
- Comparison of categorical features between target classes.
- Correlation heatmaps to identify strong relationships.

Example visualizations:

```python
df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
```
➡ Shows how many patients have or do not have heart disease.

```python
plt.figure(figsize=(15, 15))
for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='No Heart Disease', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Heart Disease', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
```
➡ Compares the distributions of categorical features between both classes.

---

## 🤖 Model Training
Models tested include:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)

Model performance is evaluated using **accuracy**, **precision**, **recall**, and **F1-score**.

---

## 📈 Results
The notebook displays:
- Confusion matrices for each model.
- Comparison of model accuracies.
- The best-performing model for predicting heart disease.

---

## ⚙️ Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd heart-disease-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the notebook:
   ```bash
   jupyter notebook heart_diaseas-prediction.ipynb
   ```

---

## 🧩 Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 🚀 Future Improvements
- Add deep learning models for better accuracy.
- Build a web interface for real-time predictions.
- Perform hyperparameter tuning for optimized results.
- Include more medical data for enhanced analysis.
