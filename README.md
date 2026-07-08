# 🎓 Student Performance Prediction System

A machine learning application that analyzes student academic data, predicts performance outcomes, identifies key factors affecting performance, and generates actionable improvement recommendations — all through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Dashboard Pages](#-dashboard-pages)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)

---

## 🔍 Project Overview

The **Student Performance Prediction System** is designed for EdTech platforms, learning management systems, and academic analytics. It provides:

- **Data Analysis**: Comprehensive exploratory data analysis of student academic records
- **Performance Prediction**: ML models that predict student scores and performance levels
- **Factor Identification**: Feature importance analysis to identify key academic success factors
- **Actionable Recommendations**: Personalized improvement suggestions for each student
- **Interactive Dashboard**: A beautiful, professional Streamlit dashboard for visualization

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **EDA Dashboard** | Interactive charts with filters for gender, income, education, and performance |
| 🤖 **3 ML Models** | Linear Regression, Decision Tree, and Random Forest |
| 🎯 **Single Student Prediction** | Input student details and get instant predictions |
| 📈 **Student Comparison** | Side-by-side radar chart comparison of any two students |
| 💡 **Smart Recommendations** | Priority-sorted improvement suggestions with impact scores |
| 🕸️ **Radar Charts** | Visual student profile vs. class average comparison |
| 📋 **Data Preprocessing** | Handles missing values, duplicates, encoding, and normalization |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML Libraries** | Scikit-learn, NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Model Persistence** | Joblib |

---

## 📁 Project Structure

```
Student-Performance-Prediction/
├── data/
│   └── student_data.csv           # Generated dataset (1,000 students)
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py        # Synthetic data generator
│   ├── data_preprocessing.py      # Data cleaning & preprocessing
│   ├── eda.py                     # EDA visualization generator
│   ├── model.py                   # ML model training & evaluation
│   └── recommender.py             # Recommendation engine
├── models/
│   ├── linear_regression.pkl      # Saved regression model
│   ├── decision_tree.pkl          # Saved decision tree model
│   └── random_forest.pkl          # Saved random forest model
├── plots/
│   └── *.png                      # Generated EDA charts
├── app.py                         # Streamlit dashboard (main entry)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Dataset
```bash
python src/generate_dataset.py
```

### Step 4: Train Models
```bash
python -c "import sys; sys.path.insert(0,'src'); from data_preprocessing import preprocess_data; from model import train_all_models; train_all_models(preprocess_data())"
```

### Step 5: Generate EDA Plots (Optional)
```bash
python -c "import sys; sys.path.insert(0,'src'); from data_preprocessing import load_data, handle_missing_values, remove_duplicates; from eda import generate_all_plots; generate_all_plots(remove_duplicates(handle_missing_values(load_data())))"
```

### Step 6: Launch Dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📊 Dataset

The system uses a **synthetic dataset** with 1,000+ student records containing 16 features:

| Feature | Type | Range |
|---------|------|-------|
| `student_id` | Integer | 1-1000 |
| `gender` | Categorical | Male, Female |
| `age` | Integer | 17-25 |
| `attendance_percentage` | Float | 40-100% |
| `study_hours_per_week` | Float | 0-40 hrs |
| `previous_cgpa` | Float | 2.0-10.0 |
| `internal_marks` | Integer | 0-50 |
| `participation_score` | Integer | 0-10 |
| `sleep_hours` | Float | 4-10 |
| `extracurricular` | Categorical | Yes, No |
| `internet_access` | Categorical | Yes, No |
| `parent_education` | Categorical | High School, Bachelor, Master, PhD |
| `family_income` | Categorical | Low, Medium, High |
| `test_preparation` | Categorical | Completed, Not Completed |
| `final_exam_score` | Float | 0-100 (Target) |
| `performance_level` | Categorical | High, Average, Low (Target) |

---

## 🤖 Machine Learning Models

### 1. Linear Regression
- **Task**: Predict continuous final exam score
- **Metrics**: R², MAE, RMSE

### 2. Decision Tree Classifier
- **Task**: Classify performance level (High/Average/Low)
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### 3. Random Forest Classifier
- **Task**: Classify performance level (High/Average/Low)
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Feature Importance**: Identifies top factors affecting performance

---

## 📱 Dashboard Pages

1. **🏠 Home** — Project overview, key metrics, performance distribution
2. **📊 EDA Dashboard** — Interactive charts with data filters
3. **🤖 Model Performance** — Model comparison, confusion matrices, feature importance
4. **🎯 Predict** — Input form for single student prediction
5. **📈 Student Comparison** — Side-by-side comparison with radar charts
6. **💡 Recommendations** — Personalized improvement suggestions

---

## 📸 Screenshots

*Add dashboard screenshots here after deployment*

---

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

**Made with ❤️ for the Machine Learning Capstone Project**
