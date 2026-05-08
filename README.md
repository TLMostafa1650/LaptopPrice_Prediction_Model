# 🖥️ Laptop Price Prediction Model

A machine learning project that predicts laptop prices using a complete regression pipeline built with **Scikit-learn** and an interactive **Streamlit** web application.

The project includes:

- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature preprocessing
- PCA dimensionality reduction
- Support Vector Regression (SVR)
- Hyperparameter tuning with GridSearchCV
- Model evaluation & baseline comparison
- Interactive Streamlit UI

---

# 📌 Project Overview

This project aims to estimate laptop prices based on hardware specifications and product attributes such as:

- RAM
- Storage
- CPU
- GPU
- Screen size
- Refresh rate
- Brand
- Operating system
- Battery capacity
- And more...

The model uses an end-to-end machine learning pipeline to automate preprocessing and prediction.

---

# 💻 Streamlit Web App

The project includes a modern interactive Streamlit web application where users can:

- Configure laptop specifications
- Predict laptop prices instantly
- View model performance metrics
- Explore a clean responsive UI
- Estimate pricing confidence ranges

---

# 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

# ⚙️ Machine Learning Pipeline

The workflow includes:

1. Train/Test Split
2. Data Cleaning
3. Missing Value Imputation
4. Feature Scaling
5. One-Hot Encoding
6. PCA Dimensionality Reduction
7. Support Vector Regression (SVR)
8. GridSearchCV Hyperparameter Tuning
9. Model Evaluation

---

# 📊 Exploratory Data Analysis

The project includes visualizations for:

- Price distribution
- Correlation heatmaps
- Top features correlated with price
- RAM vs Price analysis
- Residual distribution
- Actual vs Predicted plots
- PCA explained variance

---

# 📈 Model Evaluation Metrics

The model is evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

A baseline model using `DummyRegressor` is also included for comparison.

---

# 🚀 How to Run

## 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/LaptopPrice_Prediction_Model.git
cd LaptopPrice_Prediction_Model
```

---

## 2️⃣ Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

Or install using requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Add the dataset

Place the dataset file:

```bash
laptop_price_dataset.csv
```

inside the project directory.

---

# ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

After running the command, Streamlit will automatically open the app in your browser.

---

# 📂 Project Structure

```bash
LaptopPrice_Prediction_Model/
│
├── app.py
├── laptop_price_prediction.py
├── laptop_price_dataset.csv
├── README.md
└── requirements.txt
```

---

# 🎯 Features

✅ Full preprocessing pipeline  
✅ PCA dimensionality reduction  
✅ Hyperparameter optimization  
✅ Baseline model comparison  
✅ Interactive Streamlit interface  
✅ Data visualization  
✅ End-to-end regression workflow  
✅ Real-time laptop price prediction  

---

# 🖼️ Application Preview

The Streamlit app allows users to:

- Select laptop specifications from the sidebar
- Predict estimated prices instantly
- View confidence ranges
- See model evaluation metrics
- Explore laptop configuration summaries

