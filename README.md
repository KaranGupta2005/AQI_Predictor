# Indian Cities AQI Prediction

## Overview

This project predicts the **Air Quality Index (AQI) category** of Indian cities based on historical air pollutant data from **2015 to 2020**. The model uses **machine learning techniques** including XGBoost, Random Forest, and Decision Trees with advanced feature engineering such as **lag features, rolling averages, and city/season encoding**.

---

## Dataset

* **Source:** Public air quality datasets for Indian cities (2015–2020)
* **Size:** 29,531 rows, 16 columns
* **Columns:** `City`, `Date`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI_Bucket`
* **Data Cleaning:**

  * Dropped `Xylene` (>50% missing values)
  * Filled missing numeric pollutant values with **city-wise mean + forward/backward fill**
  * Filled missing `AQI_Bucket` values with **city-wise mode**
  * Replaced outliers using **IQR method**

---

## Feature Engineering

* **Date Features:** Extracted `Year`, `Month`, `Day` and mapped `Season` → one-hot encoded
* **City Encoding:**

  * One-hot or target encoding based on city AQI averages
  * Dropped cities with extremely low data (`Aizawl`, `Shillong`, `Coimbatore`, `Thiruvananthapuram`)
* **Lag Features:** Previous 1–3 days of pollutant values per city
* **Rolling Features:** 3-day, 7-day, 14-day rolling averages of pollutants
* **Correlation-Based Selection:** Kept pollutants with |corr| ≥ 0.4 with AQI

---

## Models

### 1. XGBoost Classifier

* Optimized for multi-class AQI prediction
* Key parameters:

```python
n_estimators=300, learning_rate=0.05, max_depth=6
```

* Accuracy: ~85–87%

### 2. Random Forest Classifier

* Tuned for balanced depth and sample splitting
* Key parameters:

```python
n_estimators=300, max_depth=15
```

* Accuracy: ~81–84%

### 3. Decision Tree Classifier

* Single-tree baseline
* Key parameters:

```python
max_depth=10
```

* Accuracy: ~77–80%

---

## Usage

1. Clone the repository:

```bash
git clone <your-repo-url>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Load the dataset and preprocess:

```python
import pandas as pd
df = pd.read_csv("indian_cities_aqi.csv")
```

4. Run the preprocessing, feature engineering, and model training scripts.

---

## Future Improvements

* Include **meteorological data** (temperature, humidity, wind)
* Hyperparameter tuning using **GridSearchCV or Optuna**
* Predict **continuous AQI values** instead of categories
* Deploy as a **real-time AQI prediction dashboard**

---

## References

* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [Indian Air Quality Data 2015–2020](#)

---

## Author

**Karan Gupta** — Web Developer & ML Practitioner
