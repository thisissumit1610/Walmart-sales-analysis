# Walmart Sales Analysis 📊 | SQL & Python

## Project Overview

This project analyzes Walmart sales transactions across three branches to extract meaningful business insights. The analysis focuses on identifying top-performing product categories, understanding customer purchasing behavior, detecting peak sales periods, and evaluating branch-level performance.

Using **SQL for data querying and aggregation** and **Python (Pandas, Matplotlib)** for data exploration and visualization, this project demonstrates how retail sales data can be transformed into actionable insights for business decision-making.

The objective of this project is to simulate a real-world retail analytics workflow where data is queried, analyzed, and visualized to support operational and strategic decisions.

---

## Dataset Description

The dataset contains **1300+ Walmart sales transactions** from three branches.

### Features Included

* **Date** – Transaction date
* **Time_of_Purchase** – Exact purchase time
* **Branch** – Store branch (A, B, C)
* **City** – Store location
* **Region** – Regional classification
* **Product_Line** – Product category
* **Customer_Type** – Member or regular customer
* **Sales_Channel** – Online or in-store purchase
* **Quantity** – Number of units purchased
* **Unit_Price** – Price per unit
* **Discount** – Discount applied on purchase
* **Revenue** – Total sales value
* **Profit** – Profit generated from the transaction
* **Payment_Method** – Mode of payment
* **Customer_Rating** – Customer satisfaction score

---

## Tools & Technologies

* **SQL** – Data querying and aggregation
* **Python** – Data analysis
* **Pandas** – Data manipulation
* **Matplotlib / Seaborn** – Data visualization
* **Jupyter Notebook / Google Colab** – Analysis environment

---

## Project Workflow

### 1. Data Import

The dataset was imported into a SQL database and queried to extract insights.

### 2. Data Cleaning

Basic validation checks were performed:

* Checking for missing values
* Verifying duplicates
* Ensuring correct data types

### 3. SQL Analysis

SQL queries were used to analyze the dataset and answer key business questions.

Examples of analysis performed:

* Revenue by product line
* Profit by branch
* Regional sales performance
* Payment method distribution
* Monthly sales trends
* Peak purchase hours

### 4. Python Analysis

Python was used to further explore the dataset and visualize trends.

Key tasks performed:

* Data exploration using Pandas
* Feature engineering (extracting hour/month from time and date)
* Creating visualizations for trends and comparisons

### 5. Visualization

Visualizations were created using Matplotlib/Seaborn to show:

* Revenue by product category
* Sales distribution by branch
* Hourly sales patterns
* Monthly sales trends

---

## Key Business Insights

* Certain **product lines generated significantly higher revenue**, indicating strong customer demand.
* **Peak sales hours occurred during evening periods**, suggesting optimal timing for promotions and staffing.
* One branch consistently generated **higher profit margins**, highlighting differences in operational efficiency.
* Digital payment methods were widely used, reflecting increasing adoption of cashless transactions.

---

## Example Business Questions Answered

* Which product line generates the highest revenue?
* Which branch produces the most profit?
* What are the peak purchasing hours?
* How do sales vary by region?
* What payment methods are most commonly used?
* What are the monthly sales trends?

---

## Project Structure

```
walmart-sales-analysis
│
├── data
│   └── walmart_sales.csv
│
├── sql
│   └── walmart_queries.sql
│
├── notebooks
│   └── walmart_sales_analysis.ipynb
│
└── README.md
```

---

## Sample SQL Query

```sql
SELECT Product_Line,
       SUM(Revenue) AS Total_Revenue
FROM walmart_sales
GROUP BY Product_Line
ORDER BY Total_Revenue DESC;
```

---

## Future Improvements

* Build a predictive model to forecast sales trends
* Perform customer segmentation
* Apply time-series forecasting for demand planning

---

## ECG Signal Analysis & ML-Based Disease Prediction 🫀

### Overview

This module provides a complete pipeline for analysing ECG (Electrocardiogram)
signals and using machine learning to predict cardiac conditions. Five disease
classes are supported:

| Label | Condition |
|-------|-----------|
| 0 | Normal |
| 1 | Arrhythmia |
| 2 | Myocardial Infarction |
| 3 | Atrial Fibrillation |
| 4 | Bradycardia |

### Pipeline

1. **Signal Generation** – Synthetic ECG signals for demonstration (replace with real data for production).
2. **Preprocessing** – Bandpass filtering, normalization, and R-peak detection.
3. **Feature Extraction** – 19 time-domain and frequency-domain features (RR intervals, heart rate statistics, HRV metrics, power spectral density bands).
4. **Model Training** – Random Forest, Gradient Boosting, and SVM classifiers.
5. **Evaluation** – Accuracy, classification report, confusion matrix, and cross-validation.
6. **Prediction** – End-to-end prediction with human-readable disease labels and probability estimates.

### Quick Start

```python
from ecg_analysis.generate_synthetic_data import generate_feature_dataset
from ecg_analysis.model import ECGClassifier
from sklearn.model_selection import train_test_split

# Generate features from synthetic ECG signals
X, y = generate_feature_dataset(n_samples_per_class=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train a Random Forest classifier
clf = ECGClassifier(model_type="random_forest", n_estimators=100)
clf.train(X_train, y_train)

# Evaluate
results = clf.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.2%}")

# Predict disease labels
print(clf.predict_disease(X_test[:5]))
```

### ECG Module Structure

```
ecg_analysis/
├── __init__.py                  # Package exports
├── preprocessing.py             # Bandpass filter, normalization, R-peak detection
├── feature_extraction.py        # Time-domain & frequency-domain features
├── model.py                     # ECGClassifier (RF, GB, SVM)
└── generate_synthetic_data.py   # Synthetic ECG signal generation

tests/
└── test_ecg_analysis.py         # Unit tests (26 tests)

ecg_analysis_notebook.ipynb      # Interactive demo notebook
```

### Dependencies

* Python 3.8+
* NumPy
* Pandas
* SciPy
* scikit-learn
* Matplotlib (for notebook visualisations)

### Running Tests

```bash
pip install pytest
python -m pytest tests/test_ecg_analysis.py -v
```

### Using Real Data

For production use, replace the synthetic data generator with real ECG datasets such as:

* [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
* [PTB-XL ECG Database](https://physionet.org/content/ptb-xl/)
* [PhysioNet Challenge Datasets](https://physionet.org/)

---

## Conclusion

This project demonstrates how **SQL and Python can be combined to analyze retail sales data and uncover actionable business insights**, and how **ML techniques can be applied to ECG signal data for cardiac disease prediction.** The workflows replicate common tasks performed by data analysts and biomedical engineers.

---

## Author

Data Analysis Project – Walmart Sales Analysis & ECG Disease Prediction
SQL | Python | Machine Learning | Signal Processing | Data Visualization
