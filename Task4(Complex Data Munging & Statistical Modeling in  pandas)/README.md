### 📄 `README.md`

```markdown
# Complex Data Munging & Statistical Modeling with Pandas

This project demonstrates end-to-end data cleaning, feature engineering, and statistical modeling using a messy real-world dataset (simulated government finance data).

## 📁 Structure

```

.
├── data/
│   ├── raw\_data.csv         # Original messy dataset
│   └── cleaned\_data.csv     # Cleaned dataset after processing
├── data\_prep.ipynb          # Notebook for data cleaning and feature engineering
├── modeling.ipynb           # Notebook for statistical modeling (linear regression)
├── slides/
│   └── summary\_slides.pdf   # Summary of challenges, solutions, and results (optional)
├── README.md
└── requirements.txt

````

## ✅ Steps Covered

### 1. Data Cleaning (`data_prep.ipynb`)
- Handle missing values, type conversion
- Outlier detection (IQR method)
- Categorical encoding and schema normalization

### 2. Feature Engineering
- Derived features: log-expenses, income-per-age
- One-hot encoding for categorical variables

### 3. Modeling (`modeling.ipynb`)
- Linear regression with `statsmodels`
- Statistical interpretation (coefficients, p-values, CI)

## 💡 Requirements

Install required libraries:
```bash
pip install -r requirements.txt
````

```

---

### 📄 `requirements.txt`
```

pandas
numpy
statsmodels

```
