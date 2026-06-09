# What Really Determines a Developer's Salary in 2024?
### An Analysis of the Stack Overflow Developer Survey

---

## Motivation

Every developer wonders: *Am I being paid fairly? What would actually boost my salary?*
This project uses the 2024 Stack Overflow Developer Survey — the largest annual snapshot
of the global software industry — to build a machine learning model that predicts US
developer salaries and uncovers the factors that matter most.

---

## Questions Answered

1. **Feature Importance** — What developer characteristics most strongly predict salary?
2. **Remote Work Premium** — Does remote work actually pay more in 2024?
3. **Model Accuracy** — How well can we predict a developer's annual compensation?
4. **Prediction Scenario** — What salary should a specific developer profile expect?

---

## Files

| File | Description |
|------|-------------|
| `so_survey_salary_analysis.ipynb` | Main Jupyter notebook — EDA, model training, predictions |
| `build_notebook.py` | Script that generates the notebook programmatically |
| `so_survey_2024.csv` | Raw 2024 Stack Overflow Developer Survey (65,437 respondents) |
| `README.md` | This file |

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, feature engineering |
| `numpy` | Numerical operations, bootstrap statistics |
| `matplotlib` | All visualisations |
| `seaborn` | Heatmap and distribution plots |
| `scikit-learn` | GradientBoostingRegressor, pipelines, cross-validation |
| `scipy` | Kruskal-Wallis statistical test |

---

## Summary of Results

### 1. Top Salary Predictors (Feature Importance)
The Gradient Boosting model ranks the following as most influential:
1. **Years of professional experience** (43.8% of importance) — the single strongest lever
2. **Developer role / DevType** (17.7%) — Eng Managers and Cloud/ML engineers earn premiums
3. **Organisation size** (17.1%) — larger companies pay significantly more

### 2. Remote Work Pays More — By a Wide Margin
- Remote median: **$155,000**
- Hybrid median: **$138,000** (+25.5% vs in-person)
- In-person median: **$110,000**

Remote roles in 2024 concentrate at high-paying, location-agnostic technology companies.
The premium narrows at senior experience levels (15+ years) where in-person leadership
roles catch up.

### 3. Model Accuracy
- **R² = 0.247** (test set) / **CV R² = 0.229 ± 0.016** — explains ~25% of salary variance
- **MAE = $44,026** — predictions typically within ±$44k of actual salary
- Remaining variance is driven by unobserved factors: specific employer, equity packages,
  negotiation, and local market premiums

### 4. Prediction Scenario
A mid-career back-end developer (8 years experience, Master's degree, mid-size firm,
uses AI tools, hybrid) has a predicted salary of **$158,081**. Switching to remote
adds a modest **+$3,720/year** (+2.4%). The largest controllable factor is experience:
a jump from 8 → 15 years of experience yields a predicted increase of ~$20–30k/year.

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# 2. Download the survey data (auto-downloaded in notebook)
# Data source: https://github.com/StackExchange/Survey

# 3. Run notebook
jupyter notebook so_survey_salary_analysis.ipynb
```

---

## Data Source & Acknowledgements

- **Data:** [Stack Overflow Developer Survey 2024](https://survey.stackoverflow.co/)
  hosted at [github.com/StackExchange/Survey](https://github.com/StackExchange/Survey)
- **Survey methodology:** ~65,000 developers worldwide, conducted May 2024
- **CRISP-DM process** followed: Business Understanding → Data Understanding →
  Data Preparation → Modelling → Evaluation → Deployment (blog post)
- Project completed as part of the Udacity Data Science Nanodegree

---

*Note: Analysis filtered to US professional developers with reported salaries between
$20,000 and $500,000/year (n = 4,533 after cleaning).*
