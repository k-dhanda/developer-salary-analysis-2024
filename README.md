# What Really Determines a Developer's Salary in 2024?

> A data science analysis of the Stack Overflow Developer Survey 2024 — exploring salary predictors, the remote work premium, and ML-based compensation forecasting.

📖 **[Read the Blog Post](https://k-dhanda.github.io/developer-salary-analysis-2024/)** | 🔗 **[View Notebook](so_survey_salary_analysis.ipynb)**

---

## Motivation

Every developer wonders: *Am I being paid fairly? What would actually boost my salary?*

This project applies the CRISP-DM data science process to the 2024 Stack Overflow Developer Survey — the largest annual snapshot of the global software industry (65,437 respondents) — to build a machine learning model that predicts US developer salaries and uncovers the factors that matter most.

---

## Questions Answered

| # | Question |
|---|---------|
| 1 | What developer characteristics most strongly **predict salary**? |
| 2 | Does **remote work** actually pay more in 2024? |
| 3 | How accurately can a model **predict annual compensation**? |
| 4 | What salary should a **specific developer profile** expect? |

---

## Files

| File | Description |
|------|-------------|
| `so_survey_salary_analysis.ipynb` | Main Jupyter notebook — EDA, feature engineering, model training, evaluation, and predictions (fully executed with 9 inline visualisations) |
| `build_notebook.py` | Reproducible build script that generates the notebook programmatically |
| `index.md` | Blog post (published via GitHub Pages) |
| `images/` | Exported chart images used in the blog post |
| `README.md` | This file |

> **Note:** `so_survey_2024.csv` is excluded from the repo (`.gitignore`) due to size. The notebook downloads it automatically from the Stack Exchange GitHub archive.

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.x | Data loading, cleaning, feature engineering |
| `numpy` | 1.x | Numerical operations |
| `matplotlib` | 3.x | All visualisations (9 figures) |
| `seaborn` | 0.x | Heatmap and distribution plots |
| `scikit-learn` | 1.x | GradientBoostingRegressor, pipelines, cross-validation |
| `scipy` | 1.x | Kruskal-Wallis non-parametric statistical test |

---

## Summary of Results

### 1. Top Salary Predictors
The Gradient Boosting model identifies three dominant features:
1. **Years of professional experience** — 43.8% of feature importance; the single strongest lever
2. **Developer role / DevType** — 17.7%; Engineering Managers and Cloud/ML engineers command premiums
3. **Organisation size** — 17.1%; larger companies pay significantly more at every experience level

### 2. Remote Work Pays More
| Work Arrangement | Median Salary |
|-----------------|--------------|
| In-person | $110,000 |
| Hybrid | $138,000 (+25.5%) |
| **Remote** | **$155,000 (+40.9%)** |

The remote premium is statistically significant (Kruskal-Wallis p < 0.001) and holds across experience brackets.

### 3. Model Accuracy
- Test **R² = 0.247**, Cross-validation **R² = 0.229 ± 0.016**
- **MAE ≈ $44,026** — predictions within ±$44k of actual salary
- Remaining variance driven by: specific employer, equity/bonus, negotiation, cost-of-living

### 4. Prediction Scenario
A mid-career back-end developer (8 yrs experience, Master's, mid-size firm, hybrid) →
**predicted $158,081/year**. Moving to remote adds +$3,720 (+2.4%). Gaining 7 more years
of experience is worth ~$30–40k more than any single structural change.

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# Clone repo
git clone https://github.com/k-dhanda/developer-salary-analysis-2024.git
cd ai-udacity

# Launch notebook (downloads data automatically on first run)
jupyter notebook so_survey_salary_analysis.ipynb
```

---

## Data Source & Acknowledgements

- **Data:** [Stack Overflow Developer Survey 2024](https://survey.stackoverflow.co/) via [github.com/StackExchange/Survey](https://github.com/StackExchange/Survey)
- **Methodology:** ~65,000 developers worldwide, conducted May 2024; analysis filtered to US professionals with salaries $20k–$500k/year (n = 4,533)
- **Process:** CRISP-DM (Business Understanding → Data Understanding → Preparation → Modelling → Evaluation → Deployment)
- Completed as part of the **Udacity Data Science Nanodegree** — Write a Data Science Blog Post project
