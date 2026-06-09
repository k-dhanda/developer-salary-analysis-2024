# What Really Determines a Developer's Salary in 2024?

*Published on [Medium / GitHub Pages] | June 2026 | 4 min read*

![Developer at laptop with salary chart overlay — illustrative image](https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=900)

---

If you've ever wondered whether your salary is competitive — or what would actually move the needle — you're not alone. Every year, Stack Overflow surveys over 65,000 developers worldwide. The 2024 edition is the most comprehensive snapshot of the industry yet.

I ran the US-only data through a machine learning model to find out: **what actually predicts a developer's pay?**

---

## The Data

65,437 global respondents. Filtered to US professional developers with reported salaries between $20k and $500k/year: **4,533 people**.

Median salary in the sample: **$142,000/year**. The range is wide — from $25k for entry-level to nearly $500k for senior engineers at top tech companies.

---

## Finding #1: Experience Is King

![Median salary by years of professional experience — line chart showing steep rise from 0–10 years then plateau](chart_placeholder_1)

Years of professional coding experience explains **43.8% of the model's predictive power** — more than education, company size, and developer role combined.

- 0–2 years: ~$95k median
- 6–10 years: ~$145k median
- 15–20 years: ~$180k median
- 30+ years: ~$175k median (slight plateau)

**Takeaway:** The sharpest salary growth happens in the first 10 years. After that, transitioning into engineering management or principal/staff roles becomes the main lever.

---

## Finding #2: Remote Work Pays More — A Lot More

This one surprised people even in the data community.

| Work Arrangement | Median Salary |
|-----------------|--------------|
| In-person | $110,000 |
| Hybrid | $138,000 |
| **Remote** | **$155,000** |

Remote workers earn **41% more** than in-person workers at the median. The gap holds even after controlling for experience level.

The likely explanation: remote-friendly companies (distributed tech startups, cloud-native firms) tend to pay at or above market rates to attract talent regardless of location — while in-person roles are concentrated in industries (finance, healthcare IT, consulting) with lower overall pay scales.

**Takeaway:** If salary is a priority, remote-first companies are worth prioritising in your job search.

---

## Finding #3: The Model and Its Limits

The Gradient Boosting model achieves an **R² of ~0.25** on held-out test data, with a mean absolute error of about **$44,000**.

That means the model is directionally correct — it captures broad patterns like experience premiums and role-based differences — but it cannot predict your specific salary with high precision. The remaining ~75% of variance comes from factors the survey doesn't capture:

- **Which company** (FAANG vs local agency)
- **Equity and bonus packages**
- **Negotiation skill**
- **Geographic cost-of-living adjustments**

Salary modelling is hard. But "broad patterns" is enough to inform career decisions.

---

## Finding #4: Should You Go Remote?

I modelled a specific scenario: a back-end developer, 8 years of experience, Master's degree, mid-size company (500–1,000 employees), uses AI tools.

- **Staying hybrid:** predicted $158,081/year
- **Going remote:** predicted $161,801/year (+$3,720)

The remote premium exists, but it's modest for this profile. The bigger opportunity is **gaining experience**: the same profile with 15 years of experience jumps to ~$195k — a $37,000 gain from staying in the field vs. chasing a remote arrangement.

---

## Key Takeaways

1. **Experience is the #1 predictor** of salary — invest in depth, not just breadth.
2. **Remote roles pay more**, especially at mid-career levels. The 2024 data confirms the remote premium is real.
3. **Large companies pay more** — organisation size is the #3 predictor. Moving from a 10-person startup to a 10,000-person company is worth ~$30–50k in this model.
4. **Education matters, but less than you'd think** — a Master's vs Bachelor's degree adds a modest premium; years of experience dominates.

---

*Data: Stack Overflow Developer Survey 2024 (survey.stackoverflow.co). Analysis and model: Gradient Boosting Regressor, scikit-learn. Full code on GitHub.*
