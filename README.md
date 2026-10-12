# Philippine Concert Ticket Pricing Analytics & Prediction

## Project Overview

This project explores concert ticket pricing in the Philippines using **Python**, combining data cleaning, exploratory data analysis (EDA), and machine learning to examine pricing patterns and predict ticket prices based on concert and ticket characteristics.

## Objectives

- Prepare and validate a concert ticket pricing dataset for analysis.
- Explore ticket price distributions and differences across artist categories, ticket tiers, venues, locations, and years.
- Examine the relationship between artist popularity and ticket prices.
- Build and compare **Linear Regression** and **Random Forest Regression** models.
- Evaluate prediction accuracy and identify important predictors.

## Tools and Libraries

**Python**, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Joblib.

## Data and Features

The analysis uses `philippine_concert_tickets.csv`. The prediction target is **`ticket_price_php`**. Depending on availability in the dataset, predictors include artist category, ticket tier, venue, city or province, artist popularity, venue capacity, concert year, and concert month.

The workflow converts prices and dates to appropriate types, removes rows without valid target prices and exact duplicate records, standardizes text fields, and derives date-based features.

## Exploratory Data Analysis

The script includes:

- Ticket price descriptive statistics, histogram, and boxplot
- Average ticket prices by artist category and ticket tier
- Ticket price distributions across tiers
- Yearly average ticket prices
- Venue and location comparisons
- Artist popularity versus ticket price, including Pearson correlation
- Correlation matrix and heatmap for numerical variables

## Machine Learning Approach

**Data splitting:** An 80/20 grouped train/test split keeps ticket tiers from the same concert together to reduce leakage across sets. The script uses a concert ID or a combination of concert name, date, and venue to define groups.

**Preprocessing:** Missing numeric values are imputed with medians and scaled; missing categorical values are imputed with the most frequent category and encoded using one-hot encoding. Preprocessing is fitted on the training data through scikit-learn pipelines.

**Models:**

1. Linear Regression
2. Random Forest Regression

**Evaluation:** Models are compared using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R²**. The model with the lowest test MAE is selected. The workflow also includes actual-versus-predicted plots, residual analysis, and Random Forest feature importance.

> Model performance values are not listed here because they depend on running the script with the project dataset. Add the actual MAE, RMSE, and R² outputs after execution.

## Project Structure

```text
philippine-concert-ticket-pricing/
├── data/
│   └── philippine_concert_tickets.csv
├── scripts/
│   └── concert_ticket_analysis_clean.py
├── models/                      # Created when the script runs
│   └── concert_ticket_price_model.pkl
└── README.md
```

Adjust filenames or directories to match your repository if needed.

## How to Run

1. Install Python and the required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

2. Place the dataset in `data/` and the script in `scripts/` as shown above.
3. Run the script **from the `scripts/` directory** so its relative paths resolve correctly:

   ```bash
   cd scripts
   python concert_ticket_analysis_clean.py
   ```

4. Review the printed summaries, visualizations, model comparisons, and sample predictions. The selected model is saved in `models/`.

## Skills Demonstrated

Data cleaning and validation · Exploratory data analysis · Data visualization · Feature engineering · Group-aware train/test splitting · Regression modeling · Model evaluation · Error analysis · Model persistence

## Author

**Neil Christian Donato**  
GitHub: [neil-donato](https://github.com/neil-donato)
