# Philippine Concert Ticket Pricing Analytics & Prediction

## Project Overview

This Python project analyzes concert ticket prices in the Philippines and builds regression models to estimate ticket prices from concert characteristics. The workflow covers data validation, cleaning, exploratory data analysis (EDA), feature engineering, group-aware model training, and error analysis.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib.

## Dataset and Preparation

- **File:** `philippine_concert_tickets.csv`
- **Records:** 20,000 ticket-tier records across **2,500 concert events**
- **Original columns:** 17
- **Missing values:** 0; **exact duplicate rows:** 0
- **Rows removed during cleaning:** 0
- **Prediction target:** `ticket_price_php`

The dataset labels its concerts and artists as simulated and contains a `data_origin` field. The findings below describe **this supplied dataset**, not measured prices across the real Philippine concert market.

Cleaning converts ticket prices to numeric values, parses concert dates, trims text fields, removes invalid target rows and exact duplicates, and derives year, month, and day-of-week features.

## Exploratory Data Analysis — Results

### Overall Ticket Prices

| Metric | Result |
|---|---:|
| Records analyzed | 20,000 |
| Average ticket price | ₱6,316.63 |
| Median ticket price | ₱5,100.00 |
| Minimum ticket price | ₱500.00 |
| Maximum ticket price | ₱25,050.00 |
| Standard deviation | ₱4,062.09 |

### Ticket Prices by Artist Category

| Artist category | Average ticket price |
|---|---:|
| K-pop | ₱7,287.17 |
| International Pop | ₱7,000.69 |
| P-pop | ₱5,982.39 |
| Rock | ₱5,977.20 |
| Hip-hop | ₱5,866.55 |
| OPM | ₱5,671.41 |
| Indie | ₱5,199.59 |

Within the dataset, K-pop has the highest average ticket price among artist categories.

### Ticket Prices by Tier

| Ticket tier | Average ticket price |
|---|---:|
| VVIP | ₱12,877.60 |
| VIP Soundcheck | ₱10,371.86 |
| VIP | ₱8,051.74 |
| Premium Seated | ₱5,717.28 |
| Lower Box | ₱4,588.74 |
| Regular Seated | ₱3,898.88 |
| Upper Box | ₱3,016.60 |
| General Admission | ₱2,010.30 |

Ticket tier is a major differentiator of price in this dataset. The Random Forest's largest individual feature-importance scores also correspond to ticket-tier indicators.

### Average Ticket Prices by Year

| Concert year | Average ticket price |
|---|---:|
| 2023 | ₱6,308.75 |
| 2024 | ₱6,232.40 |
| 2025 | ₱6,343.91 |
| 2026 | ₱6,378.26 |

These are dataset-level averages, not evidence of historical market price trends.

### Additional Visualizations

The script generates a price histogram and boxplot; artist-category, ticket-tier, venue, and location comparisons; yearly price trends; and a numerical correlation heatmap. **Artist popularity versus price was not plotted in this run**, because the script's popularity-column detection did not match the dataset field `artist_popularity_index`.

## Machine Learning

### Data Split and Features

An **80/20 group-aware train/test split** kept records belonging to the same concert in the same partition:

- Training: **16,000 records / 2,000 concerts**
- Testing: **4,000 records / 500 concerts**

The predictors selected by the executed script were `artist_category`, `ticket_tier`, `venue`, `city_or_province`, `concert_year`, and `concert_month`. Categorical predictors were one-hot encoded, and numerical predictors were imputed and standardized inside training pipelines.

### Model Performance

Results from executing `concert_ticket_analysis_clean.py` on the supplied CSV:

| Model | MAE (PHP) ↓ | RMSE (PHP) ↓ | R² ↑ |
|---|---:|---:|---:|
| Linear Regression | ₱1,292.38 | **₱1,741.46** | **0.813** |
| Random Forest Regression | **₱1,264.26** | ₱1,758.97 | 0.809 |

**Selected model: Random Forest**, based on the lower test-set MAE (approximately ₱28.12 lower than Linear Regression). Linear Regression performed slightly better on RMSE and R².

### Feature Importance and Error Analysis

The three largest Random Forest one-hot feature importance scores were:

| Feature | Importance |
|---|---:|
| VVIP ticket tier | 0.4088 |
| VIP Soundcheck ticket tier | 0.2428 |
| VIP ticket tier | 0.1225 |

For the selected Random Forest, mean absolute percentage error across test records was **23.61%**. The script also generates actual-versus-predicted scatterplots, prediction-error distributions, residual plots, and a list of the largest absolute errors.

**Evaluation caveat:** The model was selected using the test-set MAE, so the selected model's reported test performance is exploratory rather than an unbiased final estimate. A separate validation set or cross-validation should be used for model selection before final holdout testing.

## Skills Demonstrated

Data quality checks · Data cleaning · Feature engineering · Exploratory data analysis · Statistical visualization · Group-aware train/test splitting · Regression pipelines · Model comparison · Feature importance · Residual analysis · Model serialization

## Author

**Neil Christian Donato**  
GitHub: [neil-donato](https://github.com/neil-donato)
