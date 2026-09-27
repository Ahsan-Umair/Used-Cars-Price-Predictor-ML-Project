# Used Car Price Predictor

A regression project that estimates used-car prices from vehicle specifications, history, and appearance. The notebook includes substantial text cleanup and feature engineering before comparing linear and tree-based regression models.

## Workflow

- Parse currency and mileage strings and clean missing or placeholder values.
- Explore price, mileage, model year, accident history, and brand-level patterns.
- Correct multiword brand names and extract horsepower, engine size, and cylinder count.
- Identify electric vehicles and simplify transmission descriptions.
- Group rare models and colors to control categorical dimensionality.
- Encode categorical fields and standardize features for linear regression.
- Compare linear regression and random forest on raw and log-transformed targets.
- Evaluate MAE, RMSE, and R-squared, including a lower-price subset.
- Tune random-forest hyperparameters and inspect feature importance.
- Export a random-forest model and standard scaler.

## Notebook walkthrough

1. Inspect the 4,009 listings, duplicate and missing values, price distribution, mileage, model year, accident history, and brand-level prices. Parse currency and mileage strings and remove the single 2,954,083 price outlier explicitly identified in the notebook.
2. Normalize placeholder fuel values and missing categories. Repair split multiword brand names, then extract horsepower, engine displacement, and cylinder count from free-text engine descriptions.
3. Identify electric vehicles, impute engine measurements differently for electric and non-electric cars, simplify transmission descriptions, and extract transmission speeds.
4. Group rare models and colors, encode title and accident fields, one-hot encode categorical predictors, and keep the final feature table for training.
5. Make an 80/20 split with `random_state=42`. Scale features for linear regression but fit the baseline random forest on unscaled encoded features.
6. Compare linear and forest regressors on original and log-transformed prices with MAE, RMSE, and R-squared. Repeat evaluation for test listings below $200,000 and inspect forest feature importances.
7. Tune a separate forest with grid search, then evaluate that tuned estimator. The exported `used_car_price_model.joblib` is the **baseline** forest named `rf`, not the tuned forest; the saved scaler belongs to the linear-model experiment.

To predict from a new listing, reproduce every cleaning and encoding step and align its columns to the training feature order. Neither saved file contains that preprocessing pipeline.

## Dataset

`used_cars.csv` contains 4,009 listings. Fields include brand, model, model year, mileage, fuel type, engine description, transmission, exterior and interior colors, accident history, title status, and price.

## Repository contents

| File | Purpose |
| --- | --- |
| `Used-Cars-Price-Predictor.ipynb` | Cleaning, feature engineering, model comparison, tuning, and export |
| `used_cars.csv` | Source listing dataset |
| `used_car_price_model.joblib` | Exported random-forest regressor |
| `used_car_price_scaler.joblib` | Exported standard scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Used-Cars-Price-Predictor.ipynb
```

Inference requires reproducing the notebook's feature-engineering and one-hot-encoding columns. Note that the exported model is the baseline random forest assigned to `rf`, while the notebook also evaluates a separately tuned estimator.
