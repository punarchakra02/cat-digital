import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load the data with derived features
print("Loading data...")
df = pd.read_csv("equipment_rentals_with_features.csv")

# Convert dates 
df["checkout_date"] = pd.to_datetime(df["checkout_date"])
df["checkin_date"] = pd.to_datetime(df["checkin_date"])

# -----------------------------
# Data Preparation
# -----------------------------

# For simulation purposes, we'll create a split point 
# Since our data is from 2025, we'll pretend some is historical and some is future
# In a real scenario, you would use actual historical data (2006-2021) and forecast 2022

# Get the min and max dates in the dataset
min_date = df["checkout_date"].min()
max_date = df["checkout_date"].max()
print(f"Data spans from {min_date.date()} to {max_date.date()}")

# Use a fixed cutoff date (end of 2022)
cutoff_date = datetime(2023, 12, 31)
print(f"Using {cutoff_date.date()} as the cutoff for train/test split (2006-2023 for training, 2023 for testing)")

# Create a flag for train/test split
df["is_train"] = df["checkout_date"].dt.year <= 2024
df["is_test"] = df["checkout_date"].dt.year == 2025

# Print counts to verify we have 2023 data
year_counts = df["checkout_date"].dt.year.value_counts().sort_index()
print("\nData distribution by year:")
print(year_counts)
print(f"2024 records: {df[df['checkout_date'].dt.year == 2024].shape[0]}")


# -----------------------------
# Feature Engineering
# -----------------------------

# Some of these features are already created in derive_features.py, but we'll add a few more

# Extract month and day of week for seasonality
df["month"] = df["checkout_date"].dt.month
df["day_of_week"] = df["checkout_date"].dt.dayofweek

# Create features based on week number within year (for yearly seasonality)
df["week_of_year"] = df["checkout_date"].dt.isocalendar().week

# -----------------------------
# Prepare Aggregated Training Data
# -----------------------------

# Group by equipment_type and week (year_week)
# This ensures we create a demand forecast for each equipment type by week
print("Preparing aggregated training data...")

# Get unique equipment types and weeks
all_equipment_types = df["equipment_type"].unique()

# Get unique weeks in chronological order
all_weeks = sorted(df["year_week"].unique())

# Create a DataFrame with all combinations of equipment_type and week
from itertools import product
all_combos = list(product(all_equipment_types, all_weeks))
all_df = pd.DataFrame(all_combos, columns=["equipment_type", "year_week"])

# IMPORTANT: First recalculate target_checkout_count per equipment_type and week
# This fixes issues with the demand calculation
print("Recalculating target checkout counts to ensure accurate demand tracking...")
target_counts = df.groupby(["equipment_type", "year_week"])["equipment_id"].count().reset_index(name="recalculated_count")

# Join this with the actual data to get a complete dataset with all combinations
# This ensures we have rows for weeks with zero demand
agg_df = df.groupby(["equipment_type", "year_week"]).agg({
    "checkout_count_t-1": "first",     # lag 1 week
    "checkout_count_t-4": "first",     # lag 4 weeks
    "rolling_mean_4w": "first",        # rolling 4-week average
    "price_index": "mean",
    "pct_maint": "mean",
    "idle_ratio_avg": "mean",
    "equipment_type_encoded": "first", # New feature
    "usage_efficiency": "mean",        # New feature
    "rental_duration": "mean",         # New feature
    "rate_relative": "mean",           # New feature
    "recent_maintenance": "mean",      # New feature
    "maintenance_per_day": "mean",     # New feature
    "engine_hours_per_day": "mean",    # New feature
    "month_sin": "first",              # New feature
    "month_cos": "first",              # New feature
    "week_sin": "first",               # New feature
    "week_cos": "first",               # New feature
    "rolling_std_4w": "first",         # New feature
    "rolling_max_4w": "first",         # New feature
    "is_train": "first",
    "is_test": "first",
    "month": "first",
    "day_of_week": "first",
    "week_of_year": "first"
}).reset_index()

# Merge with the recalculated counts
agg_df = agg_df.merge(target_counts, on=["equipment_type", "year_week"], how="left")
agg_df["target_checkout_count"] = agg_df["recalculated_count"].fillna(0).astype(int)
agg_df.drop("recalculated_count", axis=1, inplace=True)

# Merge with all_df to get complete dataset
complete_df = all_df.merge(agg_df, on=["equipment_type", "year_week"], how="left")

# Fill NaN values (weeks with no rentals)
complete_df["target_checkout_count"] = complete_df["target_checkout_count"].fillna(0)

# For lag features, forward-fill when possible, otherwise use zeros
for col in ["checkout_count_t-1", "checkout_count_t-4", "rolling_mean_4w"]:
    complete_df[col] = complete_df.groupby(["equipment_type"])[col].ffill().fillna(0)

# Fill remaining NaN values with appropriate defaults
complete_df["price_index"] = complete_df["price_index"].fillna(1.0)  # Default price index
complete_df["pct_maint"] = complete_df["pct_maint"].fillna(0)
complete_df["idle_ratio_avg"] = complete_df["idle_ratio_avg"].fillna(0)

# Fill NaN values for new features
complete_df["equipment_type_encoded"] = complete_df["equipment_type_encoded"].fillna(0)
complete_df["usage_efficiency"] = complete_df["usage_efficiency"].fillna(0.5)  # Assume medium efficiency
complete_df["rental_duration"] = complete_df["rental_duration"].fillna(complete_df["rental_duration"].mean())
complete_df["rate_relative"] = complete_df["rate_relative"].fillna(1.0)
complete_df["recent_maintenance"] = complete_df["recent_maintenance"].fillna(0)
complete_df["maintenance_per_day"] = complete_df["maintenance_per_day"].fillna(0)
complete_df["engine_hours_per_day"] = complete_df["engine_hours_per_day"].fillna(complete_df["engine_hours_per_day"].mean())
complete_df["month_sin"] = complete_df["month_sin"].fillna(0)
complete_df["month_cos"] = complete_df["month_cos"].fillna(0)
complete_df["week_sin"] = complete_df["week_sin"].fillna(0)
complete_df["week_cos"] = complete_df["week_cos"].fillna(0)
complete_df["rolling_std_4w"] = complete_df["rolling_std_4w"].fillna(0)
complete_df["rolling_max_4w"] = complete_df["rolling_max_4w"].fillna(0)

# For categorical features, use one-hot encoding
complete_df = pd.get_dummies(complete_df, columns=["equipment_type"], drop_first=False)

# Determine training and testing sets
# Fill NaN values for is_train and is_test flags
complete_df["is_train"] = complete_df["is_train"].fillna(False)
complete_df["is_test"] = complete_df["is_test"].fillna(False)

# Split the data
train_df = complete_df[complete_df["is_train"] == True].copy()
test_df = complete_df[complete_df["is_test"] == True].copy()

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# print test data value counts
test_df.to_csv("test_data_check.csv", index=False)

# -----------------------------
# Model Training
# -----------------------------

# Define features - exclude non-feature columns
feature_cols = [col for col in train_df.columns if col not in 
                ["year_week", "target_checkout_count", "is_train", "is_test"]]

# Train a Random Forest model for demand forecasting
print("Training model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the model
model.fit(train_df[feature_cols], train_df["target_checkout_count"])

# -----------------------------
# Model Evaluation
# -----------------------------

# Make predictions on the test set
test_df["predicted_demand"] = model.predict(test_df[feature_cols])

# Round predictions to nearest integer (can't have fractional rentals)
test_df["predicted_demand"] = np.round(test_df["predicted_demand"]).clip(0)  # Ensure non-negative

# Calculate metrics
mae = mean_absolute_error(test_df["target_checkout_count"], test_df["predicted_demand"])
rmse = np.sqrt(mean_squared_error(test_df["target_checkout_count"], test_df["predicted_demand"]))
r2 = r2_score(test_df["target_checkout_count"], test_df["predicted_demand"])

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# -----------------------------
# Generate Forecasts for Future Periods (2022)
# -----------------------------

# Reconstruct equipment_type from one-hot encoded columns
def get_original_from_onehot(row, prefix):
    cols = [col for col in row.index if col.startswith(prefix)]
    for col in cols:
        if row[col] == 1:
            return col.replace(prefix, "")
    return None

# Apply function to reconstruct original columns
test_df["equipment_type"] = test_df.apply(lambda row: get_original_from_onehot(row, "equipment_type_"), axis=1)

# Fix any incorrectly reconstructed equipment types
# Make sure we only have valid equipment types
valid_equipment_types = ["Excavator", "Loader", "Bulldozer", "Compactor", "Crane"]
test_df["equipment_type"] = test_df["equipment_type"].apply(lambda x: x if x in valid_equipment_types else None)

# Check if we have any rows with None equipment_type
invalid_types = test_df["equipment_type"].isnull().sum()
if invalid_types > 0:
    print(f"Warning: Found {invalid_types} rows with invalid equipment types. These will be dropped.")
    test_df = test_df.dropna(subset=["equipment_type"])

# Create a forecast DataFrame with the key columns
forecast_df = test_df[["equipment_type", "year_week", "predicted_demand", "target_checkout_count"]].copy()

# Extract year and week for better sorting
forecast_df["year"] = forecast_df["year_week"].str.split("-W").str[0].astype(int)
forecast_df["week_num"] = forecast_df["year_week"].str.split("-W").str[1].astype(int)
forecast_df = forecast_df.sort_values(["equipment_type", "year", "week_num"])

# Save the forecasts
print("\nSaving forecasts...")
forecast_df.to_csv("equipment_demand_forecast.csv", index=False)

# -----------------------------
# Visualization - Individual plots for each equipment type
# -----------------------------

# Set styles for better visualizations
plt.style.use('ggplot')
sns.set_palette("bright")
sns.set_context("talk")

# Print detailed summary of forecasted and actual demand by equipment type
print("\nDemand Summary by Equipment Type:")
summary = forecast_df.groupby("equipment_type").agg({
    "predicted_demand": ["sum", "mean", "count"],
    "target_checkout_count": ["sum", "mean", "count"]
}).round(2)
print(summary)

# Specifically check Compactor and Excavator data
print("\nDetailed data for problematic equipment types:")
for eq_type in ["Compactor", "Excavator"]:
    eq_data = forecast_df[forecast_df["equipment_type"] == eq_type].copy()
    eq_weeks = eq_data[eq_data["target_checkout_count"] > 0]["year_week"].tolist()
    print(f"\n{eq_type} - Weeks with actual demand: {eq_weeks}")
    print(f"{eq_type} - Total actual demand: {eq_data['target_checkout_count'].sum()}")
    print(f"{eq_type} - Data sample:\n{eq_data[['year_week', 'target_checkout_count', 'predicted_demand']].head(3)}")

# Fix any zero values that should have data
for eq_type in ["Compactor", "Excavator"]:
    weeks_with_zero_but_should_have_data = forecast_df[
        (forecast_df["equipment_type"] == eq_type) & 
        (forecast_df["target_checkout_count"] == 0) & 
        (forecast_df["year_week"].str.startswith("2022"))
    ]["year_week"].tolist()
    
    if weeks_with_zero_but_should_have_data:
        print(f"Warning: Found {len(weeks_with_zero_but_should_have_data)} weeks for {eq_type} that should have data but show zero")
        
# Print the actual dataset for verification
print("\nVerifying equipment rentals in 2022:")
test_year_data = df[df["is_test"] == True].copy()
eq_counts = test_year_data.groupby("equipment_type")["equipment_id"].count()
print(eq_counts)

# Check if all equipment types are present in the forecast data
valid_equipment_types = ["Excavator", "Loader", "Bulldozer", "Compactor", "Crane"]
missing_types = set(valid_equipment_types) - set(forecast_df["equipment_type"].unique())
if missing_types:
    print(f"Warning: The following equipment types are missing from the forecast data: {missing_types}")
    print("Adding missing equipment types with zero demand...")
    
    # For each missing equipment type, create placeholder data for all weeks
    existing_weeks = forecast_df["year_week"].unique()
    for eq_type in missing_types:
        for week in existing_weeks:
            # Get the year and week number
            year = week.split("-W")[0]
            week_num = week.split("-W")[1]
            # Add a row for this equipment type and week
            new_row = pd.DataFrame({
                "equipment_type": [eq_type],
                "year_week": [week],
                "predicted_demand": [0],  # No predicted demand
                "target_checkout_count": [0],  # No actual demand
                "year": [int(year)],
                "week_num": [int(week_num)]
            })
            forecast_df = pd.concat([forecast_df, new_row], ignore_index=True)

# Create individual plots for each equipment type
valid_equipment_types = ["Excavator", "Loader", "Bulldozer", "Compactor", "Crane"]
equipment_types = valid_equipment_types  # Use all equipment types

print(f"Creating separate visualizations for equipment types: {equipment_types}")

# 1. First create a combined visualization (all equipment types in one figure)
plt.figure(figsize=(20, 15))

for i, eq_type in enumerate(equipment_types, 1):
    plt.subplot(3, 2, i)  # Layout for 5 plots (3 rows, 2 columns)
    
    # Filter for this equipment type
    eq_data = forecast_df[forecast_df["equipment_type"] == eq_type].copy()
    print(f"- {eq_type}: {len(eq_data)} weeks of data, sum of actual demand: {eq_data['target_checkout_count'].sum():.2f}")
    
    # Create x-axis labels - Week numbers in 2022
    eq_data["x_label"] = eq_data["week_num"].apply(lambda x: f"W{x}")
    
    # Sort by week
    eq_data = eq_data.sort_values("week_num")
    
    # Get the weeks in order
    weeks = eq_data["x_label"].values
    
    # Plot actual vs predicted
    total_actual = eq_data["target_checkout_count"].sum()
    
    # Always plot the actual demand line - even if zero, for consistency
    plt.plot(weeks, eq_data["target_checkout_count"], 'b-', label=f'Actual Demand ({total_actual} units)', linewidth=2)
    
    # Add markers to make the data points more visible
    plt.plot(weeks, eq_data["target_checkout_count"], 'bo', markersize=5)
    
    # Plot predicted demand
    total_predicted = eq_data["predicted_demand"].sum()
    plt.plot(weeks, eq_data["predicted_demand"], 'r-', label=f'Predicted Demand ({total_predicted:.1f} units)', linewidth=2)
    plt.plot(weeks, eq_data["predicted_demand"], 'ro', markersize=4)
    
    # If there's a discrepancy between data and visualization
    if total_actual == 0 and eq_type in ["Compactor", "Excavator"]:
        plt.text(0.5, 0.5, f"Warning: Data shows {eq_type} rentals in 2022\nbut aggregation shows zero", 
                 ha='center', va='center', transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.title(f"{eq_type} - 2022 Weekly Demand", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("Number of Rentals")
    plt.xlabel("Week of 2022")
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig("equipment_type_demand_forecast_2022.png")
print("Combined visualization saved as 'equipment_type_demand_forecast_2022.png'")

# 2. Now create individual visualizations for each equipment type
for eq_type in equipment_types:
    # Create a new figure for each equipment type
    plt.figure(figsize=(16, 10))
    
    # Filter for this equipment type
    eq_data = forecast_df[forecast_df["equipment_type"] == eq_type].copy()
    
    # Create x-axis labels - Week numbers in 2022
    eq_data["x_label"] = eq_data["week_num"].apply(lambda x: f"W{x}")
    
    # Sort by week
    eq_data = eq_data.sort_values("week_num")
    
    # Get the weeks in order
    weeks = eq_data["x_label"].values
    
    # Plot actual demand
    total_actual = eq_data["target_checkout_count"].sum()
    plt.plot(weeks, eq_data["target_checkout_count"], 'b-', marker='o', 
             label=f'Actual Demand ({total_actual} units)', linewidth=2, markersize=6)
    
    # Plot predicted demand
    total_predicted = eq_data["predicted_demand"].sum()
    plt.plot(weeks, eq_data["predicted_demand"], 'r-', marker='o', 
             label=f'Predicted Demand ({total_predicted:.1f} units)', linewidth=2, markersize=5)
    
    # Add value labels for peak points
    max_actual_idx = eq_data["target_checkout_count"].idxmax() if len(eq_data) > 0 else None
    max_pred_idx = eq_data["predicted_demand"].idxmax() if len(eq_data) > 0 else None
    
    if max_actual_idx is not None and eq_data.loc[max_actual_idx, "target_checkout_count"] > 0:
        max_week = eq_data.loc[max_actual_idx, "x_label"]
        max_value = eq_data.loc[max_actual_idx, "target_checkout_count"]
        week_idx = list(weeks).index(max_week)
        plt.annotate(f"{int(max_value)}", 
                    (week_idx, max_value),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    if max_pred_idx is not None and eq_data.loc[max_pred_idx, "predicted_demand"] > 0:
        max_week = eq_data.loc[max_pred_idx, "x_label"]
        max_value = eq_data.loc[max_pred_idx, "predicted_demand"]
        week_idx = list(weeks).index(max_week)
        plt.annotate(f"{int(max_value)}", 
                    (week_idx, max_value),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha='center', color='red')
    
    # Add a title and labels
    plt.title(f"{eq_type} - 2022 Weekly Demand Forecast", fontsize=16)
    plt.xlabel("Week of 2022", fontsize=12)
    plt.ylabel("Number of Rentals", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Improve x-axis readability
    if len(weeks) > 20:
        # Show every 4th week if we have many data points
        plt.xticks(range(0, len(weeks), 4), [weeks[i] for i in range(0, len(weeks), 4)], rotation=45)
    else:
        plt.xticks(rotation=45)
    
    # Add annotations
    mean_actual = eq_data["target_checkout_count"].mean()
    mean_pred = eq_data["predicted_demand"].mean()
    
    plt.axhline(y=mean_actual, color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=mean_pred, color='r', linestyle='--', alpha=0.5)
    
    plt.text(len(weeks)-1, mean_actual, f"Avg: {mean_actual:.1f}", 
             color='blue', ha='right', va='bottom')
    plt.text(len(weeks)-1, mean_pred, f"Avg: {mean_pred:.1f}", 
             color='red', ha='right', va='top')
    
    plt.tight_layout()
    plt.savefig(f"{eq_type}_demand_forecast_2023.png")
    print(f"Saved {eq_type}_demand_forecast_2023.png")

# -----------------------------
# Feature Importance
# -----------------------------

# Check which features are most important for the model
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# Save feature importance plot
plt.figure(figsize=(14, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Feature Importance for Demand Prediction')
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance visualization saved as 'feature_importance.png'")

# Add a visualization to show how the new features contribute
new_features = [
    'equipment_type_encoded', 'usage_efficiency', 'rental_duration',
    'rate_relative', 'recent_maintenance', 'maintenance_per_day',
    'engine_hours_per_day', 'month_sin', 'month_cos', 'week_sin',
    'week_cos', 'rolling_std_4w', 'rolling_max_4w'
]

# Filter to only show new features
new_feature_importance = feature_importance[feature_importance['Feature'].isin(new_features)]
if len(new_feature_importance) > 0:
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=new_feature_importance)
    plt.title('Importance of Newly Added Features')
    plt.tight_layout()
    plt.savefig("new_feature_importance.png")
    print("New feature importance visualization saved as 'new_feature_importance.png'")

print("\nDemand forecasting completed successfully!")
