import pandas as pd
import numpy as np

df = pd.read_csv("equipment_rentals.csv")

# Convert dates
df["checkout_date"] = pd.to_datetime(df["checkout_date"])
df["checkin_date"] = pd.to_datetime(df["checkin_date"])

# Add numeric week (for lag features)
df["year_week"] = df["checkout_date"].dt.strftime("%G-W%V")  # ISO week

# -------------------------
# DERIVED FEATURES
# -------------------------

# target_checkout_count = number of rentals in that type+week
df["target_checkout_count"] = df.groupby(["equipment_type","year_week"])["equipment_id"].transform("count")

# checkout_count_t-1 and t-4
# Map year-week to numeric index
df["week_num"] = df["checkout_date"].dt.isocalendar().week
df["year"] = df["checkout_date"].dt.year

# Helper key
df["eq_key"] = df["equipment_id"] + "_" + df["equipment_type"]

# Build lookup table of counts
weekly_counts = df.groupby(["eq_key","year","week_num"]).size().reset_index(name="count")

# Merge lag features
for lag in [1,4]:
    weekly_counts[f"checkout_count_t-{lag}"] = weekly_counts.groupby("eq_key")["count"].shift(lag)

# Merge back
df = df.merge(weekly_counts, how="left",
              left_on=["eq_key","year","week_num"],
              right_on=["eq_key","year","week_num"])

# rolling mean of last 4 weeks
weekly_counts["rolling_mean_4w"] = weekly_counts.groupby("eq_key")["count"].transform(lambda x: x.rolling(4,min_periods=1).mean())
df = df.merge(weekly_counts[["eq_key","year","week_num","rolling_mean_4w"]],
              on=["eq_key","year","week_num"], how="left")

# price_index = rate_per_day / average historical rate
df["price_index"] = df["rate_per_day"] / df.groupby(["equipment_type"])["rate_per_day"].transform("mean")

# pct_maint = maintenance_count / total rentals that week
df["pct_maint"] = df["maintenance_count"] / df["target_checkout_count"]

# engine_hours_avg = engine_hours / target_checkout_count
df["engine_hours_avg"] = df["engine_hours"] / df["target_checkout_count"]

# idle_ratio_avg = idle_hours / (engine_hours + idle_hours)
df["idle_ratio_avg"] = df["idle_hours"] / (df["engine_hours"] + df["idle_hours"])

# -------------------------
# ADDITIONAL DERIVED FEATURES
# -------------------------

# Equipment-specific features
df['equipment_type_encoded'] = pd.Categorical(df['equipment_type']).codes

# Usage efficiency (active vs. idle time)
df['usage_efficiency'] = (df['engine_hours'] - df['idle_hours']) / df['engine_hours'].clip(lower=1)

# Rental duration in days
df['rental_duration'] = (df['checkin_date'] - df['checkout_date']).dt.days

# Rate per day relative to equipment type average
df['rate_relative'] = df['rate_per_day'] / df.groupby('equipment_type')['rate_per_day'].transform('mean')

# Maintenance intensity
df['recent_maintenance'] = (df['maintenance_count'] > 0).astype(int)
df['maintenance_per_day'] = df['maintenance_count'] / df['rental_duration'].clip(lower=1)

# Engine hours per day
df['engine_hours_per_day'] = df['engine_hours'] / df['rental_duration'].clip(lower=1)

# Seasonal components using cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['checkout_date'].dt.month/12)
df['month_cos'] = np.cos(2 * np.pi * df['checkout_date'].dt.month/12)
df['week_sin'] = np.sin(2 * np.pi * df['week_num']/52)
df['week_cos'] = np.cos(2 * np.pi * df['week_num']/52)

# Rolling statistics for demand volatility
equipment_weekly = df.groupby(['equipment_type', 'year', 'week_num'])['equipment_id'].count().reset_index(name='weekly_count')
equipment_weekly['rolling_std_4w'] = equipment_weekly.groupby('equipment_type')['weekly_count'].transform(lambda x: x.rolling(4, min_periods=1).std())
equipment_weekly['rolling_max_4w'] = equipment_weekly.groupby('equipment_type')['weekly_count'].transform(lambda x: x.rolling(4, min_periods=1).max())

# Merge rolling stats back to main dataframe
df = df.merge(
    equipment_weekly[['equipment_type', 'year', 'week_num', 'rolling_std_4w', 'rolling_max_4w']], 
    on=['equipment_type', 'year', 'week_num'], 
    how='left'
)

print(df[["equipment_id","equipment_type","year_week","target_checkout_count",
          "checkout_count_t-1","checkout_count_t-4","rolling_mean_4w",
          "price_index","pct_maint","engine_hours_avg","idle_ratio_avg",
          "equipment_type_encoded", "usage_efficiency", "rental_duration",
          "rate_relative", "recent_maintenance", "month_sin", "month_cos"]])

df.to_csv("equipment_rentals_with_features.csv", index=False)