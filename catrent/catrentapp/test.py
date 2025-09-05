import pandas as pd
import numpy as np
import random
import os
import sys
import django

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'catrent.settings')
django.setup()

from django.utils import timezone
from catrentapp.models import Machine, Rental

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("catrentapp/equipment_rentals_with_features.csv")

# only the data which has start_date less than feb 15 2025
df = df[df["checkout_date"] < "2023-12-23"]
print(f"Starting upload from filtered CSV. Total rows to process: {len(df)}")

required_columns = ['equipment_id', 'checkout_date', 'checkin_date', 'site_id', 'rolling_std_4w', 'rolling_max_4w']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    exit(1)


# Ensure datetimes
df["checkout_date"] = pd.to_datetime(df["checkout_date"])
df["checkin_date"] = pd.to_datetime(df["checkin_date"])

# -------------------------
# SIMULATE ENGINE + MAINTENANCE
# -------------------------
def simulate_engine_idle(row):
    rental_duration_days = (row["checkin_date"] - row["checkout_date"]).days
    rental_duration_days = max(rental_duration_days, 1)

    max_hours_per_day = random.uniform(8, 12)
    total_possible_hours = rental_duration_days * max_hours_per_day

    engine_utilization = random.uniform(0.4, 0.8)
    engine_hours = round(total_possible_hours * engine_utilization)

    idle_ratio = random.uniform(0.1, 0.3)
    idle_hours = round(engine_hours * idle_ratio)

    return engine_hours, idle_hours, rental_duration_days

df[["engine_hours", "idle_hours", "rental_duration"]] = df.apply(
    lambda row: pd.Series(simulate_engine_idle(row)), axis=1
)
df["maintenance_count"] = np.random.randint(0, 5, size=len(df))

# -------------------------
# DERIVED FEATURES
# -------------------------
df["year_week"] = df["checkout_date"].dt.strftime("%Y-W%V")
df["week_num"] = df["checkout_date"].dt.isocalendar().week
df["year"] = df["checkout_date"].dt.year
df["eq_key"] = df["equipment_id"] + "_" + df["equipment_type"]

# target_checkout_count
df["target_checkout_count"] = df.groupby(["equipment_type","year_week"])["equipment_id"].transform("count")

# checkout_count_t-1, t-4
weekly_counts = df.groupby(["eq_key","year","week_num"]).size().reset_index(name="count")
for lag in [1,4]:
    weekly_counts[f"checkout_count_t-{lag}"] = weekly_counts.groupby("eq_key")["count"].shift(lag)
df = df.merge(weekly_counts, how="left", on=["eq_key","year","week_num"])

# rolling mean of last 4 weeks
weekly_counts["rolling_mean_4w"] = weekly_counts.groupby("eq_key")["count"].transform(lambda x: x.rolling(4,min_periods=1).mean())
df = df.merge(weekly_counts[["eq_key","year","week_num","rolling_mean_4w"]],
              on=["eq_key","year","week_num"], how="left")

# price_index
df["rate_per_day"] = df["rate_per_day"].astype(float)
df["price_index"] = df["rate_per_day"] / df.groupby("equipment_type")["rate_per_day"].transform("mean")

# pct_maint
df["pct_maint"] = df["maintenance_count"] / df["target_checkout_count"]

# engine_hours_avg
df["engine_hours_avg"] = df["engine_hours"] / df["target_checkout_count"]

# idle_ratio_avg
df["idle_ratio_avg"] = df["idle_hours"] / (df["engine_hours"] + df["idle_hours"])

# equipment type encoding
df['equipment_type_encoded'] = pd.Categorical(df['equipment_type']).codes

# usage efficiency
df['usage_efficiency'] = (df['engine_hours'] - df['idle_hours']) / df['engine_hours'].clip(lower=1)

# rate relative
df['rate_relative'] = df['rate_per_day'] / df.groupby('equipment_type')['rate_per_day'].transform('mean')

# maintenance intensity
df['recent_maintenance'] = (df['maintenance_count'] > 0).astype(int)
df['maintenance_per_day'] = df['maintenance_count'] / df['rental_duration'].clip(lower=1)

# engine hours per day
df['engine_hours_per_day'] = df['engine_hours'] / df['rental_duration'].clip(lower=1)

# seasonal
df['month_sin'] = np.sin(2 * np.pi * df['checkout_date'].dt.month/12)
df['month_cos'] = np.cos(2 * np.pi * df['checkout_date'].dt.month/12)
df['week_sin'] = np.sin(2 * np.pi * df['week_num']/52)
df['week_cos'] = np.cos(2 * np.pi * df['week_num']/52)

# Note: We're using rolling_std_4w and rolling_max_4w directly from the CSV file

# -------------------------
# UPLOAD TO RENTAL TABLE
# -------------------------
for _, row in df.iterrows():
    try:
        machine = Machine.objects.get(equipment_id=row["equipment_id"])
    except Machine.DoesNotExist:
        print(f"⚠️ Machine {row['equipment_id']} missing, skipping.")
        continue

    # Make both dates timezone-aware with the same timezone
    start_date = pd.to_datetime(row["checkout_date"])
    end_date = pd.to_datetime(row["checkin_date"])

    # Option 1: Make both timezone-naive (remove timezone info)
    if hasattr(start_date, 'tz') and start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if hasattr(end_date, 'tz') and end_date.tz is not None:
        end_date = end_date.tz_localize(None)

    operators = {
        'OPR001': "Girendra Singh",
        'OPR002': "Vijay",
        'OPR003': "Ayush Mishra"
    }

    operator_id, operator_name = random.choice(list(operators.items()))

    rental, created = Rental.objects.update_or_create(
        machine=machine,
        start_date=row["checkout_date"],  # key
        defaults={
            "site_id": row["site_id"],
            "expected_end_date": row["checkin_date"],
            "rate_per_day": machine.rate_per_day,
            "week": row["year_week"],
            "maintenance": row["maintenance_count"],
            "engine_hours": row["engine_hours"],
            "idle_hours": row["idle_hours"],
            "operator_id": operator_id,
            "operator_name": operator_name,
            "status": "Completed" if row['checkin_date'] < pd.Timestamp.now() else "Active",
            "active": False if row['checkin_date'] < pd.Timestamp.now() else True,

            # Derived features
            "target_checkout_count": row["target_checkout_count"],
            "checkout_count_t_1": row.get("checkout_count_t-1"),
            "checkout_count_t_4": row.get("checkout_count_t-4"),
            "rolling_mean_4w": row.get("rolling_mean_4w"),
            "price_index": row["price_index"],
            "pct_maint": row["pct_maint"],
            "engine_hours_avg": row["engine_hours_avg"],
            "idle_ratio_avg": row["idle_ratio_avg"],
            "equipment_type_encoded": row["equipment_type_encoded"],
            "usage_efficiency": row["usage_efficiency"],
            "rental_duration": row["rental_duration"],
            "rate_relative": row["rate_relative"],
            "recent_maintenance": bool(row["recent_maintenance"]),
            "maintenance_per_day": row["maintenance_per_day"],
            "engine_hours_per_day": row["engine_hours_per_day"],
            "month_sin": row["month_sin"],
            "month_cos": row["month_cos"],
            "week_sin": row["week_sin"],
            "week_cos": row["week_cos"],
            "rolling_std_4w": row["rolling_std_4w"],
            "rolling_max_4w": row["rolling_max_4w"],
        }
    )
    print(f"{'✅ Created' if created else '🔄 Updated'} Rental {rental.rental_id} for {machine.equipment_id}")

