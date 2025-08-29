import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters
n_rows = 2000
equipment_types = ["Excavator", "Loader", "Bulldozer", "Crane"]
sites = ["S001", "S002", "S003", "S004"]
start_date = datetime(2025, 8, 28)

# Create 5 fixed equipment IDs for each equipment type
equipment_ids = {
    "Excavator": [f"EX{random.randint(1000,9999)}" for _ in range(5)],
    "Loader": [f"LO{random.randint(1000,9999)}" for _ in range(5)],
    "Bulldozer": [f"BU{random.randint(1000,9999)}" for _ in range(5)],
    "Crane": [f"CR{random.randint(1000,9999)}" for _ in range(5)]
}

# Create fixed rate_per_day for each equipment ID
rates_per_day = {}
for eq_type, ids in equipment_ids.items():
    base_prices = [250, 300, 450, 500, 800]  # possible base prices
    random.shuffle(base_prices)  # shuffle to randomize assignment
    rates_per_day.update({eq_id: price for eq_id, price in zip(ids, base_prices[:len(ids)])})

# Generate random data
data = []
for i in range(n_rows):
    # Week calculation (reverse chronological order)
    checkout_date = start_date - timedelta(days=i * random.randint(1, 4))  # rentals every few days
    duration = random.randint(3, 10)  # rental period length
    checkin_date = checkout_date + timedelta(days=duration)
    
    site = random.choice(sites)
    eq_type = random.choice(equipment_types)
    eq_id = random.choice(equipment_ids[eq_type])
    
    rate_per_day = rates_per_day[eq_id]  # Get the fixed rate for this equipment ID
    realized_price = round(rate_per_day * random.uniform(0.9, 1.1), 2)  # ±10%
    
    # Calculate engine hours and idle hours based on rental duration
    rental_duration_days = (checkin_date - checkout_date).days
    
    # Assume equipment can be used max 8-12 hours per day
    max_hours_per_day = random.uniform(8, 12)
    
    # Total possible hours during the rental period
    total_possible_hours = rental_duration_days * max_hours_per_day
    
    # Calculate engine hours as a percentage of total possible hours (40-80%)
    engine_utilization = random.uniform(0.4, 0.8)
    engine_hours = round(total_possible_hours * engine_utilization)
    
    # Calculate idle hours as a percentage of engine hours (10-30%)
    idle_ratio = random.uniform(0.1, 0.3)
    idle_hours = round(engine_hours * idle_ratio)
    
    # Ensure minimum values
    engine_hours = max(engine_hours, 1)
    idle_hours = max(idle_hours, 0)
    
    # Random maintenance count (0-2 per week, influenced by engine hours)
    # More engine hours should correlate with more maintenance
    maintenance_probability = min(0.8, engine_hours / (total_possible_hours * 1.5))
    maintenance_count = random.choices(
        [0, 1, 2, 3], 
        weights=[1 - maintenance_probability, maintenance_probability * 0.5, maintenance_probability * 0.3, maintenance_probability * 0.2],
        k=1
    )[0]
    
    week = f"{checkout_date.isocalendar()[0]}-W{checkout_date.isocalendar()[1]}"
    
    data.append([site, eq_type, week, checkout_date.date(), checkin_date.date(),
                 eq_id, rate_per_day, maintenance_count, engine_hours, idle_hours])

# Create dataframe
columns = ["site_id", "equipment_type", "week", "checkout_date", "checkin_date",
           "equipment_id", "rate_per_day", "maintenance_count", "engine_hours", "idle_hours"]
df = pd.DataFrame(data, columns=columns)

# Sort in reverse chronological order (latest on top)
df = df.sort_values(by="checkout_date", ascending=False).reset_index(drop=True)

df.to_csv("equipment_rentals.csv", index=False)
