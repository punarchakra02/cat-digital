import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set styling for better visualizations
plt.style.use('ggplot')
sns.set_palette("bright")
sns.set_context("talk")

def main():
    # Load the dataset
    print("Loading equipment rentals data...")
    try:
        df = pd.read_csv("equipment_rentals_with_features.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Loaded {len(df)} records")
    
    # Ensure dates are in datetime format
    df["checkout_date"] = pd.to_datetime(df["checkout_date"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    
    # Extract year and week
    df["year"] = df["checkout_date"].dt.year
    
    # Print dataset overview
    print("\nDataset Overview:")
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")
    print(f"Equipment types: {sorted(df['equipment_type'].unique())}")
    
    # Count rentals by equipment type
    eq_counts = df.groupby("equipment_type")["equipment_id"].count().sort_values(ascending=False)
    print("\nRental counts by equipment type:")
    print(eq_counts)
    
    # Extract week information from year_week column
    if "year_week" in df.columns:
        # If year_week column exists, use it
        df["week_num"] = df["year_week"].str.split("-W").str[1].astype(int)
    else:
        # Otherwise use ISO calendar week
        df["week_num"] = df["checkout_date"].dt.isocalendar().week
    
    # Create visualization 1: Weekly equipment demand across all years
    plot_weekly_demand(df)
    
    # Create visualization 2: Weekly equipment demand for each year
    years = sorted(df["year"].unique())
    for year in years:
        plot_weekly_demand(df[df["year"] == year], year=year)
        
    # Create visualization 3: Monthly equipment demand trends
    plot_monthly_demand(df)
    
    # Create visualization 4: Equipment type distribution
    plot_equipment_distribution(df)
    
    print("Visualizations completed successfully!")

def plot_weekly_demand(df, year=None):
    """
    Plot weekly demand for each equipment type
    """
    title_suffix = f"in {year}" if year else "across all years"
    plt.figure(figsize=(16, 10))
    
    # Group by equipment_type and week_num, count equipment_id
    weekly_demand = df.groupby(["equipment_type", "week_num"])["equipment_id"].count().reset_index(name="demand")
    
    # Pivot to get equipment types as columns
    weekly_demand_pivot = weekly_demand.pivot(index="week_num", columns="equipment_type", values="demand")
    weekly_demand_pivot = weekly_demand_pivot.fillna(0)
    
    eqtypes = ["Compactor"]
    for eq_type in eqtypes:
        if eq_type in weekly_demand_pivot.columns:
            plt.plot(weekly_demand_pivot.index, weekly_demand_pivot[eq_type], marker='o', label=eq_type, linewidth=2)
    
    plt.title(f"Weekly Equipment Demand {title_suffix}")
    plt.xlabel("Week Number")
    plt.ylabel("Number of Rentals")
    plt.legend(title="Equipment Type")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 53, 4))  # Show every 4th week
    
    # Add value labels to important points
    eqtypes = ["Compactor"]
    for eq_type in eqtypes:
        if eq_type in weekly_demand_pivot.columns:
            max_week = weekly_demand_pivot[eq_type].idxmax()
            max_value = weekly_demand_pivot[eq_type].max()
            if max_value > 0:  # Only label if there's demand
                plt.annotate(f"{int(max_value)}", 
                            (max_week, max_value),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center')
    
    filename = f"weekly_demand_{year}.png" if year else "weekly_demand_all_years.png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def plot_monthly_demand(df):
    """
    Plot monthly demand trends for each equipment type
    """
    plt.figure(figsize=(16, 10))
    
    # Extract month
    df["month"] = df["checkout_date"].dt.month
    
    # Group by equipment_type and month, count equipment_id
    monthly_demand = df.groupby(["equipment_type", "month"])["equipment_id"].count().reset_index(name="demand")
    
    # Pivot to get equipment types as columns
    monthly_demand_pivot = monthly_demand.pivot(index="month", columns="equipment_type", values="demand")
    monthly_demand_pivot = monthly_demand_pivot.fillna(0)
    
    # Plot each equipment type
    eqtypes = ["Compactor"]
    for eq_type in eqtypes:
        if eq_type in monthly_demand_pivot.columns:
            plt.plot(monthly_demand_pivot.index, monthly_demand_pivot[eq_type], marker='o', label=eq_type, linewidth=2)
    
    plt.title(f"Monthly Equipment Demand Trends")
    plt.xlabel("Month")
    plt.ylabel("Number of Rentals")
    plt.legend(title="Equipment Type")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    
    plt.tight_layout()
    plt.savefig("monthly_demand_trends.png")
    print("Saved monthly_demand_trends.png")

def plot_equipment_distribution(df):
    """
    Plot the distribution of equipment types
    """
    plt.figure(figsize=(12, 8))
    
    # Count rentals by equipment type
    eq_counts = df.groupby("equipment_type")["equipment_id"].count().sort_values(ascending=False)
    
    # Create bar chart
    ax = eq_counts.plot(kind='bar', color=sns.color_palette("bright"))
    plt.title("Equipment Type Distribution")
    plt.xlabel("Equipment Type")
    plt.ylabel("Number of Rentals")
    
    # Add value labels on top of each bar
    for i, v in enumerate(eq_counts):
        ax.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig("equipment_distribution.png")
    print("Saved equipment_distribution.png")

# Check for specific year (2022) data
def analyze_2022_data(df):
    """
    Special analysis for 2022 data
    """
    df_2022 = df[df["year"] == 2022].copy()
    if len(df_2022) == 0:
        print("No data found for 2022")
        return
        
    print(f"\nAnalyzing 2022 data ({len(df_2022)} records):")
    
    # Count by equipment type
    eq_counts_2022 = df_2022.groupby("equipment_type")["equipment_id"].count().sort_values(ascending=False)
    print("2022 rental counts by equipment type:")
    print(eq_counts_2022)
    
    # Weekly counts for 2022
    plt.figure(figsize=(16, 10))
    weekly_2022 = df_2022.groupby(["equipment_type", "week_num"])["equipment_id"].count().reset_index(name="demand")
    weekly_2022_pivot = weekly_2022.pivot(index="week_num", columns="equipment_type", values="demand")
    weekly_2022_pivot = weekly_2022_pivot.fillna(0)
    
    for eq_type in sorted(df_2022["equipment_type"].unique()):
        if eq_type in weekly_2022_pivot.columns:
            plt.plot(weekly_2022_pivot.index, weekly_2022_pivot[eq_type], marker='o', label=eq_type, linewidth=2)
    
    plt.title("Weekly Equipment Demand in 2022 (Raw Data)")
    plt.xlabel("Week Number")
    plt.ylabel("Number of Rentals")
    plt.legend(title="Equipment Type")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 53, 4))
    
    plt.tight_layout()
    plt.savefig("weekly_demand_2022_raw.png")
    print("Saved weekly_demand_2022_raw.png")

if __name__ == "__main__":
    main()
    
    # Load the data again to specifically analyze 2022
    try:
        df = pd.read_csv("equipment_rentals_with_features.csv")
        df["checkout_date"] = pd.to_datetime(df["checkout_date"])
        df["year"] = df["checkout_date"].dt.year
        analyze_2022_data(df)
    except Exception as e:
        print(f"Error during 2022 analysis: {e}")
