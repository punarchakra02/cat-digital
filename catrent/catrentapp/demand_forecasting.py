import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Only import matplotlib for non-web usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
else:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def equipment_demand_forecast(df, output_folder='forecast/', train_year_cutoff=2024, test_year=2025):
    """
    Perform equipment demand forecasting and generate visualizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with equipment rental data
    output_folder : str
        Folder to save forecast outputs and visualizations
    train_year_cutoff : int
        Year to use as cutoff for training data (inclusive)
    test_year : int
        Year to forecast/test on
    
    Returns:
    --------
    dict
        Dictionary containing model metrics, forecasts, and feature importance
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print("Starting equipment demand forecasting...")
    
    # -----------------------------
    # Data Preparation
    # -----------------------------
    
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Convert dates 
    df["checkout_date"] = pd.to_datetime(df["checkout_date"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    
    # Get the min and max dates in the dataset
    min_date = df["checkout_date"].min()
    max_date = df["checkout_date"].max()
    print(f"Data spans from {min_date.date()} to {max_date.date()}")
    
    # Create train/test split based on years
    df["is_train"] = df["checkout_date"].dt.year <= train_year_cutoff
    df["is_test"] = df["checkout_date"].dt.year == test_year
    
    # Print data distribution
    year_counts = df["checkout_date"].dt.year.value_counts().sort_index()
    print("\nData distribution by year:")
    print(year_counts)
    
    # -----------------------------
    # Feature Engineering
    # -----------------------------
    
    # Extract temporal features
    df["month"] = df["checkout_date"].dt.month
    df["day_of_week"] = df["checkout_date"].dt.dayofweek
    df["week_of_year"] = df["checkout_date"].dt.isocalendar().week
    
    # -----------------------------
    # Prepare Aggregated Training Data
    # -----------------------------
    
    print("Preparing aggregated training data...")
    
    # Get unique equipment types and weeks
    all_equipment_types = df["equipment_type"].unique()
    all_weeks = sorted(df["year_week"].unique())
    
    # Create all combinations of equipment_type and week
    all_combos = list(product(all_equipment_types, all_weeks))
    all_df = pd.DataFrame(all_combos, columns=["equipment_type", "year_week"])
    
    # Recalculate target checkout counts
    target_counts = df.groupby(["equipment_type", "year_week"])["equipment_id"].count().reset_index(name="recalculated_count")
    
    # Aggregate features by equipment type and week
    feature_columns = [
        "checkout_count_t-1", "checkout_count_t-4", "rolling_mean_4w", "price_index",
        "pct_maint", "idle_ratio_avg", "equipment_type_encoded", "usage_efficiency",
        "rental_duration", "rate_relative", "recent_maintenance", "maintenance_per_day",
        "engine_hours_per_day", "month_sin", "month_cos", "week_sin", "week_cos",
        "rolling_std_4w", "rolling_max_4w", "is_train", "is_test", "month",
        "day_of_week", "week_of_year"
    ]
    
    # Filter to only existing columns
    existing_feature_columns = [col for col in feature_columns if col in df.columns]
    
    agg_dict = {}
    for col in existing_feature_columns:
        if col in ["is_train", "is_test", "month", "day_of_week", "week_of_year", 
                   "equipment_type_encoded", "month_sin", "month_cos", "week_sin", "week_cos",
                   "checkout_count_t-1", "checkout_count_t-4", "rolling_mean_4w", "rolling_std_4w", "rolling_max_4w"]:
            agg_dict[col] = "first"
        else:
            agg_dict[col] = "mean"
    
    agg_df = df.groupby(["equipment_type", "year_week"]).agg(agg_dict).reset_index()
    
    # Merge with recalculated counts
    agg_df = agg_df.merge(target_counts, on=["equipment_type", "year_week"], how="left")
    agg_df["target_checkout_count"] = agg_df["recalculated_count"].fillna(0).astype(int)
    agg_df.drop("recalculated_count", axis=1, inplace=True)
    
    # Merge with all combinations to get complete dataset
    complete_df = all_df.merge(agg_df, on=["equipment_type", "year_week"], how="left")
    
    # Fill missing values
    complete_df["target_checkout_count"] = complete_df["target_checkout_count"].fillna(0)
    
    # Forward-fill lag features by equipment type
    lag_features = ["checkout_count_t-1", "checkout_count_t-4", "rolling_mean_4w"]
    for col in lag_features:
        if col in complete_df.columns:
            complete_df[col] = complete_df.groupby(["equipment_type"])[col].ffill().fillna(0)
    
    # Fill other missing values with appropriate defaults
    fill_values = {
        "price_index": 1.0,
        "pct_maint": 0,
        "idle_ratio_avg": 0,
        "equipment_type_encoded": 0,
        "usage_efficiency": 0.5,
        "rate_relative": 1.0,
        "recent_maintenance": 0,
        "maintenance_per_day": 0,
        "month_sin": 0,
        "month_cos": 0,
        "week_sin": 0,
        "week_cos": 0,
        "rolling_std_4w": 0,
        "rolling_max_4w": 0
    }
    
    for col, value in fill_values.items():
        if col in complete_df.columns:
            complete_df[col] = complete_df[col].fillna(value)
    
    # Fill numeric columns with mean, respecting data types
    numeric_cols = complete_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if complete_df[col].isnull().sum() > 0:
            # Get the data type
            dtype = complete_df[col].dtype
            
            # For integer types, use the rounded mean or zero
            if pd.api.types.is_integer_dtype(dtype):
                # For unsigned integers, ensure we don't use negative values
                if 'uint' in str(dtype).lower() or 'uint' in dtype.name.lower():
                    # Use either zero or the rounded mean, ensuring it's non-negative
                    mean_val = max(0, int(complete_df[col].mean()))
                    complete_df[col] = complete_df[col].fillna(mean_val)
                else:
                    # For regular integers, round the mean
                    mean_val = int(complete_df[col].mean())
                    complete_df[col] = complete_df[col].fillna(mean_val)
            else:
                # For float types, use the mean directly
                complete_df[col] = complete_df[col].fillna(complete_df[col].mean())
    
    # One-hot encode equipment types
    complete_df = pd.get_dummies(complete_df, columns=["equipment_type"], drop_first=False)
    
    # Fill training/testing flags
    complete_df["is_train"] = complete_df["is_train"].fillna(False)
    complete_df["is_test"] = complete_df["is_test"].fillna(False)
    
    # Split data
    train_df = complete_df[complete_df["is_train"] == True].copy()
    test_df = complete_df[complete_df["is_test"] == True].copy()
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    
    if len(train_df) == 0:
        raise ValueError("No training data found. Check your train_year_cutoff parameter.")
    if len(test_df) == 0:
        raise ValueError("No test data found. Check your test_year parameter.")
    
    # -----------------------------
    # Model Training
    # -----------------------------
    
    # Define feature columns
    feature_cols = [col for col in train_df.columns if col not in 
                    ["year_week", "target_checkout_count", "is_train", "is_test"]]
    
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    model.fit(train_df[feature_cols], train_df["target_checkout_count"])
    
    # -----------------------------
    # Model Evaluation
    # -----------------------------
    
    # Make predictions
    test_df["predicted_demand"] = model.predict(test_df[feature_cols])
    test_df["predicted_demand"] = np.round(test_df["predicted_demand"]).clip(0)
    
    # Calculate metrics
    mae = mean_absolute_error(test_df["target_checkout_count"], test_df["predicted_demand"])
    rmse = np.sqrt(mean_squared_error(test_df["target_checkout_count"], test_df["predicted_demand"]))
    r2 = r2_score(test_df["target_checkout_count"], test_df["predicted_demand"])
    
    print(f"\nModel Evaluation:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # -----------------------------
    # Generate Forecasts
    # -----------------------------
    
    # Reconstruct equipment_type from one-hot encoded columns
    def get_original_from_onehot(row, prefix):
        cols = [col for col in row.index if col.startswith(prefix)]
        for col in cols:
            if row[col] == 1:
                return col.replace(prefix, "")
        return None
    
    test_df["equipment_type"] = test_df.apply(lambda row: get_original_from_onehot(row, "equipment_type_"), axis=1)
    
    # Filter valid equipment types
    valid_equipment_types = list(all_equipment_types)
    test_df = test_df[test_df["equipment_type"].isin(valid_equipment_types)]
    
    # Create forecast DataFrame
    forecast_df = test_df[["equipment_type", "year_week", "predicted_demand", "target_checkout_count"]].copy()
    
    # Extract year and week for sorting
    forecast_df["year"] = forecast_df["year_week"].str.split("-W").str[0].astype(int)
    forecast_df["week_num"] = forecast_df["year_week"].str.split("-W").str[1].astype(int)
    forecast_df = forecast_df.sort_values(["equipment_type", "year", "week_num"])
    
    # Save forecasts
    forecast_path = os.path.join(output_folder, "equipment_demand_forecast.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Forecasts saved to {forecast_path}")
    
    # -----------------------------
    # Visualizations
    # -----------------------------
    
    # Set style
    plt.style.use('ggplot')
    sns.set_palette("bright")
    sns.set_context("talk")
    
    # Print summary
    print("\nDemand Summary by Equipment Type:")
    summary = forecast_df.groupby("equipment_type").agg({
        "predicted_demand": ["sum", "mean", "count"],
        "target_checkout_count": ["sum", "mean", "count"]
    }).round(2)
    print(summary)
    
    # 1. Combined visualization for all equipment types
    equipment_types = sorted(valid_equipment_types)
    n_types = len(equipment_types)
    
    # Calculate subplot layout
    n_cols = 2
    n_rows = (n_types + 1) // 2
    
    plt.figure(figsize=(20, 5 * n_rows))
    
    for i, eq_type in enumerate(equipment_types, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # Filter for this equipment type
        eq_data = forecast_df[forecast_df["equipment_type"] == eq_type].copy()
        
        if len(eq_data) == 0:
            plt.text(0.5, 0.5, f"No data for {eq_type}", ha='center', va='center')
            plt.title(f"{eq_type} - No Data")
            continue
        
        # Create x-axis labels
        eq_data["x_label"] = eq_data["week_num"].apply(lambda x: f"W{x}")
        eq_data = eq_data.sort_values("week_num")
        
        weeks = eq_data["x_label"].values
        total_actual = eq_data["target_checkout_count"].sum()
        total_predicted = eq_data["predicted_demand"].sum()
        
        # Plot lines
        plt.plot(weeks, eq_data["target_checkout_count"], 'b-', 
                label=f'Actual ({total_actual} units)', linewidth=2, marker='o', markersize=5)
        plt.plot(weeks, eq_data["predicted_demand"], 'r-', 
                label=f'Predicted ({total_predicted:.1f} units)', linewidth=2, marker='o', markersize=4)
        
        plt.title(f"{eq_type} - {test_year} Weekly Demand", fontsize=14)
        plt.xticks(rotation=45)
        plt.ylabel("Number of Rentals")
        plt.xlabel(f"Week of {test_year}")
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    combined_path = os.path.join(output_folder, f"equipment_demand_forecast_{test_year}_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved to {combined_path}")
    
    # 2. Individual plots for each equipment type
    for eq_type in equipment_types:
        # Set the style for modern, clean charts
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Filter for this equipment type
        eq_data = forecast_df[forecast_df["equipment_type"] == eq_type].copy()
        
        if len(eq_data) == 0:
            plt.text(0.5, 0.5, f"No data available for {eq_type}", 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16, 
                    color='#555555', fontweight='bold')
            plt.title(f"{eq_type} - No Data Available", 
                      fontsize=18, fontweight='bold', color='#333333', pad=20)
        else:
            # Create x-axis labels
            eq_data["x_label"] = eq_data["week_num"].apply(lambda x: f"W{x}")
            eq_data = eq_data.sort_values("week_num")
            
            weeks = eq_data["x_label"].values
            total_actual = eq_data["target_checkout_count"].sum()
            total_predicted = eq_data["predicted_demand"].sum()
            
            # Modern color palette
            actual_color = '#3498db'  # Blue
            predicted_color = '#e74c3c'  # Red
            
            # Create gradient fill under lines
            x = np.arange(len(weeks))
            ax.fill_between(x, 0, eq_data["target_checkout_count"], 
                           color=actual_color, alpha=0.2)
            ax.fill_between(x, 0, eq_data["predicted_demand"], 
                           color=predicted_color, alpha=0.1)
            
            # Plot lines with smoother curves
            ax.plot(x, eq_data["target_checkout_count"], 
                   color=actual_color, 
                   label=f'Actual Demand ({total_actual} units)', 
                   linewidth=3, marker='o', markersize=8, 
                   markerfacecolor='white', markeredgewidth=2, markeredgecolor=actual_color)
            
            ax.plot(x, eq_data["predicted_demand"], 
                   color=predicted_color, 
                   label=f'Predicted Demand ({total_predicted:.1f} units)', 
                   linewidth=3, marker='o', markersize=7, 
                   markerfacecolor='white', markeredgewidth=2, markeredgecolor=predicted_color)
            
            # Add average lines with gradient effect
            mean_actual = eq_data["target_checkout_count"].mean()
            mean_pred = eq_data["predicted_demand"].mean()
            
            # Create a gradient effect for average lines
            for i, alpha in zip(range(3), [0.1, 0.2, 0.5]):
                ax.axhline(y=mean_actual-i*0.05, color=actual_color, linestyle='-', alpha=alpha, linewidth=1)
                ax.axhline(y=mean_pred-i*0.05, color=predicted_color, linestyle='-', alpha=alpha, linewidth=1)
            
            ax.axhline(y=mean_actual, color=actual_color, linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(y=mean_pred, color=predicted_color, linestyle='--', alpha=0.7, linewidth=2)
            
            # Add text annotations for averages with better styling
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=actual_color, alpha=0.8)
            ax.text(len(weeks)-1, mean_actual+0.5, f"Avg: {mean_actual:.1f}", 
                   color=actual_color, ha='right', va='bottom', 
                   fontsize=12, fontweight='bold', bbox=bbox_props)
            
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=predicted_color, alpha=0.8)
            ax.text(len(weeks)-1, mean_pred-0.5, f"Avg: {mean_pred:.1f}", 
                   color=predicted_color, ha='right', va='top', 
                   fontsize=12, fontweight='bold', bbox=bbox_props)
            
            # Improve x-axis readability
            if len(weeks) > 20:
                ax.set_xticks(range(0, len(weeks), 4))
                ax.set_xticklabels([weeks[i] for i in range(0, len(weeks), 4)], rotation=45)
            else:
                ax.set_xticks(range(len(weeks)))
                ax.set_xticklabels(weeks, rotation=45)
            
            # Adjust y-axis to start from 0 with a small padding
            y_max = max(eq_data["target_checkout_count"].max(), eq_data["predicted_demand"].max()) * 1.15
            ax.set_ylim(0, y_max)
            
            # Add data labels for key points
            highest_actual_idx = eq_data["target_checkout_count"].idxmax()
            highest_actual_week = eq_data.loc[highest_actual_idx, "x_label"]
            highest_actual_value = eq_data.loc[highest_actual_idx, "target_checkout_count"]
            
            highest_pred_idx = eq_data["predicted_demand"].idxmax()
            highest_pred_week = eq_data.loc[highest_pred_idx, "x_label"]
            highest_pred_value = eq_data.loc[highest_pred_idx, "predicted_demand"]
            
            ax.annotate(f"Peak: {highest_actual_value}", 
                       xy=(eq_data.index.get_loc(highest_actual_idx), highest_actual_value),
                       xytext=(10, 20), textcoords="offset points",
                       arrowprops=dict(arrowstyle="->", color=actual_color),
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=actual_color, alpha=0.8),
                       color=actual_color, fontweight='bold')
            
            ax.annotate(f"Peak: {highest_pred_value:.1f}", 
                       xy=(eq_data.index.get_loc(highest_pred_idx), highest_pred_value),
                       xytext=(-10, -20), textcoords="offset points",
                       arrowprops=dict(arrowstyle="->", color=predicted_color),
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=predicted_color, alpha=0.8),
                       color=predicted_color, fontweight='bold')
        
        # Improve title and labels
        plt.title(f"{eq_type} Demand Forecast - {test_year}", 
                 fontsize=20, fontweight='bold', color='#333333', pad=20)
        plt.xlabel(f"Week of {test_year}", fontsize=14, fontweight='semibold', labelpad=15)
        plt.ylabel("Number of Rentals", fontsize=14, fontweight='semibold', labelpad=15)
        
        # Add a subtle grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Enhance the legend
        legend = plt.legend(fontsize=12, frameon=True, facecolor='white', 
                           edgecolor='#dddddd', shadow=True, 
                           title=f"Demand for {eq_type}", title_fontsize=14)
        
        # Add a subtle border to the plot
        for spine in ax.spines.values():
            spine.set_color('#dddddd')
            spine.set_linewidth(1)
        
        # Add a watermark
        fig.text(0.99, 0.01, "CatRent Analytics", 
                 fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
        
        # Add timestamp
        current_time = datetime.now().strftime("%Y-%m-%d")
        fig.text(0.01, 0.01, f"Generated: {current_time}", 
                 fontsize=8, color='gray', ha='left', va='bottom', alpha=0.7)
        
        plt.tight_layout()
        
        individual_path = os.path.join(output_folder, f"{eq_type}_demand_forecast_{test_year}.png")
        plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved {individual_path}")
    
    # 3. Feature Importance Visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Set a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a custom color palette (blue gradient)
    n_features = min(20, len(feature_importance))
    palette = sns.color_palette("Blues_d", n_features)
    
    # Save feature importance plot with modern styling
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot the bars with a gradient
    bars = sns.barplot(x='Importance', y='Feature', data=feature_importance.head(n_features), 
                      palette=palette, ax=ax)
    
    # Add value labels to the bars
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        ax.text(width + 0.002, p.get_y() + p.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', 
                fontweight='bold', color='#555555')
    
    # Add percentage labels
    total_importance = feature_importance['Importance'].sum()
    for i, p in enumerate(bars.patches):
        width = p.get_width()
        percentage = (width / total_importance) * 100
        ax.text(0.01, p.get_y() + p.get_height()/2, 
                f'{percentage:.1f}%', ha='left', va='center', 
                fontweight='bold', color='white')
    
    # Customize the plot
    plt.title('Key Factors Influencing Equipment Demand', 
             fontsize=18, fontweight='bold', color='#333333', pad=20)
    plt.xlabel('Relative Importance', fontsize=14, fontweight='semibold', labelpad=15)
    plt.ylabel('', fontsize=14)  # Remove y-label as it's redundant
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make the left and bottom spines subtle
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    
    # Add a subtle grid for the x-axis only
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add a title for the y-axis on the plot itself
    fig.text(0.01, 0.5, 'Feature Name', 
             fontsize=14, fontweight='semibold', 
             rotation=90, ha='center', va='center')
    
    # Add an explanatory note
    fig.text(0.5, 0.01, 
             'Higher values indicate greater influence on demand forecasting accuracy',
             fontsize=11, ha='center', va='bottom', style='italic', alpha=0.7)
    
    # Add a watermark
    fig.text(0.99, 0.01, "CatRent Analytics", 
             fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    
    # Add timestamp
    current_time = datetime.now().strftime("%Y-%m-%d")
    fig.text(0.01, 0.01, f"Generated: {current_time}", 
             fontsize=8, color='gray', ha='left', va='bottom', alpha=0.7)
    
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])  # Adjust layout to make room for annotations
    
    importance_path = os.path.join(output_folder, "feature_importance.png")
    plt.savefig(importance_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Feature importance visualization saved to {importance_path}")
    
    # 4. New Features Importance (if applicable)
    new_features = [
        'equipment_type_encoded', 'usage_efficiency', 'rental_duration',
        'rate_relative', 'recent_maintenance', 'maintenance_per_day',
        'engine_hours_per_day', 'month_sin', 'month_cos', 'week_sin',
        'week_cos', 'rolling_std_4w', 'rolling_max_4w'
    ]
    
    new_feature_importance = feature_importance[feature_importance['Feature'].isin(new_features)]
    if len(new_feature_importance) > 0:
        # Create a modern visualization for new features importance
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create a custom color palette (orange/gold gradient for distinction from main chart)
        n_new_features = len(new_feature_importance)
        palette = sns.color_palette("YlOrBr", n_new_features)
        
        # Plot the bars with a gradient
        bars = sns.barplot(x='Importance', y='Feature', data=new_feature_importance, 
                          palette=palette, ax=ax)
        
        # Add value labels to the bars
        for i, p in enumerate(bars.patches):
            width = p.get_width()
            ax.text(width + 0.002, p.get_y() + p.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', 
                    fontweight='bold', color='#555555')
        
        # Add percentage labels
        total_importance = feature_importance['Importance'].sum()
        for i, p in enumerate(bars.patches):
            width = p.get_width()
            percentage = (width / total_importance) * 100
            ax.text(0.01, p.get_y() + p.get_height()/2, 
                    f'{percentage:.1f}%', ha='left', va='center', 
                    fontweight='bold', color='white')
        
        # Customize the plot
        plt.title('Impact of Newly Added Features on Demand Prediction', 
                 fontsize=18, fontweight='bold', color='#333333', pad=20)
        plt.xlabel('Relative Importance', fontsize=14, fontweight='semibold', labelpad=15)
        plt.ylabel('', fontsize=14)  # Remove y-label as it's redundant
        
        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make the left and bottom spines subtle
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#dddddd')
        
        # Add a subtle grid for the x-axis only
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add a title for the y-axis on the plot itself
        fig.text(0.01, 0.5, 'New Feature', 
                 fontsize=14, fontweight='semibold', 
                 rotation=90, ha='center', va='center')
        
        # Add an explanatory note
        fig.text(0.5, 0.01, 
                 'These newly added features were designed to improve forecasting accuracy',
                 fontsize=11, ha='center', va='bottom', style='italic', alpha=0.7)
        
        # Add a watermark
        fig.text(0.99, 0.01, "CatRent Analytics", 
                 fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
        
        # Add timestamp
        current_time = datetime.now().strftime("%Y-%m-%d")
        fig.text(0.01, 0.01, f"Generated: {current_time}", 
                 fontsize=8, color='gray', ha='left', va='bottom', alpha=0.7)
        
        plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])  # Adjust layout to make room for annotations
        
        new_importance_path = os.path.join(output_folder, "new_feature_importance.png")
        plt.savefig(new_importance_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"New feature importance visualization saved to {new_importance_path}")
    
    # -----------------------------
    # Return Results
    # -----------------------------
    
    results = {
        'model': model,
        'forecast_data': forecast_df,
        'feature_importance': feature_importance,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'summary': summary,
        'output_folder': output_folder
    }
    
    print(f"\nDemand forecasting completed successfully!")
    print(f"All outputs saved to: {output_folder}")
    
    return results

# Example usage:
if __name__ == "__main__":
    # Example of how to use the function
    # df = pd.read_csv("equipment_rentals_with_features.csv")
    # results = equipment_demand_forecast(df, output_folder='forecast/')
    
    print("Equipment demand forecasting function is ready to use!")
    print("Usage: results = equipment_demand_forecast(df, output_folder='forecast/')")