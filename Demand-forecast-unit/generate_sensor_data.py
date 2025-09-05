import pandas as pd
import numpy as np
import time
import datetime
import os
from datetime import datetime, timedelta

# Dictionary to store previous sensor values for each equipment
previous_values = {}

# Function to generate realistic sensor data with some continuity between readings
def generate_sensor_data(equipment_id, timestamp=None):
    global previous_values
    
    # Create a unique key for this equipment at this time period
    # This helps maintain some consistency in sensor values over short periods
    time_key = f"{equipment_id}_{timestamp.strftime('%Y%m%d_%H') if timestamp else 'default'}"
    
    # If we have previous values for this equipment in this time period, use them as a base
    if time_key in previous_values:
        prev = previous_values[time_key]
        
        # Generate new values with small random changes from previous values
        sensor_data = {
            'air_filter_pressure': max(80, min(120, prev['air_filter_pressure'] + np.random.uniform(-2, 2))),
            'exhaust_gas_temp': max(250, min(450, prev['exhaust_gas_temp'] + np.random.uniform(-5, 5))),
            'hydraulic_pump_rate': max(20, min(50, prev['hydraulic_pump_rate'] + np.random.uniform(-1, 1))),
            'oil_pressure': max(30, min(70, prev['oil_pressure'] + np.random.uniform(-1.5, 1.5))),
            'pedal_sensor': max(0, min(100, prev['pedal_sensor'] + np.random.uniform(-5, 5))),
            'speed': max(0, min(30, prev['speed'] + np.random.uniform(-1, 1))),
            'system_voltage': max(11.5, min(14.5, prev['system_voltage'] + np.random.uniform(-0.1, 0.1))),
            'fuel': max(10, min(100, prev['fuel'] - np.random.uniform(0, 0.02))),  # Fuel generally decreases slowly
            'temperature': max(60, min(110, prev['temperature'] + np.random.uniform(-1, 1))),
            'brake_control': max(0, min(100, prev['brake_control'] + np.random.uniform(-8, 8))),
            'transmission_pressure': max(1000, min(2500, prev['transmission_pressure'] + np.random.uniform(-25, 25))),
        }
    else:
        # Generate completely new values for first-time equipment
        sensor_data = {
            'air_filter_pressure': np.random.uniform(95, 110),  # kPa
            'exhaust_gas_temp': np.random.uniform(300, 400),  # Celsius
            'hydraulic_pump_rate': np.random.uniform(30, 45),  # L/min
            'oil_pressure': np.random.uniform(45, 65),  # PSI
            'pedal_sensor': np.random.uniform(20, 80),  # % of depression
            'speed': np.random.uniform(5, 25),  # km/h
            'system_voltage': np.random.uniform(12.5, 13.8),  # Volts
            'fuel': np.random.uniform(40, 90),  # % of tank
            'temperature': np.random.uniform(80, 100),  # Celsius
            'brake_control': np.random.uniform(10, 30),  # % of brake applied
            'transmission_pressure': np.random.uniform(1500, 2200),  # kPa
        }
    
    # Store the current values for next time
    previous_values[time_key] = sensor_data
    
    return sensor_data

# Function to generate all timestamps for a rental period with 30-second intervals
def generate_rental_timestamps(checkout_date, checkin_date):
    checkout_datetime = datetime.strptime(checkout_date, '%Y-%m-%d')
    checkin_datetime = datetime.strptime(checkin_date, '%Y-%m-%d')
    
    # Add time to make it a datetime (assuming checkout at 8 AM and checkin at 6 PM)
    checkout_datetime = checkout_datetime.replace(hour=8, minute=0, second=0)
    checkin_datetime = checkin_datetime.replace(hour=18, minute=0, second=0)
    
    # Generate timestamps every 30 seconds from checkout to checkin
    timestamps = []
    current_timestamp = checkout_datetime
    
    while current_timestamp <= checkin_datetime:
        timestamps.append(current_timestamp)
        current_timestamp += timedelta(minutes=5)
    
    return timestamps

# Main function to generate sensor data for all rental periods
def generate_equipment_sensor_data():
    # Read the equipment rentals CSV file
    try:
        rentals_df = pd.read_csv('equipment_rentals.csv')
    except Exception as e:
        print(f"Error reading equipment_rentals.csv: {e}")
        return
    
    # Create an empty list to store all sensor data rows
    all_sensor_data_rows = []
    
    # Counter for serial number
    sno = 1
    
    print(f"Processing {len(rentals_df)} equipment rentals...")
    
    # Initialize the CSV file with headers
    header_written = False
    
    # Loop through each equipment rental
    for index, rental in rentals_df.iterrows():
        equipment_id = rental['equipment_id']
        equipment_type = rental['equipment_type']
        checkout_date = rental['checkout_date']
        checkin_date = rental['checkin_date']
        
        print(f"Processing rental {index + 1}/{len(rentals_df)}: {equipment_type} {equipment_id} from {checkout_date} to {checkin_date}")
        
        # Generate all timestamps for this rental period
        timestamps = generate_rental_timestamps(checkout_date, checkin_date)
        
        print(f"  Generating sensor data for {len(timestamps)} timestamps (30-second intervals)")
        
        # Create a list to store sensor data for this rental only
        rental_sensor_data_rows = []
        
        # Generate sensor data for each timestamp
        for timestamp in timestamps:
            # Generate sensor data for this equipment at this timestamp
            sensor_data = generate_sensor_data(equipment_id, timestamp)
            
            # Create a row with the requested columns
            row = {
                'sno': sno,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'equipment_type': equipment_type,
                'equipment_id': equipment_id,
            }
            
            # Add all sensor data to the row
            row.update(sensor_data)
            
            # Add the row to this rental's list
            rental_sensor_data_rows.append(row)
            
            # Increment the serial number
            sno += 1
        
        # Convert this rental's data to DataFrame and append to CSV
        if rental_sensor_data_rows:
            rental_df = pd.DataFrame(rental_sensor_data_rows)
            
            # Write to CSV (append mode after first write)
            if not header_written:
                # First write - create file with headers
                rental_df.to_csv('sensor_data_5min.csv', mode='w', index=False, header=True)
                header_written = True
                print(f"    Created sensor_data_5min.csv with {len(rental_sensor_data_rows)} records for {equipment_id}")
            else:
                # Subsequent writes - append without headers
                rental_df.to_csv('sensor_data_5min.csv', mode='a', index=False, header=False)
                print(f"    Appended {len(rental_sensor_data_rows)} records for {equipment_id}")
            
            # Clear the list to free memory
            rental_sensor_data_rows.clear()
    
    print(f"\nCompleted! All sensor data has been saved to sensor_data.csv")
    print(f"Total equipment rentals processed: {len(rentals_df)}")

# Main function to run the sensor data generation
def main():
    print("Starting equipment sensor data generation for all rental periods...")
    generate_equipment_sensor_data()
    print("Sensor data generation complete!")

if __name__ == "__main__":
    main()