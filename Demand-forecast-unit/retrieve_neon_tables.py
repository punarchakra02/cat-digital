#!/usr/bin/env python3
"""
Script to retrieve tables from a Neon PostgreSQL database.
"""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Connection string for Neon PostgreSQL
conn_string = "postgresql://neondb_owner:npg_YwgKfyUPJ76j@ep-fragrant-river-adr03asc-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

def get_table_names(conn):
    """Get all table names in the public schema"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        return [table[0] for table in cur.fetchall()]

def main():
    try:
        # Connect to the database
        print("Connecting to Neon PostgreSQL database...")
        conn = psycopg2.connect(conn_string)
        
        # Get table names
        table_names = get_table_names(conn)
        print(f"Found {len(table_names)} tables: {', '.join(table_names)}")
        
        # Create SQLAlchemy engine for pandas
        engine = create_engine(conn_string)
        
        # Retrieve data from each table into separate variables
        # Assuming there are 4 tables as mentioned
        if len(table_names) >= 4:
            table1_name = table_names[0]
            table2_name = table_names[1]
            table3_name = table_names[2]
            table4_name = table_names[3]
            
            # Retrieve data from each table
            table1_data = pd.read_sql(f"SELECT * FROM {table1_name}", engine)
            table2_data = pd.read_sql(f"SELECT * FROM {table2_name}", engine)
            table3_data = pd.read_sql(f"SELECT * FROM {table3_name}", engine)
            table4_data = pd.read_sql(f"SELECT * FROM {table4_name}", engine)
            
            # Display basic info about each table
            print(f"\nTable 1: {table1_name} - {len(table1_data)} rows, {len(table1_data.columns)} columns")
            print(f"Columns: {', '.join(table1_data.columns)}")
            
            print(f"\nTable 2: {table2_name} - {len(table2_data)} rows, {len(table2_data.columns)} columns")
            print(f"Columns: {', '.join(table2_data.columns)}")
            
            print(f"\nTable 3: {table3_name} - {len(table3_data)} rows, {len(table3_data.columns)} columns")
            print(f"Columns: {', '.join(table3_data.columns)}")
            
            print(f"\nTable 4: {table4_name} - {len(table4_data)} rows, {len(table4_data.columns)} columns")
            print(f"Columns: {', '.join(table4_data.columns)}")
            
            # At this point, table1_data, table2_data, table3_data, and table4_data
            # contain the data from the 4 tables
            
            # Example: You can now work with these variables
            # print(table1_data.head())
            
            # Return the tables for further use
            return table1_data, table2_data, table3_data, table4_data
        else:
            print(f"Expected 4 tables but found {len(table_names)}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the connection
        if 'conn' in locals():
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    # If run directly, execute the main function
    tables = main()
    if tables:
        table1, table2, table3, table4 = tables
        # You can now work with these variables
        # For example:
        print("\nPreview of first table:")
        print(table1.head())
