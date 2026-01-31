#!/usr/bin/env python3
import pandas as pd

# Load Libraries
# Initial Load
try:
    df = pd.read_csv('sensitive_payroll.csv', nrows=5)
    print(df.head())
except Exception as e:
    print(f"Error loading file: {e}")

# Data Types
try:
    print(df.dtypes)
except Exception as e:
    print(f"Error checking data types: {e}")

# Handle Missing Values
try:
    df['Bonus'] = df['Bonus'].fillna(0)
except Exception as e:
    print(f"Error handling missing values: {e}")

# Format Conversion
try:
    df['Bonus'] = df['Bonus'].str.replace('[\$,]', '', regex=True).astype(float)
except Exception as e:
    print(f"Error formatting bonus column: {e}")

# Outlier Validation
try:
    if (df['Bonus'] < 0).any():
        print("Negative values found in the Bonus column.")
except Exception as e:
    print(f"Error validating outliers: {e}")

# Summation
try:
    total_bonus = df['Bonus'].sum()
    print(f'Total Bonus: ${total_bonus:.2f}')
except Exception as e:
    print(f"Error calculating total bonus: {e}")

# Summary Stats
try:
    count_bonuses = (df['Bonus'] > 0).sum()
    avg_bonus = df['Bonus'].mean()
    print(f'Count of Employees with Bonuses: {count_bonuses}')
    print(f'Average Bonus Amount: ${avg_bonus:.2f}')
except Exception as e:
    print(f"Error calculating summary stats: {e}")