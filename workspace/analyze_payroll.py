#!/usr/bin/env python3
import pandas as pd
from io import StringIO

# Step 1: Data Loading and Environment Setup
try:
    # Load the necessary data manipulation library (typically `pandas` for Python)
    df = pd.read_csv('sensitive_payroll.csv', encoding='utf-8')
except Exception as e:
    print(f'Error loading file: {e}')
    exit(1)

# Step 2: Schema Inspection
try:
    # Load Header: Read the first few rows to identify the exact column names
    header = df.head()
    print(header)
except Exception as e:
    print(f'Error inspecting schema: {e}')
    exit(1)

# Step 3: Data Cleaning and Preprocessing
try:
    # Identify Target Column: Locate the column representing 'Bonuses'
    bonus_columns = ['Bonus', 'Annual_Bonus', 'Bonus_Amount', 'Incentive_Pay']
    for col in bonus_columns:
        if col in df.columns:
            bonus_col = col
            break
    else:
        print('No bonus column found')
        exit(1)

    # Handle Non-Numeric Characters: Remove currency symbols, commas, or whitespace
    df[bonus_col] = df[bonus_col].str.replace('[^
\d.]', '', regex=True)

    # Type Conversion: Convert the cleaned column to a numeric data type (`float64`)
    df[bonus_col] = pd.to_numeric(df[bonus_col], errors='coerce')

    # Address Missing Values: Fill NaN or null entries with 0
    df[bonus_col].fillna(0, inplace=True)
except Exception as e:
    print(f'Error cleaning and preprocessing data: {e}')
    exit(1)

# Step 4: Calculation and Aggregation
try:
    # Summation: Execute the sum function on the processed bonus column
    total_bonus = df[bonus_col].sum()
    print(f'Total Bonus: ${total_bonus:.2f}')
except Exception as e:
    print(f'Error calculating total bonus: {e}')
    exit(1)
