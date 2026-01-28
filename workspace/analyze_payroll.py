#!/usr/bin/env python3
import pandas as pd

# Load data
df = pd.read_csv('sensitive_payroll.csv')

# Inspect the first few rows and get information about the DataFrame
df.head()
df.info()

# Identify the bonus column (example: 'Bonus_Amount')
bonus_column = 'Bonus_Amount'

# Clean data
# Strip non-numeric characters and convert to float
if df[bonus_column].dtype == 'object':
    df[bonus_column] = df[bonus_column].str.replace('[^0-9.]', '', regex=True).astype(float)

# Fill missing values with 0
df[bonus_column].fillna(0, inplace=True)

# Calculate the sum of bonuses
total_bonus = df[bonus_column].sum()

# Output the result
class Result:
    def __init__(self, total_bonus):
        self.total_bonus = total_bonus

result = Result(total_bonus)
print(result.total_bonus)

# Clear dataframe from memory (optional)
del df