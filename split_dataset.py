import pandas as pd
import math

# Load dataset
df = pd.read_csv("train.csv")

total_rows = len(df)
num_parts = 10

rows_per_part = math.ceil(total_rows / num_parts)

print(f"Total rows: {total_rows}")
print(f"Rows per part: {rows_per_part}")

for i in range(num_parts):
    start = i * rows_per_part
    end = start + rows_per_part

    part_df = df.iloc[start:end]

    filename = f"train_part_{i+1}.csv"
    part_df.to_csv(filename, index=False)

    print(f"{filename} → rows {start+1} to {min(end, total_rows)}")
