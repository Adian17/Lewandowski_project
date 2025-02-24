import pandas as pd

# Load the dataset
df = pd.read_csv("filtered_nhl_history.csv")  # Use the cleaned dataset

# Drop the "Team" column as it's no longer needed
df = df.drop(columns=["Team"])

# Convert numeric columns to the correct data type to prevent string concatenation
numeric_cols = ["GP", "G", "A", "PTS", "PIM", "+/-"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Separate NHL and non-NHL data
nhl_df = df[df["League"] == "NHL"]  # Keep NHL seasons separate
non_nhl_df = df[df["League"] != "NHL"]  # Group non-NHL leagues

# Group non-NHL leagues by Player and League, summing stats correctly
non_nhl_grouped = non_nhl_df.groupby(["Player", "League"], as_index=False)[numeric_cols].sum()

# Concatenate the grouped non-NHL data with individual NHL seasons
final_df = pd.concat([nhl_df, non_nhl_grouped]).sort_values(by=["Player", "Season"], na_position="first")

# Save the final dataset
final_df.to_csv("final_nhl_dataset.csv", index=False)

print("Grouping complete! The dataset is saved as 'final_nhl_dataset.csv'.")
