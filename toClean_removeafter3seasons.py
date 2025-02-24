import pandas as pd

df = pd.read_csv("filtered_nhl_history.csv")

df = df.drop(columns=["Team"])

numeric_cols = ["GP", "G", "A", "PTS", "PIM", "+/-"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

nhl_df = df[df["League"] == "NHL"]  
non_nhl_df = df[df["League"] != "NHL"]  

non_nhl_grouped = non_nhl_df.groupby(["Player", "League"], as_index=False)[numeric_cols].sum()

final_df = pd.concat([nhl_df, non_nhl_grouped]).sort_values(by=["Player", "Season"], na_position="first")

final_df.to_csv("final_nhl_dataset.csv", index=False)

print("The dataset is saved as 'final_nhl_dataset.csv'.")
