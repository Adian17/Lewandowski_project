import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# Load the cleaned dataset
file_path = "filteredNHLFinal.csv"  # Ensure this is the correct file path
df = pd.read_csv(file_path)


# Separate NHL and non-NHL data
nhl_df = df[df["League"] == "NHL"].copy()
non_nhl_df = df[df["League"] != "NHL"].copy()

# Get last season stats in non-NHL leagues before entering NHL
last_season_non_nhl = non_nhl_df.sort_values(by=["Player", "Season"], ascending=[True, False]).drop_duplicates("Player")

# Merge last season stats with first NHL season stats
first_nhl_season = nhl_df.groupby("Player").first().reset_index()
player_transitions = last_season_non_nhl.merge(first_nhl_season, on="Player", suffixes=("_non_nhl", "_nhl"))


# Compute translation factors (how much non-NHL stats carry over to NHL)
for stat in ["G", "A", "PTS", "PIM"]:
    player_transitions[f"{stat}_translation"] = player_transitions[f"{stat}_nhl"] / player_transitions[f"{stat}_non_nhl"].replace(0, np.nan)

# Train a model to predict league translation rates
league_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# Ensure 'League_non_nhl' exists and has valid values
if "League_non_nhl" in player_transitions.columns:
    print(player_transitions["League_non_nhl"].unique())  # Check unique values
    league_encoded = league_encoder.fit_transform(player_transitions[["League_non_nhl"]])
else:
    raise KeyError("Column 'League_non_nhl' is missing from player_transitions!")
league_encoded_df = pd.DataFrame(league_encoded, columns=league_encoder.get_feature_names_out())

X_league = league_encoded_df
Y_league = player_transitions[["PTS_translation"]].fillna(0)  # Predicting how PTS carry over

rf_league_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_league_model.fit(X_league, Y_league.values.ravel())

# Extract learned league difficulty scores
league_difficulty = pd.DataFrame({
    "League": league_encoder.categories_[0],
    "Difficulty_Score": rf_league_model.feature_importances_
}).sort_values(by="Difficulty_Score", ascending=False)

# Merge difficulty scores into player career totals
career_totals_fixed = non_nhl_df.groupby("Player")[["GP", "G", "A", "PTS", "PIM"]].sum().reset_index()
player_league_history = non_nhl_df.groupby("Player")["League"].unique().reset_index()
player_league_history["League"] = player_league_history["League"].apply(lambda x: ','.join(x))

# One-hot encode leagues for main model
# One-hot encode leagues for main model
player_league_history = player_league_history.rename(columns={"League": "League_non_nhl"})  # <-- Add this line
league_encoded_main = league_encoder.transform(player_league_history[["League_non_nhl"]])
league_encoded_main_df = pd.DataFrame(league_encoded_main, columns=league_encoder.get_feature_names_out())
league_encoded_main_df["Player"] = player_league_history["Player"]

# Apply learned difficulty weights to career stats
career_totals_weighted = career_totals_fixed.copy()
for league in league_difficulty["League"]:
    weight = league_difficulty.loc[league_difficulty["League"] == league, "Difficulty_Score"].values[0]
    career_totals_weighted.loc[:, ["G", "A", "PTS", "PIM"]] *= weight

# Merge adjusted stats with league encoding
career_totals_with_leagues = career_totals_weighted.merge(league_encoded_main_df, on="Player", how="left")

# Get first 3 NHL seasons
nhl_first_3_seasons = nhl_df.groupby("Player").head(3).copy()
nhl_first_3_seasons["Season_Number"] = nhl_first_3_seasons.groupby("Player").cumcount()
# Rename columns to remove `_x` and `_y` suffixes after merging
merged_df_seasonal = nhl_first_3_seasons.merge(career_totals_with_leagues, on="Player", how="left")
merged_df_seasonal = merged_df_seasonal.rename(columns={
    "GP_x": "GP", "G_x": "G", "A_x": "A", "PTS_x": "PTS", "PIM_x": "PIM"
})
merged_df_seasonal = merged_df_seasonal.dropna()

# Define features
features_seasonal = ["GP", "G", "A", "PTS", "PIM", "Season_Number"] + list(league_encoded_main_df.columns[:-1])
targets_seasonal = ["GP", "G", "A", "PTS", "PIM"]

# Debugging: Check the available columns in merged_df_seasonal
print("Columns in merged_df_seasonal:", merged_df_seasonal.columns)
print("First few rows of merged_df_seasonal:")
print(merged_df_seasonal.head())

# Train NHL performance prediction model
X_train, X_test, y_train, y_test = train_test_split(merged_df_seasonal[features_seasonal], merged_df_seasonal[targets_seasonal], test_size=0.2, random_state=42)
rf_model_seasonal = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model_seasonal.fit(X_train, y_train)

# Evaluate
y_pred = rf_model_seasonal.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
