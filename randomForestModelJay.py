# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load the cleaned dataset
file_path = "filteredNHLFinal.csv"  # Ensure this is the correct file path
df = pd.read_csv(file_path)

# Separate NHL and non-NHL data
nhl_df = df[df["League"] == "NHL"].copy()
non_nhl_df = df[df["League"] != "NHL"].copy()

# Aggregate career totals from non-NHL leagues
career_totals_fixed = non_nhl_df.groupby("Player")[["GP", "G", "A", "PTS", "PIM", "+/-"]].sum().reset_index()
career_totals_fixed = career_totals_fixed.rename(columns={
    "GP": "GP_non_nhl", "G": "G_non_nhl", "A": "A_non_nhl",
    "PTS": "PTS_non_nhl", "PIM": "PIM_non_nhl", "+/-": "+/-_non_nhl"
})

# Extract the leagues each player played in before the NHL
player_league_history = non_nhl_df.groupby("Player")["League"].unique().reset_index()

# Convert leagues into categorical one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
league_encoded = encoder.fit_transform(player_league_history["League"].apply(lambda x: ','.join(x)).values.reshape(-1, 1))

# Convert encoding to DataFrame and merge with player names
league_encoded_df = pd.DataFrame(league_encoded, columns=encoder.get_feature_names_out(["League"]))
league_encoded_df["Player"] = player_league_history["Player"]

# Merge league encoding with career totals
career_totals_with_leagues = career_totals_fixed.merge(league_encoded_df, on="Player", how="left")

# Get the first three NHL seasons per player
nhl_first_3_seasons = nhl_df.groupby("Player").head(3).copy()

# Add a "Season_Number" column (0 for first NHL season, 1 for second, 2 for third)
nhl_first_3_seasons["Season_Number"] = nhl_first_3_seasons.groupby("Player").cumcount()

# Merge NHL stats with career totals and league encoding
merged_df_seasonal = nhl_first_3_seasons.merge(career_totals_with_leagues, on="Player", how="left")

# Drop any remaining NaN values
merged_df_seasonal = merged_df_seasonal.dropna()

# Define features including league encoding and season number
features_seasonal = ["GP_non_nhl", "G_non_nhl", "A_non_nhl", "PTS_non_nhl", "PIM_non_nhl", "+/-_non_nhl", "Season_Number"] + list(league_encoded_df.columns[:-1])

# Define target variables: Predicting NHL performance for each of the first 3 seasons
targets_seasonal = ["GP", "G", "A", "PTS", "PIM", "+/-"]

# Split data into training and testing sets (80% train, 20% test)
X_seasonal = merged_df_seasonal[features_seasonal]
y_seasonal = merged_df_seasonal[targets_seasonal]
X_train_seasonal, X_test_seasonal, y_train_seasonal, y_test_seasonal = train_test_split(X_seasonal, y_seasonal, test_size=0.2, random_state=42)

# Train the updated Random Forest model
rf_model_seasonal = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model_seasonal.fit(X_train_seasonal, y_train_seasonal)

# Make predictions
y_pred_seasonal = rf_model_seasonal.predict(X_test_seasonal)

# Evaluate the model
mae_seasonal = mean_absolute_error(y_test_seasonal, y_pred_seasonal)
rmse_seasonal = mean_squared_error(y_test_seasonal, y_pred_seasonal) ** 0.5  # Manually compute RMSE

# Print evaluation results
print(f"Mean Absolute Error (MAE): {mae_seasonal:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_seasonal:.2f}")


# Create a test case: A player who played in Rus-MHL, AHL, and KHL before NHL
test_player_custom = pd.DataFrame({
    "GP_non_nhl": [30 + 127 + 83] * 3,   # Total games played before NHL (same for each season)
    "G_non_nhl": [6 + 9 + 21] * 3,       # Total goals before NHL
    "A_non_nhl": [6 + 23 + 15] * 3,      # Total assists before NHL
    "PTS_non_nhl": [12 + 42 + 36] * 3,   # Total points before NHL
    "PIM_non_nhl": [15 + 18 + 53] * 3,   # Total penalty minutes before NHL
    "+/-_non_nhl": [-25 + (-4) + 6] * 3, # Total plus/minus before NHL
    "Season_Number": [0, 1, 2]           # First, second, and third NHL seasons
})

# Add league one-hot encoding for Rus-MHL, AHL, and KHL
league_features_custom = {col: [0] * 3 for col in league_encoded_df.columns[:-1]}  # Default to 0
league_features_custom["League_Rus-MHL"] = [1] * 3  # Set Rus-MHL to 1
league_features_custom["League_AHL"] = [1] * 3      # Set AHL to 1
league_features_custom["League_KHL"] = [1] * 3      # Set KHL to 1

# Convert to DataFrame and merge with test player stats
league_df_custom = pd.DataFrame(league_features_custom)
test_player_custom = pd.concat([test_player_custom, league_df_custom], axis=1)

# Ensure all feature columns are present
missing_cols_custom = set(features_seasonal) - set(test_player_custom.columns)
for col in missing_cols_custom:
    test_player_custom[col] = [0] * 3  # Add missing features with default 0

# Reorder columns to match training data
test_player_custom = test_player_custom[features_seasonal]

# Predict performance for the first three NHL seasons
predicted_nhl_performance_custom = rf_model_seasonal.predict(test_player_custom)

# Display results for each season
for season_num in range(3):
    print(f"Predicted NHL Performance for Season {season_num + 1}:")
    print(f"Games Played: {predicted_nhl_performance_custom[season_num][0]:.1f}")
    print(f"Goals: {predicted_nhl_performance_custom[season_num][1]:.1f}")
    print(f"Assists: {predicted_nhl_performance_custom[season_num][2]:.1f}")
    print(f"Points: {predicted_nhl_performance_custom[season_num][3]:.1f}")
    print(f"Penalty Minutes: {predicted_nhl_performance_custom[season_num][4]:.1f}")
    print(f"Plus/Minus: {predicted_nhl_performance_custom[season_num][5]:.1f}")
    print("-" * 40)
