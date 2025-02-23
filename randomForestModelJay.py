import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

file_path = "filteredNHLFinal.csv"
df = pd.read_csv(file_path)

nhl_df = df[df["League"] == "NHL"].copy()
non_nhl_df = df[df["League"] != "NHL"].copy()

career_totals_fixed = non_nhl_df.groupby("Player")[["GP", "G", "A", "PTS", "PIM", "+/-"]].sum().reset_index()
career_totals_fixed = career_totals_fixed.rename(columns={
    "GP": "GP_non_nhl", "G": "G_non_nhl", "A": "A_non_nhl",
    "PTS": "PTS_non_nhl", "PIM": "PIM_non_nhl", "+/-": "+/-_non_nhl"
})

player_league_history = non_nhl_df.groupby("Player")["League"].unique().reset_index()

encoder = OneHotEncoder(sparse_output=False)
league_encoded = encoder.fit_transform(player_league_history["League"].apply(lambda x: ','.join(x)).values.reshape(-1, 1))

league_encoded_df = pd.DataFrame(league_encoded, columns=encoder.get_feature_names_out(["League"]))
league_encoded_df["Player"] = player_league_history["Player"]

career_totals_with_leagues = career_totals_fixed.merge(league_encoded_df, on="Player", how="left")

nhl_first_3_seasons = nhl_df.groupby("Player").head(3).copy()

nhl_first_3_seasons["Season_Number"] = nhl_first_3_seasons.groupby("Player").cumcount()

merged_df_seasonal = nhl_first_3_seasons.merge(career_totals_with_leagues, on="Player", how="left")

merged_df_seasonal = merged_df_seasonal.dropna()

features_seasonal = ["GP_non_nhl", "G_non_nhl", "A_non_nhl", "PTS_non_nhl", "PIM_non_nhl", "+/-_non_nhl", "Season_Number"] + list(league_encoded_df.columns[:-1])

targets_seasonal = ["GP", "G", "A", "PTS", "PIM", "+/-"]

X_seasonal = merged_df_seasonal[features_seasonal]
y_seasonal = merged_df_seasonal[targets_seasonal]
X_train_seasonal, X_test_seasonal, y_train_seasonal, y_test_seasonal = train_test_split(X_seasonal, y_seasonal, test_size=0.2, random_state=42)

rf_model_seasonal = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model_seasonal.fit(X_train_seasonal, y_train_seasonal)


y_pred_seasonal = rf_model_seasonal.predict(X_test_seasonal)

mae_seasonal = mean_absolute_error(y_test_seasonal, y_pred_seasonal)
rmse_seasonal = mean_squared_error(y_test_seasonal, y_pred_seasonal) ** 0.5

print(f"Mean Absolute Error (MAE): {mae_seasonal:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_seasonal:.2f}")


test_player_custom = pd.DataFrame({
    "GP_non_nhl": [30 + 127 + 83] * 3,
    "G_non_nhl": [6 + 9 + 21] * 3,
    "A_non_nhl": [6 + 23 + 15] * 3,
    "PTS_non_nhl": [12 + 42 + 36] * 3,
    "PIM_non_nhl": [15 + 18 + 53] * 3,
    "+/-_non_nhl": [-25 + (-4) + 6] * 3,
    "Season_Number": [0, 1, 2]
})

league_features_custom = {col: [0] * 3 for col in league_encoded_df.columns[:-1]}
league_features_custom["League_Rus-MHL"] = [1] * 3
league_features_custom["League_AHL"] = [1] * 3
league_features_custom["League_KHL"] = [1] * 3

league_df_custom = pd.DataFrame(league_features_custom)
test_player_custom = pd.concat([test_player_custom, league_df_custom], axis=1)

missing_cols_custom = set(features_seasonal) - set(test_player_custom.columns)
for col in missing_cols_custom:
    test_player_custom[col] = [0] * 3

test_player_custom = test_player_custom[features_seasonal]

predicted_nhl_performance_custom = rf_model_seasonal.predict(test_player_custom)

for season_num in range(3):
    print(f"Predicted NHL Performance for Season {season_num + 1}:")
    print(f"Games Played: {predicted_nhl_performance_custom[season_num][0]:.1f}")
    print(f"Goals: {predicted_nhl_performance_custom[season_num][1]:.1f}")
    print(f"Assists: {predicted_nhl_performance_custom[season_num][2]:.1f}")
    print(f"Points: {predicted_nhl_performance_custom[season_num][3]:.1f}")
    print(f"Penalty Minutes: {predicted_nhl_performance_custom[season_num][4]:.1f}")
    print(f"Plus/Minus: {predicted_nhl_performance_custom[season_num][5]:.1f}")
    print("-" * 40)
