from flask import Flask, render_template, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Ensure 'static/images' folder exists for plots
if not os.path.exists('static/images'):
    os.makedirs('static/images')


# 🔥 Function to plot and save graphs
def plot_metric(df, metric, title, filename):
    plt.figure(figsize=(12, 8))
    plt.barh(df['League'], df[metric], color='green', height=0.8)
    plt.xlabel(f'{title} per Game', fontsize=14)
    plt.ylabel('Pre-NHL League', fontsize=14)
    plt.title(f'Average NHL {title} per Game by Pre-NHL League', fontsize=16)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'static/images/{filename}', bbox_inches='tight')
    plt.close()


@app.route('/league_analysis')
def league_analysis():
    # 🔥 Load and process data
    df = pd.read_csv('cleaned_byLeague.csv')
    df['League'] = df['League'].replace({'Russia': 'KHL'})
    df = df[df['League'] != 'Exhibition']
    df = df[['Player', 'League', 'GP', 'G', 'A', 'PTS']]

    nhl_stats = df[df['League'] == 'NHL'][['Player', 'GP', 'G', 'A', 'PTS']]
    pre_nhl_league = (
        df[df['League'] != 'NHL']
        .sort_values(['Player', 'GP'], ascending=[True, False])
        .drop_duplicates('Player')[['Player', 'League']]
    )

    merged_df = pd.merge(pre_nhl_league, nhl_stats, on='Player')
    performance_by_league = merged_df.groupby('League').agg({
        'G': 'sum',
        'A': 'sum',
        'PTS': 'sum',
        'GP': 'sum'
    }).reset_index()

    performance_by_league['G_per_GP'] = performance_by_league['G'] / performance_by_league['GP']
    performance_by_league['A_per_GP'] = performance_by_league['A'] / performance_by_league['GP']
    performance_by_league['PTS_per_GP'] = performance_by_league['PTS'] / performance_by_league['GP']
    performance_by_league.sort_values(by='PTS_per_GP', ascending=False, inplace=True)

    # 🔥 Generate graphs & save them
    plot_metric(performance_by_league, 'G_per_GP', 'Goals', 'goals_per_game.png')
    plot_metric(performance_by_league, 'A_per_GP', 'Assists', 'assists_per_game.png')
    plot_metric(performance_by_league, 'PTS_per_GP', 'Points', 'points_per_game.png')

    return render_template('league_analysis.html')


@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)


if __name__ == '__main__':
    app.run(debug=True)