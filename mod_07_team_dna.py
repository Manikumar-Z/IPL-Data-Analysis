import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import theme

def run(df, match_df):
    print("Running Module 7: Team DNA...")
    out_dir = 'output/charts/07_team_dna'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Active teams filter (roughly those playing recently or historically huge)
    active_teams = ['Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 
                    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 
                    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru', 
                    'Sunrisers Hyderabad']
    
    # 1. Season by Season Win %
    season_team_wins = match_df.groupby(['season_year', 'winner']).size().reset_index(name='wins')
    season_team_matches = pd.concat([
        match_df.groupby(['season_year', 'team1']).size().reset_index(name='matches').rename(columns={'team1': 'team'}),
        match_df.groupby(['season_year', 'team2']).size().reset_index(name='matches').rename(columns={'team2': 'team'})
    ]).groupby(['season_year', 'team'])['matches'].sum().reset_index()
    
    season_stats = season_team_matches.merge(season_team_wins, left_on=['season_year', 'team'], right_on=['season_year', 'winner'], how='left')
    season_stats['wins'] = season_stats['wins'].fillna(0)
    season_stats['win_pct'] = (season_stats['wins'] / season_stats['matches']) * 100
    
    # Pivot for heatmap
    win_pct_pivot = season_stats[season_stats['team'].isin(active_teams)].pivot(index='team', columns='season_year', values='win_pct')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(win_pct_pivot, cmap='RdYlGn', annot=False, linewidths=.5, cbar_kws={'label': 'Win %'})
    ax.set_title('Franchise Era Dominance: Season-by-Season Win %', pad=20)
    ax.set_xlabel('Season')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_season_win_heatmap.png', dpi=300)
    plt.close()
    
    # 2. Head to Head Matrix
    h2h_matches = match_df[match_df['team1'].isin(active_teams) & match_df['team2'].isin(active_teams)].copy()
    
    # Ensure team1 is alphabetically before team2 for consistent grouping
    h2h_matches['t1'] = np.where(h2h_matches['team1'] < h2h_matches['team2'], h2h_matches['team1'], h2h_matches['team2'])
    h2h_matches['t2'] = np.where(h2h_matches['team1'] < h2h_matches['team2'], h2h_matches['team2'], h2h_matches['team1'])
    
    h2h_counts = h2h_matches.groupby(['t1', 't2']).size().reset_index(name='matches')
    t1_wins = h2h_matches[h2h_matches['t1'] == h2h_matches['winner']].groupby(['t1', 't2']).size().reset_index(name='t1_wins')
    
    h2h_stats = h2h_counts.merge(t1_wins, on=['t1', 't2'], how='left')
    h2h_stats['t1_wins'] = h2h_stats['t1_wins'].fillna(0)
    h2h_stats['t1_win_pct'] = (h2h_stats['t1_wins'] / h2h_stats['matches']) * 100
    
    # Build full matrix
    matrix = pd.DataFrame(index=active_teams, columns=active_teams, dtype=float)
    for _, row in h2h_stats.iterrows():
        matrix.loc[row['t1'], row['t2']] = row['t1_win_pct']
        matrix.loc[row['t2'], row['t1']] = 100 - row['t1_win_pct']
        
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap='coolwarm', center=50, annot=True, fmt='.0f', cbar_kws={'label': 'Row Team Win %'})
    ax.set_title('Head-to-Head Dominance Matrix', pad=20)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_head_to_head_matrix.png', dpi=300)
    plt.close()
