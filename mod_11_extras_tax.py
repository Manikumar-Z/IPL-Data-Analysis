import pandas as pd
import matplotlib.pyplot as plt
import os
import theme
import numpy as np

def run(df, match_df):
    print("Running Module 11: Extras Tax...")
    out_dir = 'output/charts/11_extras_tax'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Calculate extras conceded by bowling team
    extras_df = df.groupby(['match_id', 'bowling_team']).agg(
        wides=('extras_wides', 'sum'),
        noballs=('extras_noballs', 'sum'),
        total_extras=('runs_extras', 'sum')
    ).reset_index()
    
    # Merge with match result
    extras_df = extras_df.merge(match_df[['match_id', 'winner', 'season_year']], on='match_id')
    extras_df['is_loser'] = (extras_df['bowling_team'] != extras_df['winner']).astype(int)
    
    # 1. Total extras per season for top/bottom 3 disciplined teams
    team_extras_season = extras_df.groupby(['season_year', 'bowling_team'])['total_extras'].mean().reset_index()
    overall_team_extras = team_extras_season.groupby('bowling_team')['total_extras'].mean().sort_values()
    
    active_teams = ['Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 
                    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 
                    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru', 
                    'Sunrisers Hyderabad']
    
    overall_team_extras = overall_team_extras[overall_team_extras.index.isin(active_teams)]
    
    top_3 = overall_team_extras.head(3).index
    bottom_3 = overall_team_extras.tail(3).index
    
    plot_teams = list(top_3) + list(bottom_3)
    plot_data = team_extras_season[team_extras_season['bowling_team'].isin(plot_teams)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for team in top_3:
        team_data = plot_data[plot_data['bowling_team'] == team]
        ax.plot(team_data['season_year'], team_data['total_extras'], marker='o', linestyle='--', label=f'{team} (Most Disciplined)')
        
    for team in bottom_3:
        team_data = plot_data[plot_data['bowling_team'] == team]
        ax.plot(team_data['season_year'], team_data['total_extras'], marker='X', linestyle='-', linewidth=2, label=f'{team} (Least Disciplined)')
        
    ax.set_title('The Discipline Gap: Average Extras Conceded per Match', pad=20)
    ax.set_xlabel('Season')
    ax.set_ylabel('Average Extras Conceded')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_extras_trend.png', dpi=300)
    plt.close()
    
    # 2. Correlation between Extras and Losses
    # Is conceding more extras correlated with losing?
    team_match_counts = extras_df.groupby('bowling_team')['match_id'].count()
    valid_teams = team_match_counts[team_match_counts >= 30].index
    
    team_stats = extras_df[extras_df['bowling_team'].isin(valid_teams)].groupby('bowling_team').agg(
        avg_extras=('total_extras', 'mean'),
        loss_pct=('is_loser', lambda x: x.mean() * 100)
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(team_stats['avg_extras'], team_stats['loss_pct'], s=150, color='#00BFFF', alpha=0.8)
    
    # Add regression line
    m, b = np.polyfit(team_stats['avg_extras'], team_stats['loss_pct'], 1)
    ax.plot(team_stats['avg_extras'], m*team_stats['avg_extras'] + b, color='#EA1A85', linestyle='--')
    
    for i, row in team_stats.iterrows():
        ax.text(row['avg_extras'], row['loss_pct'] + 0.5, row['bowling_team'], fontsize=9, ha='center')
        
    ax.set_title('The Hidden Cost of Indiscipline: Extras vs Loss Percentage', pad=20)
    ax.set_xlabel('Average Extras Conceded per Match')
    ax.set_ylabel('Match Loss Percentage (%)')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_extras_vs_losses.png', dpi=300)
    plt.close()
