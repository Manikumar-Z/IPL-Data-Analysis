import pandas as pd
import matplotlib.pyplot as plt
import os
import theme

def run(df, match_df):
    print("Running Module 12: Batting Position Dynamics...")
    out_dir = 'output/charts/12_batting_position'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Infer batting position based on appearance in the innings
    # For each match and innings, order batters by min ball faced
    batter_first_ball = df.groupby(['match_id', 'innings', 'batter'])['legal_ball_count'].min().reset_index()
    
    # Sort and assign position
    batter_first_ball = batter_first_ball.sort_values(['match_id', 'innings', 'legal_ball_count'])
    batter_first_ball['batting_position'] = batter_first_ball.groupby(['match_id', 'innings']).cumcount() + 1
    
    # Merge back to get runs for that specific innings
    innings_runs = df.groupby(['match_id', 'innings', 'batter']).agg(
        runs=('runs_batter', 'sum'),
        balls=('is_legal_ball', 'sum')
    ).reset_index()
    
    batting_positions = batter_first_ball.merge(innings_runs, on=['match_id', 'innings', 'batter'])
    
    # Aggregate by position
    pos_stats = batting_positions.groupby('batting_position').agg(
        total_runs=('runs', 'sum'),
        total_balls=('balls', 'sum'),
        innings_count=('match_id', 'count')
    ).reset_index()
    
    pos_stats = pos_stats[pos_stats['batting_position'] <= 11]
    pos_stats['avg_score'] = pos_stats['total_runs'] / pos_stats['innings_count']
    pos_stats['strike_rate'] = (pos_stats['total_runs'] / pos_stats['total_balls']) * 100
    
    # 1. Runs & SR by position
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    bars = ax1.bar(pos_stats['batting_position'], pos_stats['avg_score'], color='#00BFFF', alpha=0.7, label='Average Score per Innings')
    line = ax2.plot(pos_stats['batting_position'], pos_stats['strike_rate'], color='#FFD700', marker='o', linewidth=2, label='Strike Rate')
    
    ax1.set_title('Does Batting Order Win Matches?\nAverage Score & Strike Rate by Position', pad=20)
    ax1.set_xlabel('Batting Position')
    ax1.set_ylabel('Average Score', color='#00BFFF')
    ax2.set_ylabel('Strike Rate', color='#FFD700')
    ax1.set_xticks(range(1, 12))
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', color='white', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_position_stats.png', dpi=300)
    plt.close()
    
    # 2. Run Contribution Pie
    top_order = pos_stats[pos_stats['batting_position'].isin([1, 2, 3])]['total_runs'].sum()
    middle_order = pos_stats[pos_stats['batting_position'].isin([4, 5, 6])]['total_runs'].sum()
    lower_order = pos_stats[pos_stats['batting_position'].isin([7, 8, 9, 10, 11])]['total_runs'].sum()
    
    labels = ['Top Order (1-3)', 'Middle Order (4-6)', 'Lower Order (7-11)']
    sizes = [top_order, middle_order, lower_order]
    colors = ['#FFD700', '#00BFFF', '#EC1C24']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, 
                                      textprops=dict(color="w", weight="bold"))
    
    for text in texts:
        text.set_color('white')
        
    centre_circle = plt.Circle((0,0), 0.70, fc='#0D1117')
    fig.gca().add_artist(centre_circle)
    
    ax.set_title('Total Run Contribution by Batting Order Phase', pad=20)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_position_contribution.png', dpi=300)
    plt.close()
