import pandas as pd
import matplotlib.pyplot as plt
import os
import theme

def run(df, match_df):
    print("Running Module 3: Top Performers...")
    out_dir = 'output/charts/03_top_performers'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Batter Stats
    batter_stats = df.groupby('batter').agg(
        runs=('runs_batter', 'sum'),
        balls=('is_legal_ball', 'sum'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        dots=('is_dot', 'sum'),
        dismissals=('is_wicket', lambda x: x[df.loc[x.index, 'wicket_player_out'] == df.loc[x.index, 'batter']].sum()),
        innings=('match_id', 'nunique')
    ).reset_index()
    
    batter_stats = batter_stats[batter_stats['runs'] >= 1000]
    batter_stats['strike_rate'] = (batter_stats['runs'] / batter_stats['balls']) * 100
    batter_stats['average'] = batter_stats['runs'] / batter_stats['dismissals'].replace(0, 1)
    
    # Impact Rating: (Runs × SR/100) + (Boundaries × 2) − (Dots × 0.5) per innings
    batter_stats['impact_score'] = (
        (batter_stats['runs'] * (batter_stats['strike_rate']/100)) + 
        ((batter_stats['fours'] + batter_stats['sixes']) * 2) - 
        (batter_stats['dots'] * 0.5)
    ) / batter_stats['innings']
    
    top_batters = batter_stats.sort_values('impact_score', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_batters['batter'][::-1], top_batters['impact_score'][::-1], color='#F26522')
    ax.set_title('Top 15 Batters by Custom Impact Rating\n(Rewards SR, Boundaries; Penalizes Dots)', pad=20)
    ax.set_xlabel('Impact Rating per Innings')
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_top_batters_impact.png', dpi=300)
    plt.close()
    
    # Bowler Stats
    bowler_stats = df.groupby('bowler').agg(
        balls=('is_legal_ball', 'sum'),
        runs_conceded=('runs_total', lambda x: x[df.loc[x.index, 'is_legal_ball'] == 1].sum() + df.loc[x.index, 'extras_wides'].sum() + df.loc[x.index, 'extras_noballs'].sum()),
        wickets=('is_wicket', 'sum'),
        dots=('is_dot', 'sum'),
        matches=('match_id', 'nunique')
    ).reset_index()
    
    bowler_stats = bowler_stats[bowler_stats['balls'] >= 600] # Min 100 overs
    bowler_stats['economy'] = (bowler_stats['runs_conceded'] / bowler_stats['balls']) * 6
    bowler_stats['dot_pct'] = (bowler_stats['dots'] / bowler_stats['balls']) * 100
    
    # Pressure Index: (Dot% × 100) + (Wickets × 15) − (Economy × 10) per match
    bowler_stats['pressure_index'] = (
        (bowler_stats['dot_pct'] * bowler_stats['matches']) + 
        (bowler_stats['wickets'] * 15) - 
        (bowler_stats['economy'] * 10 * bowler_stats['matches'])
    ) / bowler_stats['matches']
    
    top_bowlers = bowler_stats.sort_values('pressure_index', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_bowlers['bowler'][::-1], top_bowlers['pressure_index'][::-1], color='#3A225D')
    ax.set_title('Top 15 Bowlers by Pressure Index\n(Rewards Dots & Wickets; Penalizes High Economy)', pad=20)
    ax.set_xlabel('Pressure Index per Match')
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_top_bowlers_pressure.png', dpi=300)
    plt.close()
