import pandas as pd
import matplotlib.pyplot as plt
import os
import theme

def run(df, match_df):
    print("Running Module 5: The Clutch Factor...")
    out_dir = 'output/charts/05_clutch_factor'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # 1. Death Overs Specialists (Batting)
    death_df = df[df['phase'] == 'Death']
    death_batsmen = death_df.groupby('batter').agg(
        runs=('runs_batter', 'sum'),
        balls=('is_legal_ball', 'sum'),
        dismissals=('is_wicket', lambda x: x[death_df.loc[x.index, 'wicket_player_out'] == death_df.loc[x.index, 'batter']].sum())
    ).reset_index()
    
    death_batsmen = death_batsmen[death_batsmen['balls'] >= 150]
    death_batsmen['strike_rate'] = (death_batsmen['runs'] / death_batsmen['balls']) * 100
    death_batsmen['average'] = death_batsmen['runs'] / death_batsmen['dismissals'].replace(0, 1)
    
    top_death_batsmen = death_batsmen.sort_values('strike_rate', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(top_death_batsmen['average'], top_death_batsmen['strike_rate'], s=top_death_batsmen['runs'], c='#FFD700', alpha=0.7)
    for i, row in top_death_batsmen.iterrows():
        ax.text(row['average'], row['strike_rate'], row['batter'], fontsize=9, ha='center', va='center')
    ax.set_title('Death Overs Specialists (Overs 16-20)\nBubble size = Total Death Runs', pad=20)
    ax.set_xlabel('Death Overs Average')
    ax.set_ylabel('Death Overs Strike Rate')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_death_batsmen.png', dpi=300)
    plt.close()
    
    # 2. Death Overs Bowlers
    death_bowlers = death_df.groupby('bowler').agg(
        balls=('is_legal_ball', 'sum'),
        runs_conceded=('runs_total', lambda x: x[death_df.loc[x.index, 'is_legal_ball'] == 1].sum() + death_df.loc[x.index, 'extras_wides'].sum() + death_df.loc[x.index, 'extras_noballs'].sum()),
        wickets=('is_wicket', 'sum')
    ).reset_index()
    
    death_bowlers = death_bowlers[death_bowlers['balls'] >= 200]
    death_bowlers['economy'] = (death_bowlers['runs_conceded'] / death_bowlers['balls']) * 6
    
    top_death_bowlers = death_bowlers.sort_values('economy', ascending=True).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_death_bowlers['bowler'][::-1], top_death_bowlers['economy'][::-1], color='#00BFFF')
    ax.set_title('Most Economical Death Bowlers (Overs 16-20)', pad=20)
    ax.set_xlabel('Economy Rate')
    for bar in bars:
        ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_death_bowlers.png', dpi=300)
    plt.close()
