import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import theme

def run(df, match_df):
    print("Running Module 2: Phase Impact...")
    out_dir = 'output/charts/02_phase_impact'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Prepare innings level data
    innings_df = df.groupby(['match_id', 'innings', 'batting_team', 'bowling_team']).agg(
        runs_total=('runs_total', 'sum'),
        wickets=('is_wicket', 'sum')
    ).reset_index()
    
    # Phase runs
    phase_df = df.groupby(['match_id', 'innings', 'phase'])['runs_total'].sum().unstack(fill_value=0).reset_index()
    phase_df.columns = ['match_id', 'innings', 'PP_runs', 'Middle_runs', 'Death_runs']
    
    # Phase wickets
    phase_wkt_df = df.groupby(['match_id', 'innings', 'phase'])['is_wicket'].sum().unstack(fill_value=0).reset_index()
    phase_wkt_df.columns = ['match_id', 'innings', 'PP_wickets', 'Middle_wickets', 'Death_wickets']
    
    innings_data = innings_df.merge(phase_df, on=['match_id', 'innings']).merge(phase_wkt_df, on=['match_id', 'innings'])
    
    # Merge with match winner
    match_winners = match_df[['match_id', 'winner']]
    innings_data = innings_data.merge(match_winners, on='match_id', how='left')
    innings_data['is_winner'] = (innings_data['batting_team'] == innings_data['winner']).astype(int)
    
    # 1. Phase-wise run contribution
    phase_avgs = innings_data.groupby('is_winner')[['PP_runs', 'Middle_runs', 'Death_runs']].mean().T
    phase_avgs.columns = ['Losers', 'Winners']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(phase_avgs))
    width = 0.35
    ax.bar(x - width/2, phase_avgs['Losers'], width, label='Losers', color='#EC1C24')
    ax.bar(x + width/2, phase_avgs['Winners'], width, label='Winners', color='#00BFFF')
    
    ax.set_ylabel('Average Runs')
    ax.set_title('Average Runs per Phase: Winners vs Losers')
    ax.set_xticks(x)
    ax.set_xticklabels(['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)'])
    ax.legend()
    for i in x:
        ax.text(i - width/2, phase_avgs['Losers'].iloc[i] + 1, f"{phase_avgs['Losers'].iloc[i]:.1f}", ha='center')
        ax.text(i + width/2, phase_avgs['Winners'].iloc[i] + 1, f"{phase_avgs['Winners'].iloc[i]:.1f}", ha='center')
        
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_phase_runs_win_loss.png', dpi=300)
    plt.close()
    
    # 2. Correlation heatmap
    corr_cols = ['PP_runs', 'Middle_runs', 'Death_runs', 'PP_wickets', 'Middle_wickets', 'Death_wickets', 'is_winner']
    corr_matrix = innings_data[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix[['is_winner']].drop('is_winner').sort_values(by='is_winner', ascending=False), 
                annot=True, cmap='RdBu', center=0, vmin=-0.5, vmax=0.5, ax=ax)
    ax.set_title('Correlation with Winning Match', pad=20)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_phase_correlation.png', dpi=300)
    plt.close()
    
    # 3. PP wicket doom threshold
    pp_wkt_win_pct = innings_data.groupby('PP_wickets')['is_winner'].agg(['mean', 'count']).reset_index()
    pp_wkt_win_pct = pp_wkt_win_pct[pp_wkt_win_pct['count'] > 10]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(pp_wkt_win_pct['PP_wickets'], pp_wkt_win_pct['mean'] * 100, color='#8A2BE2')
    ax.axhline(50, color='white', linestyle='--', alpha=0.3)
    ax.set_xlabel('Wickets Lost in Powerplay')
    ax.set_ylabel('Win Probability (%)')
    ax.set_title('The PP Wicket Doom Threshold', pad=20)
    for bar, count in zip(bars, pp_wkt_win_pct['count']):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/03_pp_wicket_doom.png', dpi=300)
    plt.close()
