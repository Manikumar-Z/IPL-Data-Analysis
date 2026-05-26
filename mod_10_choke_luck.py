import pandas as pd
import matplotlib.pyplot as plt
import os
import theme

def run(df, match_df):
    print("Running Module 10: Choke Factor & Luck Index...")
    out_dir = 'output/charts/10_choke_luck'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Filter close matches: decided by <= 10 runs OR <= 3 wickets (and > 0) OR Super Overs (if any)
    close_matches = match_df[
        (match_df['win_by_runs'] <= 10) & (match_df['win_by_runs'] > 0) | 
        ((match_df['win_by_wickets'] <= 3) & (match_df['win_by_wickets'] > 0)) |
        (match_df['win_by_runs'] == 0) & (match_df['win_by_wickets'] == 0) # Tied / Super over
    ]
    
    # Calculate Luck Index (Win % in close matches vs Overall Win %)
    team_matches = pd.concat([match_df['team1'], match_df['team2']]).value_counts()
    team_wins = match_df['winner'].value_counts()
    overall_win_pct = (team_wins / team_matches * 100).fillna(0)
    
    close_team_matches = pd.concat([close_matches['team1'], close_matches['team2']]).value_counts()
    close_team_wins = close_matches['winner'].value_counts()
    close_win_pct = (close_team_wins / close_team_matches * 100).fillna(0)
    
    luck_df = pd.DataFrame({
        'overall_win_pct': overall_win_pct,
        'close_win_pct': close_win_pct,
        'close_matches': close_team_matches
    }).dropna()
    
    luck_df = luck_df[luck_df['close_matches'] >= 10]
    luck_df['luck_index'] = luck_df['close_win_pct'] - luck_df['overall_win_pct']
    luck_df = luck_df.sort_values('luck_index', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#EC1C24' if x < 0 else '#00BFFF' for x in luck_df['luck_index']]
    ax.barh(luck_df.index, luck_df['luck_index'], color=colors)
    ax.set_title('The Luck Index: Over/Under-performance in Close Matches\n(Close Match Win % minus Overall Win %)', pad=20)
    ax.set_xlabel('% Deviation (Positive = Clutch/Lucky, Negative = Chokers/Unlucky)')
    ax.axvline(0, color='white', linestyle='--', alpha=0.5)
    
    for i, (val, name) in enumerate(zip(luck_df['luck_index'], luck_df.index)):
        offset = 0.5 if val >= 0 else -0.5
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, i, f'{val:+.1f}%', va='center', ha=ha, fontweight='bold', color='white')
        
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_luck_index.png', dpi=300)
    plt.close()
    
    # Choke analysis: Chasing teams needing <30 off last 3 overs but lost
    # We need to find the score at over 16 (end of over 16, i.e., 3 overs left)
    chase_df = df[df['innings'] == 2].copy()
    
    # Group by match and find runs needed at ball 102 (17.1)
    choke_stats = []
    
    for match_id in chase_df['match_id'].unique():
        match_data = chase_df[chase_df['match_id'] == match_id]
        if match_data['legal_ball_count'].max() >= 102:
            ball_102_data = match_data[match_data['legal_ball_count'] <= 102]
            runs_so_far = ball_102_data['runs_total'].sum()
            
            # Target
            first_inn = df[(df['match_id'] == match_id) & (df['innings'] == 1)]
            target = first_inn['runs_total'].sum() + 1
            
            runs_needed = target - runs_so_far
            if runs_needed > 0 and runs_needed <= 30:
                winner = match_df[match_df['match_id'] == match_id]['winner'].values[0]
                chasing_team = match_data['batting_team'].values[0]
                did_choke = 1 if winner != chasing_team else 0
                choke_stats.append({'team': chasing_team, 'choke': did_choke})
                
    choke_df = pd.DataFrame(choke_stats)
    if not choke_df.empty:
        choke_summary = choke_df.groupby('team')['choke'].agg(['sum', 'count']).reset_index()
        choke_summary = choke_summary[choke_summary['count'] >= 5]
        choke_summary['choke_pct'] = (choke_summary['sum'] / choke_summary['count']) * 100
        choke_summary = choke_summary.sort_values('choke_pct', ascending=False)
        
        if not choke_summary.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(choke_summary['team'][::-1], choke_summary['choke_pct'][::-1], color='#EA1A85')
            ax.set_title('The Choke Factor: % Losses When Needing <30 Runs from Last 3 Overs', pad=20)
            ax.set_xlabel('Choke Percentage (%)')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/02_choke_factor.png', dpi=300)
            plt.close()
