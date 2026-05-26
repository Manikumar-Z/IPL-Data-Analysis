import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import theme

def run(df, match_df):
    print("Running Module 13: Surprise Insight...")
    out_dir = 'output/charts/13_surprise_insight'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # 1. Super Over Lottery
    super_overs = df[df['innings'] > 2].groupby(['match_id', 'innings', 'batting_team']).agg(
        runs=('runs_total', 'sum')
    ).reset_index()
    
    if not super_overs.empty:
        # Match winners for super over matches
        so_matches = super_overs['match_id'].unique()
        so_winners = match_df[match_df['match_id'].isin(so_matches)][['match_id', 'winner']]
        
        # Determine who batted second in the super over (innings 4)
        inn4 = super_overs[super_overs['innings'] == 4]
        if not inn4.empty:
            inn4 = inn4.merge(so_winners, on='match_id')
            inn4['won_super_over'] = (inn4['batting_team'] == inn4['winner']).astype(int)
            
            win_pct = inn4['won_super_over'].mean() * 100
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(['Batted 1st in Super Over', 'Batted 2nd in Super Over'], [100-win_pct, win_pct], color=['#EC1C24', '#00BFFF'])
            ax.set_title('The Super Over Lottery\nWho wins the one-over eliminator?', pad=20)
            ax.set_ylabel('Win Percentage (%)')
            
            for bar in ax.patches:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', fontweight='bold')
                
            plt.tight_layout()
            plt.savefig(f'{out_dir}/01_super_over_lottery.png', dpi=300)
            plt.close()
            
    # 2. Boundary Dependency Explosion
    season_boundaries = df.groupby('season_year').agg(
        total_runs=('runs_total', 'sum'),
        boundary_runs=('runs_batter', lambda x: x[x.isin([4, 6])].sum())
    ).reset_index()
    
    season_boundaries['boundary_pct'] = (season_boundaries['boundary_runs'] / season_boundaries['total_runs']) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(season_boundaries['season_year'], season_boundaries['boundary_pct'], marker='o', linewidth=3, color='#FFD700')
    ax.set_title('The Death of Strike Rotation: Boundary Dependency Over Time', pad=20)
    ax.set_xlabel('Season')
    ax.set_ylabel('% of Runs from Boundaries (4s and 6s)')
    ax.set_xticks(season_boundaries['season_year'].unique())
    ax.set_xticklabels(season_boundaries['season_year'].unique(), rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate start and end
    start_pct = season_boundaries['boundary_pct'].iloc[0]
    end_pct = season_boundaries['boundary_pct'].iloc[-1]
    diff = end_pct - start_pct
    
    ax.text(season_boundaries['season_year'].iloc[-3], end_pct + 1, f'+{diff:.1f}% Increase\nSince Inception', color='#00BFFF', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_boundary_dependency.png', dpi=300)
    plt.close()
