import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import theme

def run(df, match_df):
    print("Running Module 1: The Toss Myth...")
    out_dir = 'output/charts/01_toss_myth'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # 1. Overall toss-win correlation
    overall_win_pct = match_df['toss_winner_won'].mean() * 100
    observed = pd.crosstab(match_df['toss_winner_won'], columns='count')
    expected = [len(match_df)/2, len(match_df)/2]
    chi2, p_val = stats.chisquare(f_obs=observed['count'], f_exp=expected)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Lost Match', 'Won Match'], observed['count'].values.flatten(), color=['#EC1C24', '#00BFFF'])
    ax.set_title(f'Does Winning Toss = Winning Match?\n(Overall Win %: {overall_win_pct:.1f}%)', pad=20)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{int(yval)}\n({yval/len(match_df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
    ax.text(0.5, -0.15, f'Chi-square p-value: {p_val:.3f} ({"Significant" if p_val < 0.05 else "Not Significant at 5%"})', transform=ax.transAxes, ha='center', fontsize=12, color='#FFD700')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_overall_toss_win.png', dpi=300)
    plt.close()

    # 2. Venue specific toss advantage
    venue_counts = match_df['venue'].value_counts()
    top_venues = venue_counts[venue_counts >= 20].index
    venue_toss = match_df[match_df['venue'].isin(top_venues)].groupby('venue')['toss_winner_won'].agg(['mean', 'count']).reset_index()
    venue_toss['mean'] = venue_toss['mean'] * 100
    venue_toss = venue_toss.sort_values('mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#EC1C24' if x < 45 else '#00BFFF' if x > 55 else '#8B949E' for x in venue_toss['mean']]
    ax.barh(venue_toss['venue'], venue_toss['mean'] - 50, color=colors)
    ax.set_title('The Hidden Truth: Toss Advantage is Venue-Specific\n(% Deviation from 50/50 Baseline)', pad=20)
    ax.set_xlabel('Toss Winner Win % Deviation (0 = 50%)')
    ax.axvline(0, color='white', linestyle='--', alpha=0.5)
    
    for i, (val, name) in enumerate(zip(venue_toss['mean'], venue_toss['venue'])):
        offset = 1 if val >= 50 else -1
        ha = 'left' if val >= 50 else 'right'
        ax.text(val - 50 + offset, i, f'{val:.1f}%', va='center', ha=ha, fontweight='bold', color='white')
        
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_venue_toss_advantage.png', dpi=300)
    plt.close()
    
    # 3. Season evolution
    season_toss = match_df.groupby('season_year').agg(
        toss_win_pct=('toss_winner_won', lambda x: x.mean() * 100),
        field_first_pct=('toss_decision', lambda x: (x == 'field').mean() * 100)
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(season_toss['season_year'], season_toss['toss_win_pct'], marker='o', color='#00BFFF', linewidth=2, label='Toss Winner Win %')
    l2 = ax2.plot(season_toss['season_year'], season_toss['field_first_pct'], marker='s', color='#FFD700', linewidth=2, label='% Chose to Field')
    
    ax1.axhline(50, color='white', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Toss Winner Win %', color='#00BFFF')
    ax2.set_ylabel('% Captains Choosing to Field', color='#FFD700')
    ax1.set_title('The Rise of the "Chase Culture" vs Actual Advantage', pad=20)
    ax1.set_xticks(season_toss['season_year'].unique())
    ax1.set_xticklabels(season_toss['season_year'].unique(), rotation=45)
    
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/03_season_toss_trend.png', dpi=300)
    plt.close()
