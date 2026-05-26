import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency
from scipy import stats
import warnings
import matplotlib as mpl

warnings.filterwarnings('ignore')

# Set visual theme
plt.style.use('dark_background')
mpl.rcParams['figure.facecolor'] = '#0D1117'
mpl.rcParams['axes.facecolor'] = '#0D1117'
mpl.rcParams['axes.edgecolor'] = '#30363D'
mpl.rcParams['text.color'] = '#C9D1D9'
mpl.rcParams['axes.labelcolor'] = '#C9D1D9'
mpl.rcParams['xtick.color'] = '#8B949E'
mpl.rcParams['ytick.color'] = '#8B949E'
mpl.rcParams['grid.color'] = '#30363D'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.titleweight'] = 'bold'

TEAM_COLORS = {
    'Chennai Super Kings': '#FFFF3C',
    'Delhi Capitals': '#00008B',
    'Gujarat Titans': '#1B2133',
    'Kolkata Knight Riders': '#3A225D',
    'Lucknow Super Giants': '#00BFFF',
    'Mumbai Indians': '#004BA0',
    'Punjab Kings': '#ED1B24',
    'Rajasthan Royals': '#EA1A85',
    'Royal Challengers Bengaluru': '#EC1C24',
    'Sunrisers Hyderabad': '#F26522',
    'Deccan Chargers': '#003366',
    'Gujarat Lions': '#FF7F00',
    'Kochi Tuskers Kerala': '#8A2BE2',
    'Pune Warriors': '#2F4F4F',
    'Rising Pune Supergiants': '#D11D70'
}

def ensure_dirs():
    dirs = [
        'output/charts/01_toss_myth',
        'output/charts/02_phase_impact',
        'output/charts/03_top_performers',
        'output/charts/04_venue_intelligence',
        'output/charts/05_clutch_factor',
        'output/charts/06_over_forensics',
        'output/charts/07_team_dna',
        'output/charts/08_win_probability',
        'output/charts/09_partnerships',
        'output/charts/10_choke_luck',
        'output/charts/11_extras_tax',
        'output/charts/12_batting_position',
        'output/charts/13_surprise_insight'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_and_clean_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path, low_memory=False)
    
    print("Cleaning data...")
    # Normalize season
    def fix_season(s):
        s = str(s)
        if '/' in s:
            return int(s.split('/')[0]) + (1 if '2020' in s else 0) # e.g. 2007/08 -> 2007/08 was actually 2008
        return int(s)
    
    # Actually the 2007/08 season was 2008. 2009/10 was 2010. Let's map exactly based on start year.
    season_map = {
        '2007/08': 2008,
        '2009/10': 2010,
        '2020/21': 2020
    }
    df['season_year'] = df['season'].map(lambda x: season_map.get(str(x), str(x)[:4])).astype(int)
    
    # Standardize team names
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
    }
    for col in ['team1', 'team2', 'toss_winner', 'winner', 'batting_team']:
        df[col] = df[col].replace(team_mapping)
        
    # Infer bowling team
    df['bowling_team'] = np.where(df['batting_team'] == df['team1'], df['team2'], df['team1'])
        
    # Fill city nulls
    venue_city_map = df.dropna(subset=['city']).groupby('venue')['city'].first().to_dict()
    # Add manual ones if still missing (e.g. Dubai, Sharjah)
    venue_city_map.update({
        'Dubai International Cricket Stadium': 'Dubai',
        'Sharjah Cricket Stadium': 'Sharjah',
        'Sheikh Zayed Stadium': 'Abu Dhabi'
    })
    df['city'] = df['city'].fillna(df['venue'].map(venue_city_map))
    
    # Feature Engineering
    df['phase'] = pd.cut(df['over'], bins=[-1, 5, 14, 19], labels=['Powerplay', 'Middle', 'Death'])
    df['is_boundary'] = df['runs_batter'].isin([4, 6]).astype(int)
    df['is_four'] = (df['runs_batter'] == 4).astype(int)
    df['is_six'] = (df['runs_batter'] == 6).astype(int)
    df['is_dot'] = (df['runs_total'] == 0).astype(int)
    df['is_legal_ball'] = ((df['extras_wides'] == 0) & (df['extras_noballs'] == 0)).astype(int)
    df['is_wicket'] = df['wicket_kind'].notna() & ~df['wicket_kind'].isin(['retired hurt', 'retired out', 'obstructing the field'])
    df['is_wicket'] = df['is_wicket'].astype(int)
    
    # Add ball number per innings
    df['legal_ball_count'] = df.groupby(['match_id', 'innings'])['is_legal_ball'].cumsum()
    
    return df

def get_match_summary(df):
    # Create match level summary
    match_df = df.groupby('match_id').first().reset_index()
    match_df = match_df[['match_id', 'season_year', 'date', 'venue', 'city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner', 'win_by_runs', 'win_by_wickets', 'player_of_match']]
    
    # Exclude no results
    match_df = match_df.dropna(subset=['winner'])
    
    # Determine who batted first
    def get_bat_first(row):
        if row['toss_decision'] == 'bat':
            return row['toss_winner']
        else:
            return row['team2'] if row['toss_winner'] == row['team1'] else row['team1']
            
    match_df['bat_first'] = match_df.apply(get_bat_first, axis=1)
    match_df['field_first'] = np.where(match_df['bat_first'] == match_df['team1'], match_df['team2'], match_df['team1'])
    match_df['toss_winner_won'] = (match_df['toss_winner'] == match_df['winner']).astype(int)
    match_df['bat_first_won'] = (match_df['bat_first'] == match_df['winner']).astype(int)
    
    return match_df

def module_01_toss_myth(df, match_df):
    print("Running Module 1: The Toss Myth...")
    out_dir = 'output/charts/01_toss_myth'
    
    # 1. Overall toss-win correlation
    overall_win_pct = match_df['toss_winner_won'].mean() * 100
    
    # Chi-squared test
    observed = pd.crosstab(match_df['toss_winner_won'], columns='count')
    # Expected is 50/50
    expected = [len(match_df)/2, len(match_df)/2]
    chi2, p_val = stats.chisquare(f_obs=observed['count'], f_exp=expected)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Lost Match', 'Won Match'], observed['count'], color=['#EC1C24', '#00BFFF'])
    ax.set_title(f'Does Winning Toss = Winning Match?\n(Overall Win %: {overall_win_pct:.1f}%)', pad=20)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{yval}\n({yval/len(match_df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
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
    
    # Annotate absolute percentages
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
    ax1.set_title('The Rise of the "Chase" Culture vs Actual Advantage', pad=20)
    ax1.set_xticks(season_toss['season_year'])
    ax1.set_xticklabels(season_toss['season_year'], rotation=45)
    
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/03_season_toss_trend.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    ensure_dirs()
    df = load_and_clean_data('data.csv')
    match_df = get_match_summary(df)
    
    module_01_toss_myth(df, match_df)
    print("Base script setup complete.")
