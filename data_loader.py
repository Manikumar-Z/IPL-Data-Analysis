import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path, low_memory=False)
    
    print("Cleaning data...")
    # Map season to start year
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
    
    df['legal_ball_count'] = df.groupby(['match_id', 'innings'])['is_legal_ball'].cumsum()
    
    return df

def get_match_summary(df):
    match_df = df.groupby('match_id').first().reset_index()
    match_df = match_df[['match_id', 'season_year', 'date', 'venue', 'city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner', 'win_by_runs', 'win_by_wickets', 'player_of_match']]
    
    match_df = match_df.dropna(subset=['winner'])
    
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
