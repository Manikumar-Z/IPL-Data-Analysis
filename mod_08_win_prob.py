import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import theme

def run(df, match_df):
    print("Running Module 8: Win Probability Model...")
    out_dir = 'output/charts/08_win_probability'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Focus on 2nd innings (chasing) for probability modeling
    chase_df = df[df['innings'] == 2].copy()
    
    # We need the target score from the 1st innings
    first_inn_scores = df[df['innings'] == 1].groupby('match_id')['runs_total'].sum().reset_index()
    first_inn_scores.rename(columns={'runs_total': 'target'}, inplace=True)
    first_inn_scores['target'] = first_inn_scores['target'] + 1 # runs needed to win
    
    chase_df = chase_df.merge(first_inn_scores, on='match_id')
    
    # Target variable
    chase_df['is_winner'] = (chase_df['batting_team'] == chase_df['winner']).astype(int)
    
    # Calculate cumulative state at each ball
    chase_df['runs_scored'] = chase_df.groupby('match_id')['runs_total'].cumsum()
    chase_df['wickets_lost'] = chase_df.groupby('match_id')['is_wicket'].cumsum()
    chase_df['balls_bowled'] = chase_df.groupby('match_id')['is_legal_ball'].cumsum()
    
    chase_df['runs_needed'] = chase_df['target'] - chase_df['runs_scored']
    chase_df['balls_remaining'] = 120 - chase_df['balls_bowled']
    chase_df['balls_remaining'] = np.where(chase_df['balls_remaining'] < 1, 1, chase_df['balls_remaining']) # avoid div by zero
    
    chase_df['required_rate'] = (chase_df['runs_needed'] / chase_df['balls_remaining']) * 6
    chase_df['current_rate'] = (chase_df['runs_scored'] / chase_df['balls_bowled'].replace(0, 1)) * 6
    
    # Filter features
    features = ['runs_needed', 'balls_remaining', 'wickets_lost', 'required_rate', 'current_rate']
    model_df = chase_df.dropna(subset=features + ['is_winner'])
    
    # Split train (before 2025) / test (2025/26)
    train = model_df[model_df['season_year'] < 2025]
    test = model_df[model_df['season_year'] >= 2025]
    
    # Fallback to older test set if recent not enough
    if len(test) < 1000:
        train = model_df[model_df['season_year'] < 2023]
        test = model_df[model_df['season_year'] >= 2023]
    
    X_train = train[features]
    y_train = train['is_winner']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Feature importance
    importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
    importance['Abs_Coeff'] = importance['Coefficient'].abs()
    importance = importance.sort_values('Abs_Coeff', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#00BFFF' if x > 0 else '#EC1C24' for x in importance['Coefficient']]
    ax.barh(importance['Feature'][::-1], importance['Coefficient'][::-1], color=colors[::-1])
    ax.set_title('Logistic Regression Feature Importance\n(Predicting Chase Success)', pad=20)
    ax.axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_feature_importance.png', dpi=300)
    plt.close()
    
    # Plot win prob curve for a specific iconic match
    # Find a match where chasing team won on last ball or lost narrowly
    close_matches = match_df[(match_df['win_by_runs'] <= 5) | ((match_df['win_by_wickets'] <= 3) & (match_df['win_by_wickets'] > 0))]
    if not close_matches.empty:
        example_match = close_matches.iloc[0]['match_id']
        match_data = model_df[model_df['match_id'] == example_match].sort_values('balls_bowled')
        
        if not match_data.empty:
            X_match = scaler.transform(match_data[features])
            match_data['win_prob'] = model.predict_proba(X_match)[:, 1] * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(match_data['balls_bowled'], match_data['win_prob'], color='#00BFFF', linewidth=2)
            ax.axhline(50, color='white', linestyle='--', alpha=0.3)
            
            bat_team = match_data['batting_team'].iloc[0]
            winner = match_data['winner'].iloc[0]
            
            ax.set_title(f'Live Win Probability Curve\n{bat_team} Chasing (Winner: {winner})', pad=20)
            ax.set_xlabel('Balls Bowled in 2nd Innings')
            ax.set_ylabel('Win Probability (%)')
            ax.set_ylim(0, 100)
            ax.fill_between(match_data['balls_bowled'], match_data['win_prob'], 50, where=(match_data['win_prob'] >= 50), interpolate=True, color='#00BFFF', alpha=0.2)
            ax.fill_between(match_data['balls_bowled'], match_data['win_prob'], 50, where=(match_data['win_prob'] < 50), interpolate=True, color='#EC1C24', alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(f'{out_dir}/02_match_probability_curve.png', dpi=300)
            plt.close()
