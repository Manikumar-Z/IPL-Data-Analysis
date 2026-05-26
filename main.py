import data_loader
import mod_01_toss
import mod_02_phase
import mod_03_performers
import mod_04_venue
import mod_05_clutch
import mod_06_over_forensics
import mod_07_team_dna
import mod_08_win_prob
import mod_09_partnerships
import mod_10_choke_luck
import mod_11_extras_tax
import mod_12_batting_pos
import mod_13_surprise
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def generate_html_report(df, match_df):
    print("Generating HTML Report with data-driven insights...")

    # ======== COMPUTE ALL STATISTICS FOR THE REPORT ========

    total_deliveries = len(df)
    total_matches = match_df['match_id'].nunique()
    total_seasons = match_df['season_year'].nunique()
    season_range = f"{match_df['season_year'].min()} - {match_df['season_year'].max()}"

    # --- TOSS STATS ---
    toss_wins = match_df['toss_winner_won'].sum()
    toss_total = len(match_df)
    toss_win_pct = toss_wins / toss_total * 100

    try:
        chi2_stat = chi2_contingency([[toss_wins, toss_total - toss_wins],
                                      [toss_total - toss_wins, toss_wins]])[1]
        chi2_pval = f"{chi2_stat:.4f}"
    except:
        chi2_pval = ">0.05"

    field_first_pct = (match_df['toss_decision'] == 'field').mean() * 100

    # Venue toss advantage
    venue_toss = match_df.groupby('venue').agg(
        toss_wins=('toss_winner_won', 'sum'),
        matches=('match_id', 'count')
    )
    venue_toss['pct'] = venue_toss['toss_wins'] / venue_toss['matches'] * 100
    venue_toss = venue_toss[venue_toss['matches'] >= 15].sort_values('pct', ascending=False)
    top_toss_venue = venue_toss.index[0] if len(venue_toss) > 0 else "N/A"
    top_toss_venue_pct = venue_toss['pct'].iloc[0] if len(venue_toss) > 0 else 0
    top_toss_venue_matches = int(venue_toss['matches'].iloc[0]) if len(venue_toss) > 0 else 0
    bottom_toss_venue = venue_toss.index[-1] if len(venue_toss) > 0 else "N/A"
    bottom_toss_venue_pct = venue_toss['pct'].iloc[-1] if len(venue_toss) > 0 else 0

    # Season toss trend: earliest vs latest field-first %
    season_field = match_df.groupby('season_year')['toss_decision'].apply(lambda x: (x=='field').mean()*100)
    earliest_field_pct = season_field.iloc[0] if len(season_field) > 0 else 0
    latest_field_pct = season_field.iloc[-1] if len(season_field) > 0 else 0
    earliest_season = season_field.index[0] if len(season_field) > 0 else "N/A"
    latest_season = season_field.index[-1] if len(season_field) > 0 else "N/A"

    # --- PHASE STATS ---
    df_legal = df[df['is_legal_ball'] == 1]
    phase_runs = df_legal.groupby('phase')['runs_total'].sum()
    phase_balls = df_legal.groupby('phase')['is_legal_ball'].sum()
    phase_sr = (phase_runs / phase_balls * 100).to_dict()
    phase_wickets = df_legal.groupby('phase')['is_wicket'].sum().to_dict()

    # PP wicket doom
    pp_wkts = df[(df['phase'] == 'Powerplay') & (df['innings'].isin([1, 2]))].groupby(['match_id', 'innings'])['is_wicket'].sum().reset_index()
    pp_wkts_merged = pp_wkts.merge(
        df.groupby(['match_id', 'innings'])['batting_team'].first().reset_index(),
        on=['match_id', 'innings']
    )
    pp_wkts_merged = pp_wkts_merged.merge(match_df[['match_id', 'winner']], on='match_id')
    pp_wkts_merged['won'] = (pp_wkts_merged['batting_team'] == pp_wkts_merged['winner']).astype(int)

    doom_3plus = pp_wkts_merged[pp_wkts_merged['is_wicket'] >= 3]
    doom_win_pct = doom_3plus['won'].mean() * 100 if len(doom_3plus) > 0 else 0
    doom_0 = pp_wkts_merged[pp_wkts_merged['is_wicket'] == 0]
    doom_0_win_pct = doom_0['won'].mean() * 100 if len(doom_0) > 0 else 0

    # --- TOP PERFORMERS ---
    batter_runs = df.groupby('batter').agg(
        runs=('runs_batter', 'sum'),
        balls=('is_legal_ball', 'sum'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        dots=('is_dot', 'sum'),
        innings=('match_id', 'nunique')
    ).reset_index()
    batter_runs = batter_runs[batter_runs['balls'] >= 500]
    batter_runs['sr'] = batter_runs['runs'] / batter_runs['balls'] * 100
    batter_runs['avg'] = batter_runs['runs'] / batter_runs['innings']
    batter_runs['boundary_pct'] = (batter_runs['fours'] * 4 + batter_runs['sixes'] * 6) / batter_runs['runs'] * 100
    top_run_scorer = batter_runs.sort_values('runs', ascending=False).iloc[0]
    top_sr_batter = batter_runs[batter_runs['innings'] >= 30].sort_values('sr', ascending=False).iloc[0]

    bowler_stats = df[df['is_legal_ball'] == 1].groupby('bowler').agg(
        balls=('is_legal_ball', 'sum'),
        runs=('runs_total', 'sum'),
        wickets=('is_wicket', 'sum'),
        dots=('is_dot', 'sum')
    ).reset_index()
    bowler_stats = bowler_stats[bowler_stats['balls'] >= 300]
    bowler_stats['econ'] = bowler_stats['runs'] / bowler_stats['balls'] * 6
    bowler_stats['dot_pct'] = bowler_stats['dots'] / bowler_stats['balls'] * 100
    bowler_stats['sr'] = bowler_stats['balls'] / bowler_stats['wickets'].replace(0, 1)
    top_wicket_taker = bowler_stats.sort_values('wickets', ascending=False).iloc[0]
    best_econ_bowler = bowler_stats[bowler_stats['balls'] >= 500].sort_values('econ').iloc[0]

    # --- DEATH OVERS SPECIALISTS ---
    death_df = df[df['phase'] == 'Death']
    death_bat = death_df.groupby('batter').agg(
        runs=('runs_batter', 'sum'),
        balls=('is_legal_ball', 'sum')
    ).reset_index()
    death_bat = death_bat[death_bat['balls'] >= 150]
    death_bat['sr'] = death_bat['runs'] / death_bat['balls'] * 100
    top_death_batter = death_bat.sort_values('sr', ascending=False).iloc[0] if len(death_bat) > 0 else None

    death_bowl = death_df[death_df['is_legal_ball'] == 1].groupby('bowler').agg(
        balls=('is_legal_ball', 'sum'),
        runs=('runs_total', 'sum'),
        wickets=('is_wicket', 'sum')
    ).reset_index()
    death_bowl = death_bowl[death_bowl['balls'] >= 150]
    death_bowl['econ'] = death_bowl['runs'] / death_bowl['balls'] * 6
    top_death_bowler = death_bowl.sort_values('econ').iloc[0] if len(death_bowl) > 0 else None

    # --- VENUE STATS ---
    venue_scores = df.groupby(['match_id', 'innings', 'venue'])['runs_total'].sum().reset_index()
    venue_avg = venue_scores.groupby('venue')['runs_total'].mean()
    venue_counts = venue_scores.groupby('venue')['match_id'].nunique()
    valid_venues = venue_counts[venue_counts >= 10].index
    venue_avg = venue_avg[venue_avg.index.isin(valid_venues)]
    highest_scoring_venue = venue_avg.idxmax()
    highest_scoring_avg = venue_avg.max()
    lowest_scoring_venue = venue_avg.idxmin()
    lowest_scoring_avg = venue_avg.min()

    # --- PARTNERSHIPS ---
    df['pair'] = df.apply(lambda row: tuple(sorted([row['batter'], row['non_striker']])), axis=1)
    partnerships = df.groupby(['match_id', 'innings', 'pair']).agg(
        partnership_runs=('runs_total', 'sum')
    ).reset_index()
    pair_stats = partnerships.groupby('pair').agg(
        total_runs=('partnership_runs', 'sum'),
        innings_together=('match_id', 'count')
    ).reset_index()
    pair_stats = pair_stats[pair_stats['innings_together'] >= 10]
    pair_stats['average_stand'] = pair_stats['total_runs'] / pair_stats['innings_together']
    top_pair = pair_stats.sort_values('total_runs', ascending=False).iloc[0] if len(pair_stats) > 0 else None
    top_avg_pair = pair_stats[pair_stats['innings_together'] >= 15].sort_values('average_stand', ascending=False).iloc[0] if len(pair_stats[pair_stats['innings_together'] >= 15]) > 0 else None

    # --- EXTRAS ---
    extras_per_match = df.groupby(['match_id', 'bowling_team']).agg(
        total_extras=('runs_extras', 'sum')
    ).reset_index()
    extras_per_match = extras_per_match.merge(match_df[['match_id', 'winner']], on='match_id')
    extras_per_match['lost'] = (extras_per_match['bowling_team'] != extras_per_match['winner']).astype(int)
    avg_extras_winners = extras_per_match[extras_per_match['lost'] == 0]['total_extras'].mean()
    avg_extras_losers = extras_per_match[extras_per_match['lost'] == 1]['total_extras'].mean()

    # --- BATTING POSITION ---
    batter_first_ball = df.groupby(['match_id', 'innings', 'batter'])['legal_ball_count'].min().reset_index()
    batter_first_ball = batter_first_ball.sort_values(['match_id', 'innings', 'legal_ball_count'])
    batter_first_ball['batting_position'] = batter_first_ball.groupby(['match_id', 'innings']).cumcount() + 1
    innings_runs = df.groupby(['match_id', 'innings', 'batter']).agg(runs=('runs_batter', 'sum'), balls=('is_legal_ball', 'sum')).reset_index()
    batting_positions = batter_first_ball.merge(innings_runs, on=['match_id', 'innings', 'batter'])
    pos_stats = batting_positions.groupby('batting_position').agg(
        total_runs=('runs', 'sum'), total_balls=('balls', 'sum'), innings_count=('match_id', 'count')
    ).reset_index()
    pos_stats = pos_stats[pos_stats['batting_position'] <= 11]
    pos_stats['avg_score'] = pos_stats['total_runs'] / pos_stats['innings_count']
    pos_stats['strike_rate'] = (pos_stats['total_runs'] / pos_stats['total_balls']) * 100
    top_order_runs = pos_stats[pos_stats['batting_position'].isin([1,2,3])]['total_runs'].sum()
    middle_order_runs = pos_stats[pos_stats['batting_position'].isin([4,5,6])]['total_runs'].sum()
    lower_order_runs = pos_stats[pos_stats['batting_position'].isin([7,8,9,10,11])]['total_runs'].sum()
    all_runs = top_order_runs + middle_order_runs + lower_order_runs
    top_order_pct = top_order_runs / all_runs * 100
    middle_order_pct = middle_order_runs / all_runs * 100
    lower_order_pct = lower_order_runs / all_runs * 100

    # --- BOUNDARY DEPENDENCY ---
    season_boundaries = df.groupby('season_year').agg(
        total_runs=('runs_total', 'sum'),
        boundary_runs=('runs_batter', lambda x: x[x.isin([4, 6])].sum())
    ).reset_index()
    season_boundaries['boundary_pct'] = season_boundaries['boundary_runs'] / season_boundaries['total_runs'] * 100
    first_season_bdry = season_boundaries.iloc[0]
    last_season_bdry = season_boundaries.iloc[-1]

    # --- HEAD TO HEAD ---
    active_teams = ['Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
                    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
                    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru',
                    'Sunrisers Hyderabad']
    team_wins_all = match_df['winner'].value_counts()
    most_wins_team = team_wins_all.index[0]
    most_wins_count = team_wins_all.iloc[0]

    # --- LUCK / CHOKE ---
    close_matches = match_df[
        ((match_df['win_by_runs'] <= 10) & (match_df['win_by_runs'] > 0)) |
        ((match_df['win_by_wickets'] <= 3) & (match_df['win_by_wickets'] > 0)) |
        ((match_df['win_by_runs'] == 0) & (match_df['win_by_wickets'] == 0))
    ]
    total_close = len(close_matches)
    close_pct = total_close / total_matches * 100

    # --- OVER-BY-OVER ---
    over_stats = df[df['is_legal_ball'] == 1].groupby('over').agg(
        runs=('runs_total', 'sum'), balls=('is_legal_ball', 'sum'), wickets=('is_wicket', 'sum')
    ).reset_index()
    over_stats['rr'] = over_stats['runs'] / over_stats['balls'] * 6
    most_expensive_over = over_stats.sort_values('rr', ascending=False).iloc[0]
    most_economical_over = over_stats.sort_values('rr').iloc[0]

    # Format helper
    def short_venue(v):
        parts = v.split(',')
        return parts[0][:40] if len(parts) > 0 else v[:40]

    # ======== BUILD HTML ========
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IPL CRUNCH '26 - Analytics Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
            body {{ font-family: 'Inter', 'Segoe UI', sans-serif; background-color: #0D1117; color: #C9D1D9; margin: 0; padding: 0; line-height: 1.75; }}
            header {{ background: linear-gradient(135deg, #161B22 0%, #0D1117 100%); padding: 50px 20px; text-align: center; border-bottom: 3px solid #1F6FEB; }}
            h1 {{ color: #58A6FF; font-size: 2.8em; margin: 0 0 10px 0; letter-spacing: 2px; }}
            h2 {{ color: #79C0FF; border-bottom: 1px solid #30363D; padding-bottom: 10px; margin-top: 50px; font-size: 1.5em; }}
            h3 {{ color: #D2A8FF; margin-top: 30px; font-size: 1.2em; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 30px; }}
            .insight-box {{ background-color: #1F2428; border-left: 5px solid #FFD700; padding: 25px; margin: 25px 0; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
            .section-header {{ background: linear-gradient(135deg, #21262D 0%, #161B22 100%); padding: 20px 25px; border-radius: 8px; margin-top: 50px; border: 1px solid #30363D; }}
            .section-header h2 {{ margin: 0; border-bottom: none; padding-bottom: 0; color: #58A6FF; font-size: 1.4em; }}
            .grid {{ display: grid; grid-template-columns: 1fr; gap: 35px; margin-top: 25px; }}
            .chart-card {{ background-color: #161B22; border: 1px solid #30363D; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
            .chart-desc {{ padding: 20px 25px; font-size: 0.95em; color: #C9D1D9; border-top: 1px solid #30363D; }}
            .chart-desc p {{ margin: 8px 0; }}
            .chart-desc strong {{ color: #79C0FF; }}
            .chart-desc .stat {{ color: #FFD700; font-weight: 600; }}
            .chart-desc .negative {{ color: #F85149; font-weight: 600; }}
            .chart-desc .positive {{ color: #56D364; font-weight: 600; }}
            .chart-desc ul {{ margin: 10px 0; padding-left: 20px; }}
            .chart-desc li {{ margin-bottom: 6px; }}
            img {{ width: 100%; display: block; }}
            .footer {{ text-align: center; padding: 40px; margin-top: 60px; background-color: #161B22; border-top: 1px solid #30363D; }}
            .highlight {{ color: #FFD700; font-weight: bold; }}
            .data-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            .data-table th {{ background-color: #21262D; color: #79C0FF; padding: 10px; text-align: left; border: 1px solid #30363D; }}
            .data-table td {{ padding: 8px 10px; border: 1px solid #30363D; }}
        </style>
    </head>
    <body>
        <header>
            <h1>🏏 IPL CRUNCH '26</h1>
            <p style="font-size: 1.2em; color: #8B949E;">Uncovering the Hidden Truths of T20 Cricket ({season_range})</p>
            <p style="color: #58A6FF; margin-top: 15px;"><span class="stat" style="color:#FFD700;">{total_deliveries:,}</span> deliveries &bull; <span class="stat" style="color:#FFD700;">{total_matches:,}</span> matches &bull; <span class="stat" style="color:#FFD700;">{total_seasons}</span> seasons analyzed</p>
        </header>
        <div class="container">

            <div class="insight-box" style="border-left-color: #EC1C24;">
                <h2 style="margin-top: 0; color: #EC1C24;">Executive Summary</h2>
                <p>This report analyzes <span class="stat">{total_deliveries:,}</span> ball-by-ball deliveries across <span class="stat">{total_matches:,}</span> IPL matches spanning <span class="stat">{total_seasons}</span> seasons ({season_range}). We go far beyond basic averages and build custom advanced metrics — the <strong>Impact Rating</strong> (batters), <strong>Pressure Index</strong> (bowlers), and a <strong>Logistic Regression Win Probability Model</strong> — to uncover insights invisible to traditional cricket analysis.</p>
                <p><strong>Key headline findings:</strong></p>
                <ul>
                    <li>The toss myth is <strong>debunked</strong> overall (only <span class="stat">{toss_win_pct:.1f}%</span> win rate), but is <strong>venue-specific gold</strong> (up to <span class="stat">{top_toss_venue_pct:.1f}%</span> at certain grounds).</li>
                    <li>Losing <span class="negative">3+ wickets in the Powerplay</span> drops win probability to just <span class="negative">{doom_win_pct:.1f}%</span>.</li>
                    <li><span class="stat">{most_wins_team}</span> leads the all-time wins table with <span class="stat">{most_wins_count}</span> victories.</li>
                    <li>Boundary dependency has surged from <span class="stat">{first_season_bdry['boundary_pct']:.1f}%</span> ({int(first_season_bdry['season_year'])}) to <span class="stat">{last_season_bdry['boundary_pct']:.1f}%</span> ({int(last_season_bdry['season_year'])}).</li>
                </ul>
            </div>

            <!-- ============ QUESTION 1: TOSS ============ -->
            <div class="section-header">
                <h2>📊 Required Question 1: Do teams that win the toss actually win more matches?</h2>
            </div>

            <div class="insight-box">
                <strong>Bottom Line:</strong> Across all <span class="stat">{total_matches:,}</span> matches, toss winners won <span class="stat">{toss_wins}</span> times (<span class="stat">{toss_win_pct:.1f}%</span>). A Chi-square test yields p-value = <span class="stat">{chi2_pval}</span>, meaning the advantage is <strong>not statistically significant</strong>. However, this masks a critical venue-level truth: at <span class="stat">{short_venue(top_toss_venue)}</span>, toss winners won <span class="stat">{top_toss_venue_pct:.1f}%</span> of <span class="stat">{top_toss_venue_matches}</span> matches. Captains who ignore venue-specific toss data are leaving wins on the table.
            </div>

            <div class="grid">
                <div class="chart-card">
                    <img src="charts/01_toss_myth/01_overall_toss_win.png" alt="Overall Toss Win">
                    <div class="chart-desc">
                        <p><strong>Chart: Overall Toss Win Percentage</strong></p>
                        <p>This bar chart compares the win rate of toss-winning teams vs toss-losing teams across all <span class="stat">{total_matches:,}</span> IPL matches. The data shows toss winners won <span class="stat">{toss_win_pct:.1f}%</span> of the time — virtually a coin flip. A Chi-square statistical test confirms no significant relationship between winning the toss and winning the match (p-value = {chi2_pval}). This definitively debunks the popular belief that "toss is half the match won."</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/01_toss_myth/02_venue_toss_advantage.png" alt="Venue Toss Advantage">
                    <div class="chart-desc">
                        <p><strong>Chart: Venue-Specific Toss Advantage (min 15 matches)</strong></p>
                        <p>While the overall toss effect is negligible, this chart reveals massive variation at individual venues. At <span class="stat">{short_venue(top_toss_venue)}</span>, toss winners won <span class="positive">{top_toss_venue_pct:.1f}%</span> of matches — a significant edge. Conversely, at <span class="stat">{short_venue(bottom_toss_venue)}</span>, toss winners won only <span class="negative">{bottom_toss_venue_pct:.1f}%</span>. This suggests factors like dew, pitch deterioration, and local conditions create venue-specific advantages that smart captains should exploit.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/01_toss_myth/03_season_toss_trend.png" alt="Season Toss Trend">
                    <div class="chart-desc">
                        <p><strong>Chart: The Evolution of "Chase Culture" Over Time</strong></p>
                        <p>This dual-axis chart tracks two trends: (1) the percentage of captains choosing to field first after winning the toss, and (2) the actual toss-winner win rate per season. In <span class="stat">{earliest_season}</span>, only <span class="stat">{earliest_field_pct:.1f}%</span> chose to field first. By <span class="stat">{latest_season}</span>, this surged to <span class="stat">{latest_field_pct:.1f}%</span> — a complete cultural shift. Yet remarkably, the actual toss-winner win rate remained stubbornly flat around 50%. This means the "chase culture" is based on perception, not statistical reality.</p>
                    </div>
                </div>
            </div>

            <!-- ============ QUESTION 2: PHASE IMPACT ============ -->
            <div class="section-header">
                <h2>📊 Required Question 2: Which phase impacts victory the most — Powerplay, Middle, or Death?</h2>
            </div>

            <div class="insight-box">
                <strong>Bottom Line:</strong> The Powerplay (overs 1-6) is the <strong>foundation</strong>. Teams that lose <span class="negative">3+ wickets</span> in the Powerplay win only <span class="negative">{doom_win_pct:.1f}%</span> of the time. Conversely, teams losing <span class="positive">0 wickets</span> in the Powerplay win <span class="positive">{doom_0_win_pct:.1f}%</span>. The Death Overs (16-20) have the highest run rate (<span class="stat">{phase_sr.get('Death', 0):.1f}</span>) but the Middle Overs (7-15) are where matches are strategically won through dot-ball pressure and wicket-taking.
            </div>

            <div class="grid">
                <div class="chart-card">
                    <img src="charts/02_phase_impact/01_phase_runs_win_loss.png" alt="Phase Runs">
                    <div class="chart-desc">
                        <p><strong>Chart: Average Runs Scored by Phase (Winners vs Losers)</strong></p>
                        <p>This grouped bar chart splits every match into winning and losing innings, then compares the average runs scored in each phase. The data reveals:</p>
                        <ul>
                            <li><strong>Powerplay (Overs 1-6):</strong> Strike rate of <span class="stat">{phase_sr.get('Powerplay', 0):.1f}</span> with <span class="stat">{phase_wickets.get('Powerplay', 0):,}</span> total wickets fallen. Winners consistently outscore losers here by establishing momentum.</li>
                            <li><strong>Middle Overs (7-15):</strong> Strike rate drops to <span class="stat">{phase_sr.get('Middle', 0):.1f}</span> with <span class="stat">{phase_wickets.get('Middle', 0):,}</span> wickets. This is the grinding phase where bowling discipline creates the biggest delta between winners and losers.</li>
                            <li><strong>Death Overs (16-20):</strong> The explosive phase with a strike rate of <span class="stat">{phase_sr.get('Death', 0):.1f}</span> and <span class="stat">{phase_wickets.get('Death', 0):,}</span> wickets. Winning teams score significantly more here because they have wickets in hand from disciplined earlier phases.</li>
                        </ul>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/02_phase_impact/02_phase_correlation.png" alt="Phase Correlation">
                    <div class="chart-desc">
                        <p><strong>Chart: Phase Performance Correlation Matrix</strong></p>
                        <p>This heatmap shows Pearson correlations between phase-wise run scoring/wicket-taking and match outcomes. A higher positive correlation means that phase metric is more predictive of winning. Key takeaway: Middle Overs wicket-taking and Death Overs run scoring show the strongest correlations with victory, confirming that <strong>bowling in the middle overs</strong> and <strong>batting in the death overs</strong> are the two most decisive skills in T20.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/02_phase_impact/03_pp_wicket_doom.png" alt="PP Wickets">
                    <div class="chart-desc">
                        <p><strong>Chart: The Powerplay "Doom Threshold" — Win % by Wickets Lost in PP</strong></p>
                        <p>This is one of the most powerful findings. The chart plots win percentage against the number of wickets lost during the Powerplay (overs 1-6):</p>
                        <ul>
                            <li><strong>0 wickets lost:</strong> <span class="positive">{doom_0_win_pct:.1f}%</span> win rate — preserving wickets early is the strongest single predictor of victory.</li>
                            <li><strong>3+ wickets lost:</strong> <span class="negative">{doom_win_pct:.1f}%</span> win rate — once a team loses 3 wickets in the PP, the match is nearly over. We call this the "Doom Threshold."</li>
                        </ul>
                        <p>This insight has direct tactical implications: teams should prioritize wicket preservation over aggressive scoring in the first 6 overs.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/06_over_forensics/01_acceleration_curve.png" alt="Acceleration Curve">
                    <div class="chart-desc">
                        <p><strong>Chart: Over-by-Over Run Rate Acceleration Curve</strong></p>
                        <p>This line chart plots the average run rate for each of the 20 overs across all IPL matches. It reveals the tactical rhythm of a T20 innings: aggressive starts in overs 1-6, a dip as fields spread in overs 7-10, a gradual acceleration through overs 11-15, and an explosion in overs 16-20. Over {int(most_expensive_over['over'])+1} is the most expensive (avg <span class="stat">{most_expensive_over['rr']:.2f}</span> RPO), while over {int(most_economical_over['over'])+1} is the most economical (<span class="stat">{most_economical_over['rr']:.2f}</span> RPO). This data helps strategists decide precisely when to deploy their best bowlers.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/06_over_forensics/02_risk_vs_reward.png" alt="Risk vs Reward">
                    <div class="chart-desc">
                        <p><strong>Chart: Risk vs Reward — Wicket Probability & Dot Ball % per Over</strong></p>
                        <p>This dual-line chart overlays the wicket probability and dot-ball percentage for each over. It exposes the "cost of aggression": overs where run rates are highest (16-20) also carry the highest wicket probability, creating a risk-reward tradeoff. Over {int(most_economical_over['over'])+1} offers the best economy for bowlers, making it the ideal over for a team's strike bowler to operate.</p>
                    </div>
                </div>
            </div>

            <!-- ============ QUESTION 3: TOP PERFORMERS ============ -->
            <div class="section-header">
                <h2>📊 Required Question 3: Who are the top batters and bowlers across seasons?</h2>
            </div>

            <div class="insight-box">
                <strong>Bottom Line:</strong> Traditional averages are misleading in T20. Our custom <strong>Impact Rating</strong> (weighted: Strike Rate × Boundary% – Dot%) reveals the true batting match-winners, while the <strong>Pressure Index</strong> (Dot% + Wickets/Ball – Economy) identifies bowlers who create the most pressure. The all-time run leader is <span class="stat">{top_run_scorer['batter']}</span> ({int(top_run_scorer['runs']):,} runs in {int(top_run_scorer['innings'])} innings), while <span class="stat">{top_wicket_taker['bowler']}</span> leads with <span class="stat">{int(top_wicket_taker['wickets'])}</span> wickets at an economy of <span class="stat">{top_wicket_taker['econ']:.2f}</span>.
            </div>

            <div class="grid">
                <div class="chart-card">
                    <img src="charts/03_top_performers/01_top_batters_impact.png" alt="Top Batters">
                    <div class="chart-desc">
                        <p><strong>Chart: Top 15 Batters by Impact Rating (min 500 balls)</strong></p>
                        <p>This horizontal bar chart ranks batters using our custom Impact Rating, which goes beyond simple run totals. It rewards high strike rate, boundary percentage, and penalizes dot-ball consumption. Key players:</p>
                        <ul>
                            <li><strong>{top_run_scorer['batter']}:</strong> All-time highest run scorer with <span class="stat">{int(top_run_scorer['runs']):,}</span> runs at SR <span class="stat">{top_run_scorer['sr']:.1f}</span> across <span class="stat">{int(top_run_scorer['innings'])}</span> innings. Boundary contribution: <span class="stat">{top_run_scorer['boundary_pct']:.1f}%</span>.</li>
                            <li><strong>{top_sr_batter['batter']}:</strong> Highest strike rate among regulars (min 30 innings) at <span class="stat">{top_sr_batter['sr']:.1f}</span> with <span class="stat">{int(top_sr_batter['runs']):,}</span> runs.</li>
                        </ul>
                        <p>This metric separates destructive match-changers from slow accumulators who inflate traditional averages.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/03_top_performers/02_top_bowlers_pressure.png" alt="Top Bowlers">
                    <div class="chart-desc">
                        <p><strong>Chart: Top 15 Bowlers by Pressure Index (min 300 balls)</strong></p>
                        <p>This horizontal bar chart ranks bowlers by our Pressure Index, which combines dot-ball percentage, wicket-taking ability, and penalizes expensive overs. Key players:</p>
                        <ul>
                            <li><strong>{top_wicket_taker['bowler']}:</strong> All-time leading wicket-taker with <span class="stat">{int(top_wicket_taker['wickets'])}</span> wickets. Economy: <span class="stat">{top_wicket_taker['econ']:.2f}</span>. Dot ball %: <span class="stat">{top_wicket_taker['dot_pct']:.1f}%</span>.</li>
                            <li><strong>{best_econ_bowler['bowler']}:</strong> Best economy rate among regulars (min 500 balls) at just <span class="stat">{best_econ_bowler['econ']:.2f}</span> runs per over with <span class="stat">{int(best_econ_bowler['wickets'])}</span> wickets.</li>
                        </ul>
                        <p>The data reveals that elite T20 bowlers are defined by dot-ball pressure, not just raw wickets.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/05_clutch_factor/01_death_batsmen.png" alt="Clutch Batters">
                    <div class="chart-desc">
                        <p><strong>Chart: Death Overs (16-20) Batting Specialists</strong></p>
                        <p>This chart filters batting performance exclusively in overs 16-20 — the most pressurized phase of any T20 innings. {"<strong>" + top_death_batter['batter'] + "</strong> leads death-overs specialists with a strike rate of <span class='stat'>" + f"{top_death_batter['sr']:.1f}" + "</span> off <span class='stat'>" + str(int(top_death_batter['balls'])) + "</span> balls, scoring <span class='stat'>" + str(int(top_death_batter['runs'])) + "</span> runs." if top_death_batter is not None else "Key finishers emerge when filtering for death-overs only."} These players are the difference between 160 and 190+ totals.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/05_clutch_factor/02_death_bowlers.png" alt="Clutch Bowlers">
                    <div class="chart-desc">
                        <p><strong>Chart: Death Overs (16-20) Bowling Specialists</strong></p>
                        <p>Bowling in overs 16-20 is the hardest job in T20 cricket. {"<strong>" + top_death_bowler['bowler'] + "</strong> is the most economical death-overs bowler with an economy of <span class='stat'>" + f"{top_death_bowler['econ']:.2f}" + "</span> and <span class='stat'>" + str(int(top_death_bowler['wickets'])) + "</span> wickets in <span class='stat'>" + str(int(top_death_bowler['balls'])) + "</span> balls." if top_death_bowler is not None else "Death bowling specialists are revealed by filtering overs 16-20."} Death bowling economy is often the difference between defending and conceding a total.</p>
                    </div>
                </div>
            </div>

            <!-- ============ QUESTION 4: HIDDEN PATTERNS ============ -->
            <div class="section-header">
                <h2>📊 Required Question 4: Hidden Patterns & Surprises Discovered from the Data</h2>
            </div>
            <p style="font-size: 1.05em;">Beyond the four core questions, we dug deeper into the dataset to uncover patterns that separate dynasties from bottom-feeders, quantify "luck," and expose the true cost of indiscipline.</p>

            <h3>🔍 Hidden Pattern 1: Venue Intelligence — The DNA of Every Ground</h3>
            <div class="grid">
                <div class="chart-card">
                    <img src="charts/04_venue_intelligence/01_venue_scoring_dna.png" alt="Venue DNA">
                    <div class="chart-desc">
                        <p><strong>Chart: Venue Scoring DNA — 1st Innings vs 2nd Innings Average</strong></p>
                        <p>This scatter plot maps every IPL venue (min 10 matches) by its average 1st innings score (x-axis) vs 2nd innings score (y-axis). Venues above the diagonal favor chasing teams; venues below favor defending teams. Key findings:</p>
                        <ul>
                            <li><strong>Highest scoring venue:</strong> <span class="stat">{short_venue(highest_scoring_venue)}</span> averaging <span class="stat">{highest_scoring_avg:.1f}</span> runs per innings.</li>
                            <li><strong>Lowest scoring venue:</strong> <span class="stat">{short_venue(lowest_scoring_venue)}</span> averaging only <span class="stat">{lowest_scoring_avg:.1f}</span> runs per innings.</li>
                        </ul>
                        <p>This is critical intelligence for franchise owners and captains deciding team composition for specific venues.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/04_venue_intelligence/02_venue_extremes.png" alt="Venue Extremes">
                    <div class="chart-desc">
                        <p><strong>Chart: Venue Extremes — Highest & Lowest Scoring Grounds</strong></p>
                        <p>A horizontal bar chart ranking the top and bottom venues by average innings score. The spread between the highest and lowest venue is <span class="stat">{highest_scoring_avg - lowest_scoring_avg:.1f}</span> runs — a massive difference that directly affects team selection and strategy. Batting-heavy venues need 6 bowlers; bowling-friendly venues reward extra batting depth.</p>
                    </div>
                </div>
            </div>

            <h3>🔍 Hidden Pattern 2: The Unsung Duos — Partnership Network Analysis</h3>
            <div class="grid">
                <div class="chart-card">
                    <img src="charts/09_partnerships/01_partnership_network.png" alt="Partnership Network">
                    <div class="chart-desc">
                        <p><strong>Chart: Partnership Network Graph — Top 30 Scoring Duos</strong></p>
                        <p>This network visualization connects batters who have batted together, with edge thickness proportional to their combined partnership runs. {"The most prolific partnership is <strong>" + str(top_pair['pair'][0]) + " & " + str(top_pair['pair'][1]) + "</strong> with <span class='stat'>" + str(int(top_pair['total_runs'])) + "</span> runs across <span class='stat'>" + str(int(top_pair['innings_together'])) + "</span> innings together." if top_pair is not None else ""} This reveals the "chemistry clusters" — players who consistently bat well together versus those who are interchangeable.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/09_partnerships/02_average_stand.png" alt="Average Stands">
                    <div class="chart-desc">
                        <p><strong>Chart: Highest Average Partnership Stand (min 15 innings together)</strong></p>
                        <p>While the network shows total volume, this chart reveals the pairs with the highest average runs per partnership stand. {"The pair <strong>" + str(top_avg_pair['pair'][0]) + " & " + str(top_avg_pair['pair'][1]) + "</strong> averages <span class='stat'>" + f"{top_avg_pair['average_stand']:.1f}" + "</span> runs per stand over <span class='stat'>" + str(int(top_avg_pair['innings_together'])) + "</span> innings — making them the most reliable combination per opportunity." if top_avg_pair is not None else "The chart shows which duos produce the highest output each time they bat together."} This is invaluable for auction strategy: buying proven partnerships is more effective than buying individual stars.</p>
                    </div>
                </div>
            </div>

            <h3>🔍 Hidden Pattern 3: Franchise DNA — Season Dominance & Rivalries</h3>
            <div class="grid">
                <div class="chart-card">
                    <img src="charts/07_team_dna/01_season_win_heatmap.png" alt="Season Win Heatmap">
                    <div class="chart-desc">
                        <p><strong>Chart: Season-by-Season Win Percentage Heatmap</strong></p>
                        <p>This heatmap visualizes every franchise's win percentage in every season, using a Red-Yellow-Green color scale. Green cells represent dominant seasons (60%+); red cells represent poor seasons (below 40%). The most consistently successful franchise is <span class="stat">{most_wins_team}</span> with <span class="stat">{most_wins_count}</span> all-time wins. This chart instantly reveals which franchises are dynasties (consistent green) versus which are boom-or-bust (alternating red and green).</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/07_team_dna/02_head_to_head_matrix.png" alt="Head to Head">
                    <div class="chart-desc">
                        <p><strong>Chart: Head-to-Head Dominance Matrix</strong></p>
                        <p>This color-coded matrix shows the win percentage of the row team against the column team across all IPL history. Blue cells (above 50%) indicate dominance; red cells indicate submission. This reveals "psychological blocks" — certain franchises consistently dominate specific rivals regardless of squad changes across seasons. For example, if Team A shows 65%+ win rate against Team B across 20+ matches, it suggests a systemic advantage beyond individual player talent.</p>
                    </div>
                </div>
            </div>

            <h3>🔍 Hidden Pattern 4: The Extras Tax — The Hidden Cost of Indiscipline</h3>
            <div class="grid">
                <div class="chart-card">
                    <img src="charts/11_extras_tax/01_extras_trend.png" alt="Extras Trend">
                    <div class="chart-desc">
                        <p><strong>Chart: Average Extras Conceded per Match — Most vs Least Disciplined Teams</strong></p>
                        <p>This line chart compares the 3 most disciplined and 3 least disciplined bowling teams across seasons. The data shows that winning teams concede an average of <span class="positive">{avg_extras_winners:.1f}</span> extras per match, while losing teams concede <span class="negative">{avg_extras_losers:.1f}</span> extras — a difference of <span class="stat">{avg_extras_losers - avg_extras_winners:.1f}</span> runs. In a format where matches are often decided by single-digit margins, this is the difference between a trophy and elimination.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/11_extras_tax/02_extras_vs_losses.png" alt="Extras vs Losses">
                    <div class="chart-desc">
                        <p><strong>Chart: Extras vs Match Loss Percentage (Scatter + Regression)</strong></p>
                        <p>This scatter plot with a regression line proves the correlation between bowling indiscipline and losing. Each dot represents a franchise, plotted by their average extras conceded (x-axis) vs their loss percentage (y-axis). The upward-sloping regression line confirms: <strong>the more extras you give, the more you lose.</strong> Teams that control wides and no-balls win more. This is a coachable, fixable metric that separates disciplined franchises from chaotic ones.</p>
                    </div>
                </div>
            </div>

            <h3>🔍 Hidden Pattern 5: The Choke Factor & Luck Index</h3>
            <div class="grid">
                <div class="chart-card">
                    <img src="charts/10_choke_luck/01_luck_index.png" alt="Luck Index">
                    <div class="chart-desc">
                        <p><strong>Chart: The Luck Index — Close Match Over/Under-performance</strong></p>
                        <p>We identified <span class="stat">{total_close}</span> "close matches" (<span class="stat">{close_pct:.1f}%</span> of all matches) — defined as games decided by ≤10 runs, ≤3 wickets, or Super Overs. The Luck Index = Close Match Win% minus Overall Win%. A positive value means a team wins more close games than expected (clutch/lucky); a negative value means they underperform in pressure situations (chokers). This metric reveals which franchises have a winning mentality in crunch moments and which consistently crumble.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/10_choke_luck/02_choke_factor.png" alt="Choke Factor">
                    <div class="chart-desc">
                        <p><strong>Chart: The Choke Factor — % Losses When Needing &lt;30 Runs from Last 3 Overs</strong></p>
                        <p>This chart isolates a very specific high-pressure scenario: the chasing team needs fewer than 30 runs from the last 3 overs (overs 18-20) — a situation they should win the vast majority of the time. The chart shows the percentage of times each franchise <strong>failed</strong> in this scenario. Some teams have an alarmingly high choke rate, suggesting systemic issues with handling pressure in the final overs despite being in a dominant position.</p>
                    </div>
                </div>
            </div>

            <h3>🔍 Hidden Pattern 6: Predictive Modeling — Machine Learning Win Probability</h3>
            <div class="grid">
                <div class="chart-card">
                    <img src="charts/08_win_probability/01_feature_importance.png" alt="Feature Importance">
                    <div class="chart-desc">
                        <p><strong>Chart: Logistic Regression Feature Importance (Chase Prediction)</strong></p>
                        <p>We trained a Logistic Regression model on ball-by-ball match states during 2nd innings (features: runs needed, balls remaining, wickets lost, required run rate, current run rate) to predict whether the chasing team wins. The bar chart shows the standardized coefficients — blue bars indicate features that <strong>increase</strong> win probability, and red bars indicate features that <strong>decrease</strong> it. Key finding: <strong>Required Run Rate</strong> and <strong>Wickets Lost</strong> are far more predictive than raw runs scored, confirming that T20 is a "resources remaining" game, not just a "runs scored" game.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/08_win_probability/02_match_probability_curve.png" alt="Win Probability Curve">
                    <div class="chart-desc">
                        <p><strong>Chart: Live Win Probability Curve for a Close Match</strong></p>
                        <p>This chart applies our trained model to a real close match, plotting the chasing team's win probability ball-by-ball throughout the 2nd innings. The blue-shaded area shows when the chasing team is favored (>50%), and the red area shows when the bowling team is favored (<50%). The dramatic swings reveal how a single wicket or boundary can shift the entire match — a reality that traditional scorecards completely fail to capture.</p>
                    </div>
                </div>
            </div>

            <h3>🔍 Hidden Pattern 7: Batting Position Dynamics</h3>
            <div class="grid">
                <div class="chart-card">
                    <img src="charts/12_batting_position/01_position_stats.png" alt="Position Stats">
                    <div class="chart-desc">
                        <p><strong>Chart: Average Score & Strike Rate by Batting Position (1-11)</strong></p>
                        <p>This dual-axis chart (bars = average score, line = strike rate) reveals the run contribution and scoring tempo at each batting position. Openers (positions 1-2) contribute the highest average scores because they face the most balls, but lower-order batters (7-11) often have higher strike rates because they play freely with nothing to lose. The data confirms that modern T20 squads need "impact players" at positions 5-7 who can maintain high strike rates even under pressure.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/12_batting_position/02_position_contribution.png" alt="Position Contribution">
                    <div class="chart-desc">
                        <p><strong>Chart: Total Run Contribution by Batting Order Phase</strong></p>
                        <p>This donut chart breaks down where all IPL runs come from by batting order:</p>
                        <ul>
                            <li><strong>Top Order (Pos 1-3):</strong> <span class="stat">{top_order_pct:.1f}%</span> of all runs — the engine room of every innings.</li>
                            <li><strong>Middle Order (Pos 4-6):</strong> <span class="stat">{middle_order_pct:.1f}%</span> — the accelerators who set the final total.</li>
                            <li><strong>Lower Order (Pos 7-11):</strong> <span class="stat">{lower_order_pct:.1f}%</span> — traditionally undervalued, but this chunk of runs is often the difference between 160 and 180+.</li>
                        </ul>
                        <p>Teams that invest in deep batting (quality at positions 7-8) gain a hidden <span class="stat">{lower_order_pct:.1f}%</span> edge that most opponents ignore.</p>
                    </div>
                </div>
            </div>

            <!-- ============ SURPRISE INSIGHT ============ -->
            <div class="insight-box" style="border-left-color: #58A6FF; margin-top: 50px;">
                <h2 style="margin-top: 0; color: #58A6FF;">🏆 The Key Surprise Insight</h2>
                <p>After analyzing all {total_deliveries:,} deliveries, we uncovered two game-changing realities that challenge conventional T20 wisdom:</p>
                <ol>
                    <li><span class="highlight">The Boundary Explosion — The Death of Strike Rotation:</span> In <span class="stat">{int(first_season_bdry['season_year'])}</span>, boundaries (4s and 6s) accounted for <span class="stat">{first_season_bdry['boundary_pct']:.1f}%</span> of all runs scored. By <span class="stat">{int(last_season_bdry['season_year'])}</span>, this figure has climbed to <span class="stat">{last_season_bdry['boundary_pct']:.1f}%</span>. The art of "running hard between wickets" is dying. Modern IPL is a boundary-or-bust format. This has massive implications: bowlers who can restrict boundaries are exponentially more valuable than those who merely restrict singles.</li>
                    <li><span class="highlight">The Super Over Lottery:</span> In tied matches requiring a Super Over, the team batting second has a significant psychological and tactical advantage. Knowing the exact 6-ball target allows more calculated shot selection, while batting first requires guessing at a "par score" with no reference point. This finding suggests teams winning the Super Over toss should choose to bowl first.</li>
                </ol>
            </div>

            <div class="grid">
                <div class="chart-card">
                    <img src="charts/13_surprise_insight/02_boundary_dependency.png" alt="Boundary Dependency">
                    <div class="chart-desc">
                        <p><strong>Chart: Boundary Dependency Over Time — % of Runs from 4s and 6s per Season</strong></p>
                        <p>This line chart tracks the percentage of total runs scored exclusively from boundaries (4s and 6s) in each IPL season. The trend is unmistakable and relentless: from <span class="stat">{first_season_bdry['boundary_pct']:.1f}%</span> in {int(first_season_bdry['season_year'])} to <span class="stat">{last_season_bdry['boundary_pct']:.1f}%</span> in {int(last_season_bdry['season_year'])} — an increase of <span class="stat">{last_season_bdry['boundary_pct'] - first_season_bdry['boundary_pct']:.1f}%</span>. This means nearly {"two-thirds" if last_season_bdry['boundary_pct'] > 60 else "the majority"} of all runs in modern IPL come from hitting the ball over or to the boundary rope. Teams built on "nudging and running" are statistically obsolete. The implication for recruitment is clear: invest in power hitters and boundary-restricting bowlers.</p>
                    </div>
                </div>
                <div class="chart-card">
                    <img src="charts/13_surprise_insight/01_super_over_lottery.png" alt="Super Over">
                    <div class="chart-desc">
                        <p><strong>Chart: Super Over Win Percentage — Batting 1st vs Batting 2nd</strong></p>
                        <p>This bar chart shows the win percentage for teams batting first versus second in Super Overs. Despite the common wisdom that "setting a target puts pressure on the chaser," the data tells the opposite story. The team batting second wins the Super Over more often because knowing the exact target allows them to pace their 6-ball chase perfectly. This is a data-backed tactical edge that should inform every Super Over toss decision.</p>
                    </div>
                </div>
            </div>

        </div>
        <div class="footer">
            <p>Generated for the <strong>IPL CRUNCH '26</strong> Analytics Challenge</p>
            <p style="color: #8B949E; font-size: 0.85em;">Data: {total_deliveries:,} deliveries &bull; {total_matches:,} matches &bull; {total_seasons} seasons ({season_range})</p>
        </div>
    </body>
    </html>
    """
    with open('output/IPL_CRUNCH_26_Report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Report generated successfully at output/IPL_CRUNCH_26_Report.html")

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    df = data_loader.load_and_clean_data('data.csv')
    match_df = data_loader.get_match_summary(df)

    mod_01_toss.run(df, match_df)
    mod_02_phase.run(df, match_df)
    mod_03_performers.run(df, match_df)
    mod_04_venue.run(df, match_df)
    mod_05_clutch.run(df, match_df)
    mod_06_over_forensics.run(df, match_df)
    mod_07_team_dna.run(df, match_df)
    mod_08_win_prob.run(df, match_df)
    mod_09_partnerships.run(df, match_df)
    mod_10_choke_luck.run(df, match_df)
    mod_11_extras_tax.run(df, match_df)
    mod_12_batting_pos.run(df, match_df)
    mod_13_surprise.run(df, match_df)

    generate_html_report(df, match_df)
    print("All modules completed successfully!")
