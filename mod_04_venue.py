import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import theme

def run(df, match_df):
    print("Running Module 4: Venue Intelligence...")
    out_dir = 'output/charts/04_venue_intelligence'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # 1. Venue Scoring DNA
    # First innings scores
    first_inn = df[df['innings'] == 1].groupby('match_id').agg(
        runs=('runs_total', 'sum'),
        venue=('venue', 'first')
    ).reset_index()
    
    # Second innings scores
    second_inn = df[df['innings'] == 2].groupby('match_id').agg(
        runs=('runs_total', 'sum'),
        venue=('venue', 'first')
    ).reset_index()
    
    venue_stats = first_inn.groupby('venue').agg(
        avg_1st_inn=('runs', 'mean'),
        matches=('match_id', 'count')
    ).reset_index()
    
    venue_stats_2nd = second_inn.groupby('venue').agg(
        avg_2nd_inn=('runs', 'mean')
    ).reset_index()
    
    venue_stats = venue_stats.merge(venue_stats_2nd, on='venue')
    venue_stats = venue_stats[venue_stats['matches'] >= 15] # Only established venues
    
    # Scatter plot: 1st vs 2nd innings average
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=venue_stats, x='avg_1st_inn', y='avg_2nd_inn', size='matches', sizes=(50, 400), color='#EA1A85', alpha=0.8, ax=ax)
    
    # Diagonal line (where 1st inn = 2nd inn)
    min_val = min(venue_stats['avg_1st_inn'].min(), venue_stats['avg_2nd_inn'].min()) - 5
    max_val = max(venue_stats['avg_1st_inn'].max(), venue_stats['avg_2nd_inn'].max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.5, label='Equal Scoring Line')
    
    for i, row in venue_stats.iterrows():
        # Only annotate extreme venues to avoid clutter
        if abs(row['avg_1st_inn'] - row['avg_2nd_inn']) > 15 or row['avg_1st_inn'] > 175 or row['avg_1st_inn'] < 145:
            # Shorten venue names
            name = row['venue'].replace(' Stadium', '').replace(' Cricket', '').split(',')[0]
            ax.text(row['avg_1st_inn'], row['avg_2nd_inn'] + 1, name, fontsize=9, ha='center')
            
    ax.set_title('Venue Scoring DNA: Chasing Paradise vs Defending Fortresses', pad=20)
    ax.set_xlabel('Average 1st Innings Score')
    ax.set_ylabel('Average 2nd Innings Score')
    ax.text(min_val+5, max_val-10, 'Chasing Advantage\n(Dew Factor / Flat Tracks)', color='#00BFFF', fontweight='bold', alpha=0.7)
    ax.text(max_val-25, min_val+10, 'Defending Advantage\n(Pitch Degrades / Pressure)', color='#EC1C24', fontweight='bold', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_venue_scoring_dna.png', dpi=300)
    plt.close()
    
    # 2. Top 10 Highest and Lowest Scoring Venues
    venue_stats = venue_stats.sort_values('avg_1st_inn', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    top5 = venue_stats.head(5)
    bottom5 = venue_stats.tail(5).sort_values('avg_1st_inn')
    
    ax1.barh(top5['venue'][::-1], top5['avg_1st_inn'][::-1], color='#EC1C24')
    ax1.set_title('Batting Paradises (Highest Avg 1st Inn)')
    for bar in ax1.patches:
        ax1.text(bar.get_width() - 20, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.0f}", va='center', color='white', fontweight='bold')
        
    ax2.barh(bottom5['venue'][::-1], bottom5['avg_1st_inn'][::-1], color='#00BFFF')
    ax2.set_title('Bowler Friendly (Lowest Avg 1st Inn)')
    for bar in ax2.patches:
        ax2.text(bar.get_width() - 20, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.0f}", va='center', color='white', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_venue_extremes.png', dpi=300)
    plt.close()
