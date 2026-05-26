import pandas as pd
import matplotlib.pyplot as plt
import os
import theme

def run(df, match_df):
    print("Running Module 6: Over-by-Over Forensics...")
    out_dir = 'output/charts/06_over_forensics'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Analyze by over (0-19)
    over_stats = df.groupby('over').agg(
        total_runs=('runs_total', 'sum'),
        total_legal_balls=('is_legal_ball', 'sum'),
        total_wickets=('is_wicket', 'sum'),
        total_dots=('is_dot', 'sum'),
        total_balls=('match_id', 'count')
    ).reset_index()
    
    over_stats['run_rate'] = (over_stats['total_runs'] / over_stats['total_legal_balls']) * 6
    over_stats['wicket_prob'] = (over_stats['total_wickets'] / over_stats['total_balls']) * 100
    over_stats['dot_pct'] = (over_stats['total_dots'] / over_stats['total_balls']) * 100
    
    # 1. Acceleration Curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(over_stats['over'] + 1, over_stats['run_rate'], marker='o', color='#EA1A85', linewidth=3)
    
    # Shading phases
    ax.axvspan(0.5, 6.5, alpha=0.1, color='#FFD700', label='Powerplay')
    ax.axvspan(6.5, 15.5, alpha=0.1, color='#00BFFF', label='Middle')
    ax.axvspan(15.5, 20.5, alpha=0.1, color='#EC1C24', label='Death')
    
    ax.set_title('The Acceleration Curve: Over-by-Over Average Run Rate', pad=20)
    ax.set_xlabel('Over Number')
    ax.set_ylabel('Run Rate')
    ax.set_xticks(range(1, 21))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_acceleration_curve.png', dpi=300)
    plt.close()
    
    # 2. Risk vs Reward (Wicket Prob vs Dot Pct)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    ax1.bar(over_stats['over'] + 1, over_stats['wicket_prob'], color='#EC1C24', alpha=0.7, label='Wicket Probability (%)')
    ax2.plot(over_stats['over'] + 1, over_stats['dot_pct'], marker='s', color='#00BFFF', linewidth=2, label='Dot Ball %')
    
    ax1.set_title('Risk Profile: Wickets vs Dots by Over', pad=20)
    ax1.set_xlabel('Over Number')
    ax1.set_ylabel('Wicket Probability (%)', color='#EC1C24')
    ax2.set_ylabel('Dot Ball %', color='#00BFFF')
    ax1.set_xticks(range(1, 21))
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_risk_vs_reward.png', dpi=300)
    plt.close()
