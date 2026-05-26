import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import theme

def run(df, match_df):
    print("Running Module 9: Partnership Network...")
    out_dir = 'output/charts/09_partnerships'
    os.makedirs(out_dir, exist_ok=True)
    theme.set_theme()
    
    # Calculate partnership runs
    # A partnership is defined by the match_id, innings, and the two batters currently at the crease
    df['pair'] = df.apply(lambda row: tuple(sorted([row['batter'], row['non_striker']])), axis=1)
    
    partnerships = df.groupby(['match_id', 'innings', 'pair']).agg(
        partnership_runs=('runs_total', 'sum')
    ).reset_index()
    
    # Aggregate pairs across all matches
    pair_stats = partnerships.groupby('pair').agg(
        total_runs=('partnership_runs', 'sum'),
        innings_together=('match_id', 'count')
    ).reset_index()
    
    pair_stats['average_stand'] = pair_stats['total_runs'] / pair_stats['innings_together']
    pair_stats = pair_stats[pair_stats['innings_together'] >= 10] # Filter out rare pairs
    pair_stats = pair_stats.sort_values('total_runs', ascending=False)
    
    # Top 20 pairs by total runs for Network Graph
    top_pairs = pair_stats.head(30)
    
    G = nx.Graph()
    for _, row in top_pairs.iterrows():
        b1, b2 = row['pair']
        G.add_edge(b1, b2, weight=row['total_runs'])
        
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] / 500 for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='#00BFFF', alpha=0.8, edgecolors='white', ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='#EA1A85', alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='white', ax=ax)
    
    ax.set_title('The Unsung Duos: Highest Scoring Partnership Networks', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_partnership_network.png', dpi=300)
    plt.close()
    
    # Bar chart for highest average stand (min 15 innings)
    avg_pairs = pair_stats[pair_stats['innings_together'] >= 15].sort_values('average_stand', ascending=False).head(15)
    avg_pairs['pair_name'] = avg_pairs['pair'].apply(lambda x: f"{x[0]} & {x[1]}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(avg_pairs['pair_name'][::-1], avg_pairs['average_stand'][::-1], color='#FFD700')
    ax.set_title('Highest Average Partnership Stands (Min 15 innings together)', pad=20)
    ax.set_xlabel('Average Runs per Stand')
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_average_stand.png', dpi=300)
    plt.close()
