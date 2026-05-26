import matplotlib.pyplot as plt
import matplotlib as mpl

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

def set_theme():
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
