#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# In[7]:


url = 'https://www.sports-reference.com/cbb/conferences/atlantic-sun/men/2025.html'
tables = pd.read_html(url)
asun = tables[2]
asun.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in asun.columns]
asun['Conf %'] = asun['Conf._W'] / (asun['Conf._W'] + asun['Conf._L'])
asun['Conf %'] = asun['Conf %'].fillna(0)
asun = asun.drop(columns=['Unnamed: 36_level_0_Unnamed: 36_level_1'])
selected_columns = [
    'Unnamed: 1_level_0_School',
    'Ratings_ORtg', 'Ratings_DRtg',
    'Per Game_FG', 'Per Game_FGA', 'Per Game_FG%',
    'Per Game_3P', 'Per Game_3PA', 'Per Game_3P%',
    'Per Game_eFG%', 'Per Game_FT', 'Per Game_FTA', 'Per Game_FT%',
    'Per Game_ORB', 'Per Game_TRB', 'Per Game_AST',
    'Per Game_STL', 'Per Game_BLK', 'Per Game_TOV', 'Per Game_PF',
    'Advanced_Pace', 'Conf %'
]

asun = asun[selected_columns]

# Rename 'Unnamed: 1_level_0_School' to 'School'
asun = asun.rename(columns={'Unnamed: 1_level_0_School': 'School'})

# Rename other columns by dropping the prefix before the first underscore
asun.columns = [
    'School' if col == 'School' else col.split('_', 1)[1] if '_' in col else col
    for col in asun.columns
]
asun.loc[:, 'School'] = asun['School'] + ' 2024'



asun_2024 = asun



url = 'https://www.sports-reference.com/cbb/conferences/atlantic-sun/men/2024.html'
tables = pd.read_html(url)
asun = tables[2]
asun.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in asun.columns]
asun['Conf %'] = asun['Conf._W'] / (asun['Conf._W'] + asun['Conf._L'])
asun = asun.drop(columns=['Unnamed: 36_level_0_Unnamed: 36_level_1'])
selected_columns = [
    'Unnamed: 1_level_0_School',
    'Ratings_ORtg', 'Ratings_DRtg',
    'Per Game_FG', 'Per Game_FGA', 'Per Game_FG%',
    'Per Game_3P', 'Per Game_3PA', 'Per Game_3P%',
    'Per Game_eFG%', 'Per Game_FT', 'Per Game_FTA', 'Per Game_FT%',
    'Per Game_ORB', 'Per Game_TRB', 'Per Game_AST',
    'Per Game_STL', 'Per Game_BLK', 'Per Game_TOV', 'Per Game_PF',
    'Advanced_Pace', 'Conf %'
]

asun = asun[selected_columns]

# Rename 'Unnamed: 1_level_0_School' to 'School'
asun = asun.rename(columns={'Unnamed: 1_level_0_School': 'School'})

# Rename other columns by dropping the prefix before the first underscore
asun.columns = [
    'School' if col == 'School' else col.split('_', 1)[1] if '_' in col else col
    for col in asun.columns
]
asun.loc[:, 'School'] = asun['School'] + ' 2023'


# In[16]:


asun_2023 = asun


# In[18]:


# Concatenate the two tables
combined_asun = pd.concat([asun_2024, asun_2023], ignore_index=True)

# Sort by 'Conf %' in descending order
combined_asun = combined_asun.sort_values(by='Conf %', ascending=False)

# Display the result
combined_asun.reset_index(drop=True, inplace=True)

# Select only numeric columns
numeric_cols = combined_asun.select_dtypes(include='number').columns

# Columns where lower is better
reverse_cols = ['DRtg', 'TOV', 'PF']

# Copy the original DataFrame
combined_asun_rescaled = combined_asun.copy()

# Define custom scaling function that accounts for mean
def adjusted_scale(col):
    col_min = col.min()
    col_max = col.max()
    col_mean = col.mean()
    # Prevent division by zero
    if col_max == col_min:
        return pd.Series([5] * len(col), index=col.index)  # Neutral value
    else:
        # Adjusted scaling with mean influence
        scaled = 1 + 9 * ((col - col_mean) / (col_max - col_min) + 0.5)
        return scaled.clip(1, 10)

# Apply adjusted scaling
for col in numeric_cols:
    if col in reverse_cols:
        combined_asun_rescaled[col] = 10 - adjusted_scale(combined_asun_rescaled[col]) + 1
    else:
        combined_asun_rescaled[col] = adjusted_scale(combined_asun_rescaled[col])



import plotly.graph_objects as go

# Define the feature columns (excluding the 'School' column)
features = combined_asun_rescaled.columns[1:]
num_vars = len(features)

# Create the figure
fig = go.Figure()

# Add a trace for each team
for i, row in combined_asun_rescaled.iterrows():
    team_name = row['School']
    values = row[1:].tolist()
    values += values[:1]  # Complete the circle

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=list(features) + [features[0]],  # close the loop
        name=team_name,
        line=dict(width=2)
    ))

# Update layout: Legend on the right
fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 10])
    ),
    showlegend=True,
    title='ASUN Teams Radar Chart 2023-25 (Rescaled)',
    width=1000,
    height=800,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.01,  # Move legend closer (try 1.01 or 1.02)
        font=dict(size=12),
        bordercolor="Black",
        borderwidth=1,
        bgcolor="white"
    )
)


# Set up the heatmap figure
plt.figure(figsize=(14, 8))

# Set index to 'School' for better row labels
heatmap_data = combined_asun_rescaled.set_index('School')

# Create the heatmap
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, vmin=1, vmax=10, linewidths=0.5, linecolor='gray')

plt.title('ASUN Stats (Scale: 1-10)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- STREAMLIT APP ---

st.set_page_config(layout="wide")

# Optional CSS to reduce padding
st.markdown(
    """
    <style>
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("ASUN Basketball Insights")

tab1, tab2, tab3, tab4 = st.tabs([
    "Radar Chart",
    "Heatmap", 
    "W/L Stats Comparison", 
    "Play-By-Play Analysis"
])


with tab1:
    st.markdown("""
    ### Understanding the Radar Chart
    
    This radar chart visualizes and compares the performance of all ASUN Conference basketball teams across the 2023-24 and 2024-25 seasons.  
    Each statistical category is rescaled to a 1–10 range, where **higher values are better** (10 being best, 1 being worst).  
    For most stats, higher numbers are closer to 10. However, for **DRtg** (Defensive Rating), **TOV** (Turnovers), and **PF** (Personal Fouls), 
    **lower values are better**—so teams with lower numbers in these categories score closer to 10.
    
    **How to use the chart:**  
    - Each colored shape represents a team.  
    - Hover over the lines to see each team's value for that stat.  
    - Compare the size and shape: a more "filled out" shape means stronger performance across more categories.
    
    ---
    **Stat Glossary:**  
    - **ORtg (Offensive Rating):** Estimates the number of points a team scores per 100 possessions. Higher is better.
    - **DRtg (Defensive Rating):** Estimates the number of points a team allows per 100 possessions. Lower is better.
    """)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    ### Understanding the Heatmap
    
    Use the controls below to select specific teams or statistics to focus the heatmap.  
    This interactive heatmap shows scaled values (1–10) for each stat, for each team.  
    **Red** cells indicate high values, and **blue** cells indicate low values in each category.
    
    **How to use the heatmap:**  
    - Use the multi-select boxes to customize which teams and stats you want to compare.
    - Hover over a cell to see the exact value.
    """)
    # Interactive controls
    all_teams = heatmap_data.index.tolist()
    all_stats = heatmap_data.columns.tolist()
    selected_teams = st.multiselect("Select teams to display:", all_teams, default=all_teams)
    selected_stats = st.multiselect("Select stats to display:", all_stats, default=all_stats)
    filtered_data = heatmap_data.loc[selected_teams, selected_stats]
    fig2, ax = plt.subplots(figsize=(max(8, len(selected_stats)*1.2), max(4, len(selected_teams)*0.5)))
    sns.heatmap(filtered_data, cmap='coolwarm', annot=True, vmin=1, vmax=10, linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_title('ASUN Stats (Scale: 1-10)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


with tab3:
    def fetch_and_process_team_stats(url, table_idx):
        try:
            tables = pd.read_html(url)
            df = tables[table_idx]
            # The Austin Peay logic here—adapt column names if needed for each site!
            df[['Score', 'Opp Score']] = df['Score'].str.split('-', expand=True).astype(int)
            df[['PF', 'Opp PF']] = df['PF'].str.split('/', expand=True).astype(int)
            df[['TO', 'Opp TO']] = df['TO'].str.split('/', expand=True).astype(int)
            df[['BLK', 'Opp BLK']] = df['BLK'].str.split('/', expand=True).astype(int)
            df[['STL', 'Opp STL']] = df['STL'].str.split('/', expand=True).astype(int)
            df[['FT PCT', 'Opp FT PCT']] = df['FT PCT'].str.split('-', expand=True).astype(float)
            df[['FG PCT', 'Opp FG PCT']] = df['FG PCT'].str.split('/', expand=True).astype(float)
            df[['3FG PCT', 'Opp 3FG PCT']] = df['3FG PCT'].str.split('/', expand=True).astype(float)
                    
            fg_split = df['FG'].str.split('/', expand=True)
            df['FGM'] = fg_split[0].str.split('-', expand=True)[0].astype(int)
            df['FGA'] = fg_split[0].str.split('-', expand=True)[1].astype(int)
            df['Opp FGM'] = fg_split[1].str.split('-', expand=True)[0].astype(int)
            df['Opp FGA'] = fg_split[1].str.split('-', expand=True)[1].astype(int)
            fg_split = df['3FG'].str.split('/', expand=True)
            df['3FGM'] = fg_split[0].str.split('-', expand=True)[0].astype(int)
            df['3FGA'] = fg_split[0].str.split('-', expand=True)[1].astype(int)
            df['Opp 3FGM'] = fg_split[1].str.split('-', expand=True)[0].astype(int)
            df['Opp 3FGA'] = fg_split[1].str.split('-', expand=True)[1].astype(int)
            df.drop(columns=['FG', '3FG'], inplace=True)
            df[['AST', 'Opp AST']] = df['AST'].str.split('/', expand=True).astype(int)
            rb_split = df['RB'].str.split(' ').str[0]
            df['RB'] = rb_split.str.split('/').str[0].astype(int)
            df['Opp RB'] = rb_split.str.split('/').str[1].astype(int)
            fg_split = df['FT'].str.split('/', expand=True)
            df['FTM'] = fg_split[0].str.split('-', expand=True)[0].astype(int)
            df['FTA'] = fg_split[0].str.split('-', expand=True)[1].astype(int)
            df['Opp FTM'] = fg_split[1].str.split('-', expand=True)[0].astype(int)
            df['Opp FTA'] = fg_split[1].str.split('-', expand=True)[1].astype(int)
            df.drop(columns=['FT'], inplace=True)
            win_df = df[df['MAR'] > 0].copy()
            lose_df = df[df['MAR'] < 0].copy()
            win_avg = win_df.mean(numeric_only=True)
            lose_avg = lose_df.mean(numeric_only=True)
            comparison = pd.DataFrame({
                'Win Average': win_avg,
                'Lose Average': lose_avg
            })
            comparison['Difference'] = comparison['Win Average'] - comparison['Lose Average']
            return comparison
        except Exception as e:
            return f"Error processing stats: {e}"

    # Team URLs and (guessed) stat table indices
    team_urls = {
        "Austin Peay": ("https://letsgopeay.com/sports/mens-basketball/stats/2024-25", 7),
        "Lipscomb": ("https://lipscombsports.com/sports/mens-basketball/stats", 7),
        "North Alabama": ("https://roarlions.com/sports/mens-basketball/stats/2024-25", 7),
        "FGCU": ("https://fgcuathletics.com/sports/mens-basketball/stats", 7),
        "Jacksonville": ("https://judolphins.com/sports/mens-basketball/stats/2024-25", 7),
        "EKU": ("https://ekusports.com/sports/mens-basketball/stats/2024-25", 7),
        "Queens": ("https://queensathletics.com/sports/mens-basketball/stats/2024-2025", 7),
        "North Florida": ("https://unfospreys.com/sports/mens-basketball/stats/2024-25", 7),
        "Stetson": ("https://gohatters.com/sports/mens-basketball/stats", 7),
        "Central Arkansas": ("https://ucasports.com/sports/mens-basketball/stats/2024-25", 7),
        "Bellarmine": ("https://athletics.bellarmine.edu/sports/mens-basketball/stats", 7),
    }
    
    st.markdown("### W/L Stats Comparison")


    team_list = list(team_urls.keys())
    selected_team = st.radio("Select a Team", team_list, index=0)

    url, table_idx = team_urls[selected_team]
    comparison = fetch_and_process_team_stats(url, table_idx)

    if isinstance(comparison, str):
        st.error(comparison)
    else:
        # Safe percent handling (multiply by 100 for percent columns)
        for pct_col in ['FT PCT', 'Opp FT PCT', '3FG PCT', 'Opp 3FG PCT', 'FG PCT', 'Opp FG PCT']:
            if pct_col in comparison.index:
                for c in ['Win Average', 'Lose Average', 'Difference']:
                    if c in comparison.columns:
                        comparison.loc[pct_col, c] = float(comparison.loc[pct_col, c]) * 100

        # Format percent columns for display
        def add_percent(val):
            try:
                return f"{val:.2f}%" if isinstance(val, float) else val
            except:
                return val

        comparison_display = comparison.copy()
        for pct_col in ['FT PCT', 'Opp FT PCT', '3FG PCT', 'Opp 3FG PCT', 'FG PCT', 'Opp FG PCT']:
            if pct_col in comparison_display.index:
                for c in ['Win Average', 'Lose Average', 'Difference']:
                    if c in comparison_display.columns:
                        comparison_display.loc[pct_col, c] = add_percent(comparison_display.loc[pct_col, c])

        # Round all numbers to 2 decimals for display (except percent strings)
        def round_if_number(val):
            if isinstance(val, float):
                return round(val, 2)
            return val
        comparison_display = comparison_display.applymap(round_if_number)

        st.dataframe(comparison_display)

        # -- Interactive bar chart --
        comparison_stats = comparison.index.tolist()
        selected_stat = st.selectbox("Select stat to visualize:", comparison_stats)

        win_val = float(comparison.loc[selected_stat, "Win Average"])
        lose_val = float(comparison.loc[selected_stat, "Lose Average"])

        fig_bar = go.Figure(
            data=[
                go.Bar(name="Win Average", x=["Win"], y=[win_val], marker_color='seagreen', text=[f"{win_val:.2f}"], textposition='outside'),
                go.Bar(name="Lose Average", x=["Lose"], y=[lose_val], marker_color='crimson', text=[f"{lose_val:.2f}"], textposition='outside'),
            ]
        )
        fig_bar.update_layout(
            title=f"{selected_stat} (Win vs Lose)",
            yaxis_title=selected_stat,
            xaxis_title="Game Result",
            barmode='group',
            bargap=0.4
        )
        diff = win_val - lose_val
        fig_bar.add_annotation(
            x=0.5, y=max(win_val, lose_val),
            text=f"Difference: {diff:.2f}",
            showarrow=False,
            yshift=20,
            font=dict(size=14, color="black")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    


with tab4:
    st.header("Play-By-Play Analysis")

    url = st.text_input("Enter the URL of the play-by-play stats page:")

    if url:
        try:
            tables = pd.read_html(url)
            st.success(f"Loaded {len(tables)} tables from the URL.")

            # Guess play-by-play tables: look for those with 'Time' in the columns.
            pbp_tables = []
            for idx, table in enumerate(tables):
                table.columns = table.columns.astype(str).str.strip()
                if any('time' in str(col).lower() for col in table.columns):
                    pbp_tables.append((idx, table))
            st.write(f"Detected possible play-by-play tables at indices: {[idx for idx, _ in pbp_tables]}")
            table_indices = [idx for idx, _ in pbp_tables]
            if not pbp_tables:
                st.warning("No play-by-play tables detected automatically. Please try selecting manually.")
                for i, table in enumerate(tables):
                    st.write(f"Table {i}:")
                    st.dataframe(table.head())
                table_indices = list(range(len(tables)))

            # Let user select first and second half (or both) play-by-play tables
            idx1 = st.selectbox("First half play-by-play table index:", options=table_indices, index=0)
            idx2 = st.selectbox("Second half play-by-play table index:", options=table_indices, index=1 if len(table_indices) > 1 else 0)

            first_half_play_by_play = tables[idx1].copy()
            second_half_play_by_play = tables[idx2].copy()

            columns_to_drop = ["Play Team Indicator", "Game Score", "Team Indicator", "Play"]
            for df in [first_half_play_by_play, second_half_play_by_play]:
                df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
                df.columns = df.columns.astype(str).str.strip()
                # Standardize time column name
                time_col = None
                for col in df.columns:
                    if 'time' in str(col).lower():
                        time_col = col
                        break
                if time_col and time_col != 'Time Remaining':
                    df.rename(columns={time_col: 'Time Remaining'}, inplace=True)
                df['Time Remaining'] = df['Time Remaining'].replace('--', np.nan)
                df['Time Remaining'] = df['Time Remaining'].ffill()
                df['Time Remaining'] = df['Time Remaining'].astype(str).apply(lambda x: x.strip())

            def pad_time(t):
                if pd.isna(t): return t
                parts = str(t).split(":")
                if len(parts) == 2:
                    m, s = parts
                elif len(parts) == 1:
                    m, s = parts[0], "00"
                else:
                    return t
                return f"{int(m):02d}:{int(s):02d}"

            first_half_play_by_play['Time Remaining'] = first_half_play_by_play['Time Remaining'].apply(pad_time)
            second_half_play_by_play['Time Remaining'] = second_half_play_by_play['Time Remaining'].apply(pad_time)

            # Improved team column guessing: skip score columns, select columns with play text
            def guess_team_columns(df):
                exclude_cols = ['TIME REMAINING', 'AWAY TEAM SCORE', 'HOME TEAM SCORE']
                possible_team_cols = []
                for col in df.columns:
                    if col.upper() in exclude_cols:
                        continue
                    # check if any value in the column is a non-null, non-numeric string
                    sample_vals = df[col].dropna().astype(str).head(10)
                    if any(any(c.isalpha() for c in val) and not val.replace('.', '', 1).isdigit() for val in sample_vals):
                        possible_team_cols.append(col)
                return possible_team_cols[:2]  # return first two found

            team_cols = guess_team_columns(first_half_play_by_play)
            if len(team_cols) < 2:
                st.warning("Couldn't confidently detect both team columns. Please select them manually.")
                candidate_cols = [c for c in first_half_play_by_play.columns if c.upper() not in ['TIME REMAINING', 'AWAY TEAM SCORE', 'HOME TEAM SCORE']]
                team_cols = st.multiselect("Select the two team columns:", options=candidate_cols, default=None, max_selections=2)
            if len(team_cols) == 2:
                team_names = [str(c).strip().upper() for c in team_cols]
                st.info(f"Detected teams: {team_names}")
            else:
                st.error("Could not detect two team columns. Please check your table.")
                st.stop()

            # For first half: add 20 minutes (so 20:00 → 40:00, 0:00 → 20:00)
            def first_half_game_time_seconds(time_str):
                if pd.isna(time_str):
                    return None
                m, s = map(int, str(time_str).split(":"))
                game_m = m + 20  # shift to 40:00-20:00
                return game_m * 60 + s

            # For second half: use as is (20:00 → 20:00, 0:00 → 0:00)
            def second_half_game_time_seconds(time_str):
                if pd.isna(time_str):
                    return None
                m, s = map(int, str(time_str).split(":"))
                return m * 60 + s

            first_half_play_by_play['game_time_seconds'] = first_half_play_by_play['Time Remaining'].apply(first_half_game_time_seconds)
            second_half_play_by_play['game_time_seconds'] = second_half_play_by_play['Time Remaining'].apply(second_half_game_time_seconds)

            # Extract plays
            def extract_plays(df, team_col, team_name):
                rows = []
                for idx, row in df.iterrows():
                    play_text = str(row.get(team_col, "")).upper()
                    time = row['Time Remaining']
                    game_time = row['game_time_seconds']
                    for category in ['MISS', 'REBOUND', 'ASSIST', 'TURNOVER']:
                        if category == 'MISS':
                            if 'MISS FT' in play_text:
                                rows.append({'team': team_name, 'time': time, 'game_time_seconds': game_time, 'play_type': 'MISS', 'value': 0.5})
                            elif 'MISS' in play_text:
                                rows.append({'team': team_name, 'time': time, 'game_time_seconds': game_time, 'play_type': 'MISS', 'value': 1})
                        elif category in play_text:
                            rows.append({'team': team_name, 'time': time, 'game_time_seconds': game_time, 'play_type': category, 'value': 1})
                return rows

            team1, team2 = team_cols
            teamname1, teamname2 = team_names
            ap_rows_1 = extract_plays(first_half_play_by_play, team1, teamname1)
            ut_rows_1 = extract_plays(first_half_play_by_play, team2, teamname2)
            ap_rows_2 = extract_plays(second_half_play_by_play, team1, teamname1)
            ut_rows_2 = extract_plays(second_half_play_by_play, team2, teamname2)

            # Combine all plays
            events = pd.DataFrame(ap_rows_1 + ut_rows_1 + ap_rows_2 + ut_rows_2)
            events = events.dropna(subset=['game_time_seconds'])
            events['game_time_seconds'] = events['game_time_seconds'].astype(int)
            events = events.sort_values('game_time_seconds', ascending=False).reset_index(drop=True)

            def seconds_to_mmss(seconds):
                m, s = divmod(int(seconds), 60)
                return f"{m:02d}:{s:02d}"

            play_types = ['MISS', 'REBOUND', 'ASSIST', 'TURNOVER']
            teams = team_names
            time_points = sorted(events['game_time_seconds'].unique(), reverse=True)
            time_labels = [seconds_to_mmss(t) for t in time_points]

            st.markdown("### Interactive Cumulative Bar Chart")
            play_type = st.selectbox("Select play type to visualize:", play_types)

            frames = []
            for t in time_points:
                subset = events[(events['game_time_seconds'] >= t) & (events['play_type'] == play_type)]
                summary = (
                    subset.groupby('team')['value']
                    .sum()
                    .reindex(teams, fill_value=0)
                    .reset_index()
                )
                summary['game_time_seconds'] = t
                frames.append(summary)
            cum_df = pd.concat(frames, ignore_index=True)

            y_initial = [
                cum_df[(cum_df['team'] == team) & (cum_df['game_time_seconds'] == time_points[0])]['value'].values[0]
                for team in teams
            ]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=teams,
                        y=y_initial,
                        marker=dict(line=dict(width=1)),
                    )
                ]
            )

            steps = []
            for idx, t in enumerate(time_points):
                y_step = [
                    cum_df[(cum_df['team'] == team) & (cum_df['game_time_seconds'] == t)]['value'].values[0]
                    for team in teams
                ]
                step = dict(
                    method="update",
                    args=[{"y": [y_step]}],
                    label=time_labels[idx]
                )
                steps.append(step)

            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Time: "},
                pad={"t": 50},
                steps=steps
            )]

            fig.update_layout(
                sliders=sliders,
                barmode='group',
                title=f"Cumulative {play_type} Count Over Time (Full Game)",
                xaxis_title="Team",
                yaxis_title=f"Cumulative {play_type} Count",
                xaxis=dict(
                    tickmode='array',
                    tickvals=teams,
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading or processing the URL: {e}")
