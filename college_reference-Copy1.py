#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[9]:


# In[12]:


asun_2024 = asun


# In[14]:


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


# In[20]:


# In[22]:


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


# In[32]:


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

tab1, tab2 = st.tabs(["Radar Chart", "Heatmap"])

with tab1:
    st.plotly_chart(fig, use_container_width=True)

with tab2:
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
