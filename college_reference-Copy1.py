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

# Update layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 10])
    ),
    showlegend=True,
    title='ASUN Teams Radar Chart 2023-25 (Rescaled)',
    width=1000,    # Increase width
    height=800     # Increase height

)

fig.show()
fig.write_html("asun_radar_chart.html")


# In[34]:


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


# In[36]:


# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Example: Replace this with your actual figure and data
st.title("ASUN Basketball Insights")

st.plotly_chart(fig)


# In[ ]:




