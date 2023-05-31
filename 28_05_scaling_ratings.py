# -*- coding: utf-8 -*-
"""
Created on Sun May 28 20:09:28 2023

@author: User
"""
import pandas as pd

df_ratings = pd.read_csv("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\MEMO\\data_exploration\\output.csv")
#df_ratings.columns
#df_ratings.dtypes
#df_ratings.shape
overlap_df = df_ratings[df_ratings.duplicated(['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'])]
overlap_df.count()
df_ratings.count()
overlap_df.columns

#df_ratings_overlap = df_ratings.loc[df_ratings['Start Time (formatted)'] == df_ratings['End Time (formatted)']]

#df_ratings_overlap[df_ratings_overlap.loc[df_ratings_overlap['Annotator'] == 1]]

# overlapping df

df_ratings = overlap_df.filter(['Start Time (ms)', 'End Time (ms)', 'Group', 'Session', 'Annotator', 'Involvement'])
df_ratings.columns #columns needed: start time, end time, annotator, rating

df_ratings = df_ratings.groupby(['Start Time (ms)', 'End Time (ms)']).aggregate('count') ## combine start time and end time
df_ratings
#Step 1
df_average_rating = df_ratings.groupby(['Group', 'Session']).mean('Involvement')

#Step 2
df_factors1 = df_ratings.loc[df_ratings['Annotator'] == 1]/df_average_rating
df_factors2 = df_ratings.loc[df_ratings['Annotator'] == 2]/df_average_rating
df_factors3 = df_ratings.loc[df_ratings['Annotator'] == 3]/df_average_rating
df_factors4 = df_ratings.loc[df_ratings['Annotator'] == 4]/df_average_rating


