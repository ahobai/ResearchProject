# -*- coding: utf-8 -*-
"""
Created on Sun May 28 20:09:28 2023

@author: User
"""
import pandas as pd

df_ratings = pd.read_csv("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\output.csv")

# overlapping df
df_ratings = df_ratings[df_ratings.duplicated(subset=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'], keep=False)]
df_ratings = df_ratings[['Start Time (ms)', 'End Time (ms)', 'Group', 'Session', 'Annotator', 'Involvement']]
#columns needed: start time, end time, annotator, rating

df_ratings['Group_session'] = (df_ratings['Group'].astype(str) + df_ratings['Session'].astype(str)).astype(int)
data = df_ratings.loc[df_ratings['Group_session'] == 51]

import seaborn as sns

sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    data['Involvement'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Group Involvement (Group: 5, Session: 1)', ylabel='Count');





# import random
# rnd = random.randint(0, 41)
# df = df_ratings.copy()
# df = df[df['Group_session'] == rnd]

# # LABEL ENCODER
# from sklearn.preprocessing import LabelEncoder
# label = LabelEncoder()

# label.fit(df.Group_session)
# df.Group_session = label.transform(df.Group_session)

annotator1 = df_ratings[df_ratings['Annotator'] == 1]
annotator2 = df_ratings[df_ratings['Annotator'] == 2]
annotator3 = df_ratings[df_ratings['Annotator'] == 3]
annotator4 = df_ratings[df_ratings['Annotator'] == 4]

# =============================================================================
# # average for group 2 for each annotator
# 
# r1q1 = annotator1[annotator1['Group'] == 2]
# r1q1 = r1q1['Involvement']
# avg1 = sum(r1q1)/len(r1q1)
# 
# r2q1 = annotator2[annotator2['Group'] == 2]
# r2q1 = r2q1['Involvement']
# avg2 = sum(r2q1)/len(r2q1)
# 
# r3q1 = annotator3[annotator3['Group'] == 2]
# r3q1 = r3q1['Involvement']
# avg3 = sum(r3q1)/len(r3q1)
# 
# r4q1 = annotator4[annotator4['Group'] == 2]
# r4q1 = r4q1['Involvement']
# avg4 = sum(r4q1)/len(r4q1)=
# =============================================================================

# involvement per annotator for each group
r1 = []
r2 = []
r3 = []
r4 = []

for x in range(2, 16):
    r1q1 = annotator1[annotator1['Group'] == x]
    r1q1 = r1q1['Involvement']
    avg1 = round(sum(r1q1)/len(r1q1))
    r1.append(avg1)

    r2q1 = annotator2[annotator2['Group'] == x]
    r2q1 = r2q1['Involvement']
    avg2 = round(sum(r2q1)/len(r2q1))
    r2.append(avg2)
    
    r3q1 = annotator3[annotator3['Group'] == x]
    r3q1 = r3q1['Involvement']
    avg3 = round(sum(r3q1)/len(r3q1))
    r3.append(avg3)
    
    r4q1 = annotator4[annotator4['Group'] == x]
    r4q1 = r4q1['Involvement']
    avg4 = round(sum(r4q1)/len(r4q1))
    r4.append(avg4)


# =============================================================================
# COHEN KAPPA SCORE
# =============================================================================

from sklearn.metrics import cohen_kappa_score
# get the overlap between each two annotators
overlap12 = pd.merge(annotator1, annotator2, on=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'])
kappa12 = cohen_kappa_score(overlap12['Involvement_x'], overlap12['Involvement_y'])

overlap13 = pd.merge(annotator1, annotator3, on=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'])
kappa13 = cohen_kappa_score(overlap13['Involvement_x'], overlap13['Involvement_y'])

overlap14 = pd.merge(annotator1, annotator4, on=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'])
kappa14 = cohen_kappa_score(overlap14['Involvement_x'], overlap14['Involvement_y'])

overlap23 = pd.merge(annotator2, annotator3, on=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'])
kappa23 = cohen_kappa_score(overlap23['Involvement_x'], overlap23['Involvement_y'])

overlap24 = pd.merge(annotator2, annotator4, on=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'])
kappa24 = cohen_kappa_score(overlap24['Involvement_x'], overlap24['Involvement_y'])

overlap34 = pd.merge(annotator4, annotator3, on=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'])
kappa34 = cohen_kappa_score(overlap34['Involvement_x'], overlap34['Involvement_y'])


# kappa score for the inter-annotator agreement between the averages of each group
kappa_r1r2 = cohen_kappa_score(r1, r2)
kappa_r1r3 = cohen_kappa_score(r1, r3)
kappa_r1r4 = cohen_kappa_score(r1, r4)
kappa_r2r3 = cohen_kappa_score(r2, r3)
kappa_r2r4 = cohen_kappa_score(r2, r4)
kappa_r3r4 = cohen_kappa_score(r3, r4)


# =============================================================================
# OBSERVED AGREEMENT
# =============================================================================

# Calculate observed agreement
total_items = len(r2)
agreement_count = sum(1 for i in range(total_items) if overlap12['Involvement_x'][i] == overlap12['Involvement_y'][i])
observed_agreement12 = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if overlap13['Involvement_x'][i] == overlap13['Involvement_y'][i])
observed_agreement13 = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if overlap14['Involvement_x'][i] == overlap14['Involvement_y'][i])
observed_agreement14 = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if overlap23['Involvement_x'][i] == overlap23['Involvement_y'][i])
observed_agreement23 = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if overlap24['Involvement_x'][i] == overlap24['Involvement_y'][i])
observed_agreement24 = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if overlap34['Involvement_x'][i] == overlap34['Involvement_y'][i])
observed_agreement34 = agreement_count / total_items



# Calculate observed agreement between the averages of the groups
total_items = len(r2)
agreement_count = sum(1 for i in range(total_items) if r1[i] == r2[i])
observed_agreement12_avg = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if r1[i] == r3[i])
observed_agreement13_avg = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if r1[i] == r4[i])
observed_agreement14_avg = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if r2[i] == r3[i])
observed_agreement23_avg = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if r2[i] == r4[i])
observed_agreement24_avg = agreement_count / total_items

agreement_count = sum(1 for i in range(total_items) if r3[i] == r4[i])
observed_agreement34_avg = agreement_count / total_items


# =============================================================================
# ICC INTER-ANNOTATOR AGREEMENT
# =============================================================================

df_ratings['Group_session'] = "g" + df_ratings['Group'].astype(str) +"-s"+ df_ratings['Session'].astype(str)
data = df_ratings[['Group_session', 'Annotator', 'Involvement']]
data12 = data.loc[data['Annotator'].isin([1, 2])]
data12 = data12.loc[data12.duplicated()]
data13 = data.loc[data['Annotator'].isin([1, 3])]
data13 = data13.loc[data13.duplicated()]
data14 = data.loc[data['Annotator'].isin([1, 4])]
data14 = data14.loc[data14.duplicated()]
data23 = data.loc[data['Annotator'].isin([2, 3])]
data23 = data23.loc[data23.duplicated()]
data24 = data.loc[data['Annotator'].isin([2, 4])]
data24 = data24.loc[data24.duplicated()]
data34 = data.loc[data['Annotator'].isin([3, 4])]
data34 = data34.loc[data34.duplicated()]

import pingouin as pg

# Calculate ICC
icc_result12 = pg.intraclass_corr(data=data12, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result13 = pg.intraclass_corr(data=data13, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result14 = pg.intraclass_corr(data=data14, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result23 = pg.intraclass_corr(data=data23, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result24 = pg.intraclass_corr(data=data24, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result34 = pg.intraclass_corr(data=data34, targets='Group_session', raters='Annotator', ratings='Involvement')



# =============================================================================
# Build the dataset, dealing with the overlapped segments
# annotated by different annotators by either keeping that value if it is equal
# or keeping the weighted mean (weights = annotators) otherwise
# =============================================================================

# Group the dataframe by the columns that contain duplicate values
df = df_ratings[['Start Time (ms)', 'End Time (ms)', 'Group', 'Session', 'Annotator', 'Involvement']] # has 3247 rows
groups = df.groupby(['Start Time (ms)', 'End Time (ms)', 'Group', 'Session']) # are 1599 duplicates
duplicates = []

# Iterate over each group
for (start, end, group, session), annotations in groups:
    if len(annotations) > 1:  # Check if there are duplicate rows
        duplicates.append(annotations)
        
        if not annotations.empty:
            # Check if the duplicated rows are not equal
            if not annotations.equals(annotations.iloc[0]):
                # Calculate the weighted mean
                weighted_mean = (annotations['Involvement'] * annotations['Annotator']).sum() / annotations['Annotator'].sum()
                
                # Update all duplicated rows with the weighted mean
                df.loc[annotations.index, 'Involvement'] = round(weighted_mean)
            

final_df = df.drop_duplicates(subset=df.columns.difference(['Annotator'])) # has 1599 rows, ????so what happens to the extra 49 rows



# =============================================================================
# NORMALIZATION OF THE DATA SET
# =============================================================================

# =============================================================================
# Calculate the z-score of the involvement for z-score standarsization 
# =============================================================================

from sklearn.preprocessing import StandardScaler

# Apply Z-score standardization to the 'Involvement' column
scaler = StandardScaler()
final_df['Involvement_ZScore'] = scaler.fit_transform(final_df[['Involvement']])
# Z-score standardization
# final_df['Normalized_Involvement'] = (final_df['Involvement'] - final_df['Involvement'].mean()) / final_df['Involvement'].std()


# =============================================================================
# Calculate the log tranformation of the involvement
# =============================================================================

import numpy as np

# Apply log transformation to the 'Involvement' column
final_df['Involvement_Log'] = np.log(final_df['Involvement'])




