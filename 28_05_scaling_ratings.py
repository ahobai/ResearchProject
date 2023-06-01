# -*- coding: utf-8 -*-
"""
Created on Sun May 28 20:09:28 2023

@author: User
"""
import pandas as pd

df_ratings = pd.read_csv("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\output.csv")

# overlapping df
overlap_df = df_ratings[df_ratings.duplicated(subset=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'], keep=False)]
df_ratings = overlap_df[['Start Time (ms)', 'End Time (ms)', 'Group', 'Session', 'Annotator', 'Involvement']]
df_ratings.columns #columns needed: start time, end time, annotator, rating


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
# avg4 = sum(r4q1)/len(r4q1)
# 
# 
# #average for group 3 for each annotator
# 
# r1q1 = annotator1[annotator1['Group'] == 3]
# r1q1 = r1q1['Involvement']
# avg1 = sum(r1q1)/len(r1q1)
# 
# r2q1 = annotator2[annotator2['Group'] == 3]
# r2q1 = r2q1['Involvement']
# avg2 = sum(r2q1)/len(r2q1)
# 
# r3q1 = annotator3[annotator3['Group'] == 3]
# r3q1 = r3q1['Involvement']
# avg3 = sum(r3q1)/len(r3q1)
# 
# r4q1 = annotator4[annotator4['Group'] == 3]
# r4q1 = r4q1['Involvement']
# avg4 = sum(r4q1)/len(r4q1)
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

raters = pd.DataFrame({'r1': r1, 'r4': r4, 'r2': r2, 'r3': r3})

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















df_ratings = df_ratings.groupby(['Start Time (ms)', 'End Time (ms)']) ## combine start time and end time



#Step 1
df_average_rating = df_ratings.groupby(['Group', 'Session']).mean('Involvement')

#Step 2
df_factors1 = df_ratings.loc[df_ratings['Annotator'] == 1]/df_average_rating
df_factors2 = df_ratings.loc[df_ratings['Annotator'] == 2]/df_average_rating
df_factors3 = df_ratings.loc[df_ratings['Annotator'] == 3]/df_average_rating
df_factors4 = df_ratings.loc[df_ratings['Annotator'] == 4]/df_average_rating


