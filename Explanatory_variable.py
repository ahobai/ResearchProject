# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
df1 = pd.read_csv("profilicId.csv")
df2 = pd.read_excel("personal_info.xlsx")
#df3 = pd.read_excel("post_questionaire.xlsx", dtype='object')
df4 = pd.read_excel("MEMO_participant_group_assignment.xlsx")

merged_df = pd.merge(df1, df2, on='ProfilicID')
merged_df = pd.merge(merged_df, df4, on='ProfilicID')
#merged_df = pd.merge(merged_df, df3, on='ProfilicID')

filtered_df = merged_df[merged_df['ProfilicID'].isin(df1['ProfilicID'])]
filtered_df = filtered_df.dropna(subset=['GENDER', 'Demographic', 'age', 'online_meetings_experience'])
filtered_df['Group'] = filtered_df['group'].copy()


# Keep only the profilicId, group, online_meetings_experience, age,
# Covid-19_affected_group, Covid-19_affected_group_extra, Demographic,
# Perceived_group, GENDER.
final_data = filtered_df[['ProfilicID', 'Group', 'online_meetings_experience', 'age','Demographic', 'GENDER']]
final_data = final_data.drop_duplicates()

# =============================================================================
# PLOTS
# 
# # =============================================================================
# # CATEGORICAL DATA PLOTS
# # =============================================================================
# import seaborn as sns
# sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
# # plot the gender and demographics (categorical data)
# data = filtered_df.loc[filtered_df['group'] == 15]
# # Convert 'gender' column to categorical data type
# gender= data['GENDER']
# gender = pd.DataFrame(gender)
# 
# # Plot the gender counts using Seaborn
# sns.countplot(x='GENDER', data=gender).set(xlabel='gender in Group 15', ylabel='Count');
# 
# # =============================================================================
# # NUMERICAL DATA PLOTS
# # =============================================================================
# # Keep only the rows where the 'group' column has the value 5
# group_age = filtered_df.loc[filtered_df['group'] == 15]
# # plot the age histogram (numerical variables)
# sns.distplot(
#     group_age['age'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
# ).set(xlabel='Age in Group 15', ylabel='Count');
# 
# =============================================================================

# =============================================================================
# ONE HOT ENCODER
# =============================================================================
# data = final_data
# X_enc = data.copy()

# #ONEHOT ENCODING 
# X_enc = pd.get_dummies(X_enc, columns = ['GENDER', 'Demographic', 'online_meetings_experience'], drop_first=True)

# data = data.drop(['ProfilicID','Group', 'age', 'GENDER', 'Demographic', 'online_meetings_experience'], axis=1)

# final_data = pd.concat([data, X_enc], axis=1)



# =============================================================================
# LABEL ENCODER
# from sklearn.preprocessing import LabelEncoder
# label = LabelEncoder()
# 
# label.fit(filtered_df.GENDER)
# filtered_df.GENDER = label.transform(filtered_df.GENDER)
# print(filtered_df.GENDER.head)
# 
# label.fit(filtered_df.Demographic)
# filtered_df.Demographic = label.transform(filtered_df.Demographic)
# print(filtered_df.Demographic.head)
# 
# label.fit(filtered_df.online_meetings_experience)
# filtered_df.online_meetings_experience = label.transform(filtered_df.online_meetings_experience)
# =============================================================================

