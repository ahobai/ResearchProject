# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
df1 = pd.read_csv("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\profilicId.csv")
df2 = pd.read_excel("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\personal_info.xlsx")
#df3 = pd.read_excel("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\post_questionaire.xlsx", dtype='object')
df4 = pd.read_excel("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\MEMO_participant_group_assignment.xlsx")

merged_df = pd.merge(df1, df2, on='ProfilicID')
merged_df = pd.merge(merged_df, df4, on='ProfilicID')
#merged_df = pd.merge(merged_df, df3, on='ProfilicID')

filtered_df = merged_df[merged_df['ProfilicID'].isin(df2['ProfilicID'])]
filtered_df = filtered_df.dropna(subset=['GENDER', 'Demographic', 'age', 'online_meetings_experience'])


# Keep only the profilicId, group, online_meetings_experience, age,
# Covid-19_affected_group, Covid-19_affected_group_extra, Demographic,
# Perceived_group, GENDER.

filtered_df = filtered_df[['ProfilicID', 'group', 'online_meetings_experience', 'age','Demographic', 'GENDER']]
data = filtered_df
X_enc = data.copy()

#ONEHOT ENCODING 
X_enc = pd.get_dummies(X_enc, columns = ['GENDER', 'Demographic', 'online_meetings_experience'])

data = data.drop(['GENDER', 'Demographic', 'online_meetings_experience'], axis=1)

final_data = pd.concat([data, X_enc], axis=1)



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


#from sklearn.model_selection import train_test_split
#output_df = pd.read_excel("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\MEMO\\data_exploration\\output.xlsx")

#x_lin = filtered_df
#y_lin = output_df




print("Nr rows before dropping withdrawn: {}".format(df.shape[0]))

df_w_withdrawn = df[df.group != 'withdrawn']

print("Nr rows after dropping withdrawn: {}".format(df_w_withdrawn.shape[0]))

df_w_withdrawn.columns


print()


